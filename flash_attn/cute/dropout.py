"""Attention dropout keyed by logical coordinates.

One 32-bit hash covers the 2x2 group ``(q >> 1, k >> 1)``::

    h = fmix32((q >> 1) * PHI_Q ^ (k >> 1) * PHI_K ^ fmix32((b * H + h) ^ key))

and element ``(q, k)`` is kept iff byte ``2 * (q & 1) + (k & 1)`` of ``h`` is at
least ``t = round(p * 256)``. Kept probabilities are scaled by
``256 / (256 - t)``, so the effective drop probability is ``t / 256``. The row
and column terms are separable: the forward hoists the row term (one row per
thread) and the backward hoists the column term (one KV row per thread), and
each hash serves two elements in both directions.

The mask depends only on logical coordinates, so forward and backward kernels
regenerate it regardless of tiling, 2CTA, PackGQA, SplitKV or KV slot order.
``q_positions`` / ``kv_positions`` remap kernel-local rows / columns to the
coordinates the mask is keyed by (context parallelism passes global token
positions). Within every ``POSITION_BLOCK``-aligned block of local indices they
must ascend by one from an even position: the kernels read one position per
block and offset it, which also keeps aligned pairs aligned.

``rng_state`` is a device ``int64[2]`` tensor ``(seed, offset)`` read by the
kernel, so CUDA Graph replays can advance it without recapturing.
"""

from __future__ import annotations

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32

from flash_attn.cute import utils

_M1 = 0x85EBCA6B
_M2 = 0xC2B2AE35
_PHI_Q = 0x9E3779B9
_PHI_K = 0x85EBCA77
_MASK32 = 0xFFFFFFFF
# Local rows / columns of one block share a single position read (see module docstring).
POSITION_BLOCK = 128


class DropoutArgs(NamedTuple):
    rng_state: cute.Tensor
    threshold: Uint32
    rp: Float32
    num_heads_q: Int32
    q_positions: cute.Tensor | None = None
    kv_positions: cute.Tensor | None = None


class DropoutTensors(NamedTuple):
    """Host-side dropout inputs; ``p`` is the drop probability in ``[0, 1)``."""

    p: float
    rng_state: torch.Tensor
    q_positions: torch.Tensor | None = None
    kv_positions: torch.Tensor | None = None


def next_rng_state(device: torch.device) -> torch.Tensor:
    """Draw ``(seed, offset)`` from the CUDA generator and advance its offset."""
    gen = torch.cuda.default_generators[device.index if device.index is not None else 0]
    seed, offset = gen.initial_seed(), gen.get_offset()
    gen.set_offset(offset + 4)
    seed = seed - (1 << 64) if seed >= 1 << 63 else seed
    return torch.tensor([seed, offset], dtype=torch.int64, device=device)


def dropout_threshold(p: float) -> int:
    """Byte threshold ``t``; the effective drop probability is ``t / 256``."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout p must be in [0, 1), got {p}")
    return min(round(p * 256), 255)


def dropout_scale(p: float) -> float:
    """Unbiased rescale ``1 / (1 - t / 256)`` for kept elements."""
    return 256.0 / (256 - dropout_threshold(p))


def validate_dropout(dropout: DropoutTensors, device: torch.device) -> None:
    state = dropout.rng_state
    if state.dtype != torch.int64 or state.shape != (2,) or state.device != device:
        raise ValueError("dropout rng_state must be a device int64 tensor [seed, offset]")
    for name in ("q_positions", "kv_positions"):
        positions = getattr(dropout, name)
        if positions is not None and (
            positions.dtype != torch.int32
            or positions.ndim != 1
            or positions.device != device
            or positions.numel() % 2
        ):
            raise ValueError(f"dropout {name} must be a 1-D even-length int32 tensor on {device}")
    dropout_threshold(dropout.p)


def check_block_contiguous(positions: torch.Tensor) -> None:
    """Positions must ascend by one within each ``POSITION_BLOCK``-aligned block, from an
    even position (syncs the device)."""
    idx = torch.arange(positions.numel(), device=positions.device)
    offset = idx % POSITION_BLOCK
    first = positions.long()[idx - offset]
    if not bool(((positions.long() == first + offset) & (first % 2 == 0)).all()):
        raise ValueError(
            f"dropout positions must ascend by one within each {POSITION_BLOCK}-token block "
            "from an even position"
        )


def to_dropout_args(
    dropout: DropoutTensors | None, num_heads_q: int, to_tensor, *, for_compile: bool
) -> DropoutArgs | None:
    """Compile args wrap scalars in DSL types; call args pass plain Python scalars."""
    if dropout is None:
        return None
    threshold, rp = dropout_threshold(dropout.p), dropout_scale(dropout.p)
    return DropoutArgs(
        rng_state=to_tensor(dropout.rng_state),
        threshold=Uint32(threshold) if for_compile else threshold,
        rp=Float32(rp) if for_compile else rp,
        num_heads_q=Int32(num_heads_q) if for_compile else num_heads_q,
        q_positions=None if dropout.q_positions is None else to_tensor(dropout.q_positions),
        kv_positions=None if dropout.kv_positions is None else to_tensor(dropout.kv_positions),
    )


def dropout_compile_key(dropout: DropoutTensors | None) -> tuple:
    if dropout is None:
        return (None,)
    return (True, dropout.q_positions is not None, dropout.kv_positions is not None)


@cute.jit
def fmix32(h: Uint32) -> Uint32:
    h = h ^ (h >> Uint32(16))
    h = h * Uint32(_M1)
    h = h ^ (h >> Uint32(13))
    h = h * Uint32(_M2)
    return h ^ (h >> Uint32(16))


@cute.jit
def dropout_key(rng_state: cute.Tensor) -> Uint32:
    seed = rng_state[0]
    offset = rng_state[1]
    key = fmix32(Uint32(offset >> 32))
    key = fmix32(Uint32(offset & _MASK32) ^ key)
    key = fmix32(Uint32(seed >> 32) ^ key)
    return fmix32(Uint32(seed & _MASK32) ^ key)


@cute.jit
def block_position(positions: cute.Tensor, i: Int32) -> Int32:
    """Position of local index ``i``, read through the first entry of its block."""
    start = cutlass.min(i, cute.size(positions) - 1) & -POSITION_BLOCK
    return positions[start] + (i - start)


class DropoutCtx:
    """Per-kernel dropout constants; positions remap local rows / columns."""

    def __init__(self, args: DropoutArgs):
        self.key = dropout_key(args.rng_state)
        self.threshold_top = args.threshold << Uint32(24)
        # Per halfword: 0x8000 + (byte - t) * 128 stays in [128, 0xFF80] and has
        # bit 15 set iff byte >= t, so two bytes compare without carries.
        self.keep_bias = (Uint32(0x8000) - (args.threshold << Uint32(7))) * Uint32(0x10001)
        self.rp = args.rp
        self.num_heads_q = args.num_heads_q
        self.q_positions = args.q_positions
        self.kv_positions = args.kv_positions

    @cute.jit
    def global_q(self, q: Int32) -> Int32:
        """Local row -> keyed row; rows past the end extend the last block."""
        if cutlass.const_expr(self.q_positions is not None):
            q = block_position(self.q_positions, q)
        return q

    @cute.jit
    def global_k(self, k: Int32) -> Int32:
        if cutlass.const_expr(self.kv_positions is not None):
            k = block_position(self.kv_positions, k)
        return k

    @cute.jit
    def prefetch_k(self, k: Int32):
        """Prefetch the position read by ``global_k(k)`` into L1 (``k >= 0``)."""
        if cutlass.const_expr(self.kv_positions is not None):
            start = cutlass.min(k, cute.size(self.kv_positions) - 1) & -POSITION_BLOCK
            cute.arch.prefetch((self.kv_positions.iterator + start).llvm_ptr, cache_level="L1")

    @cute.jit
    def head_key(self, batch_idx: Int32, head_idx: Int32) -> Uint32:
        return fmix32(Uint32(batch_idx * self.num_heads_q + head_idx) ^ self.key)

    @cute.jit
    def q_term(self, q: Int32) -> Uint32:
        return self.pair_q_term(q >> 1)

    @cute.jit
    def pair_q_term(self, pair: Int32) -> Uint32:
        """``q_term`` of row pair ``pair = q >> 1``."""
        return Uint32(pair) * Uint32(_PHI_Q)

    @cute.jit
    def k_term(self, k: Int32) -> Uint32:
        return self.pair_k_term(k >> 1)

    @cute.jit
    def pair_k_term(self, pair: Int32) -> Uint32:
        """``k_term`` of column pair ``pair = k >> 1``."""
        return Uint32(pair) * Uint32(_PHI_K)

    @cute.jit
    def keep_byte(self, h: Uint32, byte: Int32) -> cutlass.Boolean:
        # Byte ``byte`` moved to the top: the lower bits cannot change the comparison.
        return (h << Uint32(24 - byte * 8)) >= self.threshold_top

    @cute.jit
    def row_byte_sel(self, q: Int32) -> Uint32:
        """``prmt`` selector moving row ``q``'s two bytes to the low byte of each halfword."""
        return Uint32(0x4140) + Uint32(q & 1) * Uint32(0x0202)

    @cute.jit
    def pair_hash(self, row_term: Uint32, k_term: Uint32) -> Uint32:
        """Hash of the 2x2 group; ``row_term = q_term(q) ^ head_key``."""
        return fmix32(row_term ^ k_term)

    @cute.jit
    def keep_halves(self, h: Uint32, byte_sel: Uint32) -> Uint32:
        """Bit 15 of each halfword is set iff the byte of ``h`` that ``byte_sel`` moves there is kept."""
        return utils.prmt_b32(h, Uint32(0), byte_sel) * Uint32(128) + self.keep_bias

    @cute.jit
    def keep_mask(self, h: Uint32, byte_sel: Uint32) -> Uint32:
        """bf16x2 AND-mask from the two bytes of ``h`` picked by ``byte_sel = row_byte_sel(q)``."""
        return utils.prmt_b32(self.keep_halves(h, byte_sel), Uint32(0), Uint32(0xBB99))

    @cute.jit
    def shared_pair_hashes(
        self, lane_term: Uint32, own_term: Uint32, pair_lanes: cutlass.Constexpr[int]
    ):
        """Hashes of pairs a and b for two lanes ``pair_lanes`` apart that share every 2x2
        hash: rows ``q`` / ``q ^ 1`` in the forward, KV rows ``k`` / ``k ^ 1`` in the backward.

        ``lane_term`` is the lane's fixed term; the even lane hashes pair a and the odd
        lane pair b (``own_term``). An idx shuffle with segment mask ``~pair_lanes`` reads
        lane ``(lane & ~pair_lanes) | offset``, so offsets 0 / ``pair_lanes`` fetch the
        even / odd lane's hash.
        """
        own = fmix32(lane_term ^ own_term)
        seg = ((31 ^ pair_lanes) << 8) | 31
        h_a = cute.arch.shuffle_sync(own, 0, mask_and_clamp=seg)
        h_b = cute.arch.shuffle_sync(own, pair_lanes, mask_and_clamp=seg)
        return h_a, h_b

    @cute.jit
    def col_keep_pair(self, h: Uint32, k: Int32):
        """Keep of ``(q0, k)`` and ``(q0 + 1, k)`` from the hash ``h`` of their 2x2 group."""
        byte = k & 1
        return self.keep_byte(h, byte), self.keep_byte(h, byte + 2)


# --------------------------------------------------------------------------- #
# Torch reference (bit-identical to the kernel predicate).
# --------------------------------------------------------------------------- #


def _fmix32_torch(h: torch.Tensor) -> torch.Tensor:
    h = h ^ (h >> 16)
    h = (h * _M1) & _MASK32
    h = h ^ (h >> 13)
    h = (h * _M2) & _MASK32
    return h ^ (h >> 16)


def dropout_keep_mask(
    rng_state: torch.Tensor,
    p: float,
    num_heads_q: int,
    batch_idx: int,
    head_idx: int,
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Bool ``[len(q), len(k)]`` keep mask for logical rows ``q`` and columns ``k``."""
    seed, offset = (int(v) & ((1 << 64) - 1) for v in rng_state.tolist())
    device = q.device

    def mix(value: int | torch.Tensor) -> torch.Tensor:
        return _fmix32_torch(torch.as_tensor(value, dtype=torch.int64, device=device))

    key = mix(offset >> 32)
    key = mix((offset & _MASK32) ^ key)
    key = mix((seed >> 32) ^ key)
    key = mix((seed & _MASK32) ^ key)
    head_key = mix((batch_idx * num_heads_q + head_idx) ^ key)
    q, k = q.to(torch.int64), k.to(torch.int64)
    q_term = ((q >> 1) * _PHI_Q) & _MASK32
    k_term = ((k >> 1) * _PHI_K) & _MASK32
    h = _fmix32_torch(q_term[:, None] ^ k_term[None, :] ^ head_key)
    byte = 2 * (q[:, None] & 1) + (k[None, :] & 1)
    return ((h >> (8 * byte)) & 0xFF) >= dropout_threshold(p)
