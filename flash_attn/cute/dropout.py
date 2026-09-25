"""Attention dropout keyed by logical coordinates.

Element ``(b, h, q, k)`` is kept iff ``hash(seed, offset, b * H + h, q, k) >=
threshold`` where ``threshold = floor(p * 2**32)``. Kept probabilities are
scaled by ``1 / (1 - p)``. The hash only depends on the logical coordinates, so
forward and backward kernels regenerate the same mask regardless of tiling,
2CTA, PackGQA, SplitKV or KV slot order. ``q_positions`` / ``kv_positions``
remap kernel-local rows / columns to the coordinates the mask is keyed by
(context parallelism passes global token positions).

``rng_state`` is a device ``int64[2]`` tensor ``(seed, offset)`` read by the
kernel, so CUDA Graph replays can advance it without recapturing.
"""

from __future__ import annotations

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32

_M1 = 0x85EBCA6B
_M2 = 0xC2B2AE35
_GOLDEN = 0x9E3779B9
_MASK32 = 0xFFFFFFFF


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
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout p must be in [0, 1), got {p}")
    return min(int(p * 2**32), _MASK32)


def validate_dropout(dropout: DropoutTensors, device: torch.device) -> None:
    state = dropout.rng_state
    if state.dtype != torch.int64 or state.shape != (2,) or state.device != device:
        raise ValueError("dropout rng_state must be a device int64 tensor [seed, offset]")
    for name in ("q_positions", "kv_positions"):
        positions = getattr(dropout, name)
        if positions is not None and (
            positions.dtype != torch.int32 or positions.ndim != 1 or positions.device != device
        ):
            raise ValueError(f"dropout {name} must be a 1-D int32 tensor on {device}")
    dropout_threshold(dropout.p)


def to_dropout_args(
    dropout: DropoutTensors | None, num_heads_q: int, to_tensor, *, for_compile: bool
) -> DropoutArgs | None:
    """Compile args wrap scalars in DSL types; call args pass plain Python scalars."""
    if dropout is None:
        return None
    threshold, rp = dropout_threshold(dropout.p), 1.0 / (1.0 - dropout.p)
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
def dropout_keep(key: Uint32, bh: Int32, q: Int32, k: Int32, threshold: Uint32) -> cutlass.Boolean:
    row = fmix32(Uint32(q) ^ fmix32(Uint32(bh) ^ key))
    return fmix32(row ^ (Uint32(k) * Uint32(_GOLDEN))) >= threshold


@cute.jit
def dropout_bit(bits: cute.Tensor, i: cutlass.Constexpr[int]) -> cutlass.Boolean:
    """Keep bit of fragment element ``i`` in a word-packed keep mask."""
    return ((bits[i // 32] >> Uint32(i % 32)) & Uint32(1)) != Uint32(0)


class DropoutCtx:
    """Per-kernel dropout constants; positions remap local rows / columns."""

    def __init__(self, args: DropoutArgs):
        self.key = dropout_key(args.rng_state)
        self.threshold = args.threshold
        self.rp = args.rp
        self.num_heads_q = args.num_heads_q
        self.q_positions = args.q_positions
        self.kv_positions = args.kv_positions

    @cute.jit
    def keep(self, batch_idx: Int32, head_idx: Int32, q: Int32, k: Int32) -> cutlass.Boolean:
        """``q`` / ``k`` are kernel-local; out-of-range indices read the last position."""
        if cutlass.const_expr(self.q_positions is not None):
            q = self.q_positions[cutlass.min(q, cute.size(self.q_positions) - 1)]
        if cutlass.const_expr(self.kv_positions is not None):
            k = self.kv_positions[cutlass.min(k, cute.size(self.kv_positions) - 1)]
        bh = batch_idx * self.num_heads_q + head_idx
        return dropout_keep(self.key, bh, q, k, self.threshold)


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
    bh = batch_idx * num_heads_q + head_idx
    row = _fmix32_torch(q.to(torch.int64) ^ mix(torch.as_tensor(bh, device=device) ^ key))
    col = (k.to(torch.int64) * _GOLDEN) & _MASK32
    return _fmix32_torch(row[:, None] ^ col[None, :]) >= dropout_threshold(p)
