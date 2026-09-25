"""HSTU function-encoded masks on the FA4 ``mask_mod`` + block-sparsity path.

An HSTU mask stores, for every query row ``q``, an odd number ``n_func`` of
column bounds ``F[:, q]``. Row ``q`` attends key ``k`` iff ``k < F[0, q]`` or
``F[2i + 1, q] <= k < F[2i + 2, q]`` for some ``i``. Unused intervals are
empty (for example ``[0, 0)``). One mask is shared by all batches and heads.

``magi_to_hstu`` converts MagiAttention slices into this encoding. The
attention kernels are unchanged: the mask is evaluated elementwise by a
``mask_mod`` over per-function ``aux_tensors`` and tiles are skipped through
forward (Q-outer) and backward (KV-outer) block-sparse tensors.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch, compute_dq_write_order

# Kernels evaluate mask_mod on whole tiles, so rows past seqlen_q read padding.
HSTU_ROW_PADDING = 256

_MASK_MODS: dict[int, Callable] = {}


def _check_func(func: torch.Tensor, seqlen_q: int) -> torch.Tensor:
    if func.ndim == 4:
        if func.shape[:2] != (1, 1):
            raise ValueError("HSTU func must be shared by batches and heads: [1, 1, n_func, L]")
        func = func[0, 0]
    if func.ndim != 2 or func.dtype != torch.int32:
        raise ValueError("HSTU func must be int32 [n_func, L] or [1, 1, n_func, L]")
    if func.shape[0] % 2 != 1:
        raise ValueError("HSTU func must hold an odd number of functions")
    if func.shape[1] < seqlen_q:
        raise ValueError(f"HSTU func covers {func.shape[1]} rows, expected >= {seqlen_q}")
    return func[:, :seqlen_q]


def magi_to_hstu(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    n_max_func: int = 5,
) -> torch.Tensor:
    """Convert MagiAttention slices to an HSTU func tensor ``[n_func, seqlen_q]``.

    ``mask_types`` uses 0=full, 1=causal, 2=inverse causal, 3=bi-causal. Per
    query row, the slice intervals are sorted and merged (touching intervals
    merge). ``n_func`` is the largest count used by any row, at least 1.
    """
    if q_ranges.shape != k_ranges.shape or q_ranges.ndim != 2 or q_ranges.shape[1] != 2:
        raise ValueError("q_ranges and k_ranges must both be [num_slices, 2]")
    if mask_types.shape != (q_ranges.shape[0],):
        raise ValueError("mask_types must be [num_slices]")
    if seqlen_q <= 0 or seqlen_k <= 0 or n_max_func <= 0:
        raise ValueError("seqlen_q, seqlen_k and n_max_func must be positive")
    device = q_ranges.device
    q = torch.arange(seqlen_q, device=device, dtype=torch.int64)[:, None]
    q_start, q_end = q_ranges[:, 0].long(), q_ranges[:, 1].long()
    k_start, k_end = k_ranges[:, 0].long(), k_ranges[:, 1].long()
    types = mask_types.long()
    # [seqlen_q, num_slices] interval bounds; empty intervals have start >= end.
    start = k_start + torch.where((types & 2) != 0, q - q_start, 0)
    end = k_end - torch.where((types & 1) != 0, q_end - q - 1, 0)
    active = (q >= q_start) & (q < q_end) & (start < end)
    if bool(((start < 0) | (end > seqlen_k))[active].any()):
        raise ValueError("slice intervals must lie within [0, seqlen_k)")
    return intervals_to_hstu(torch.where(active, start, 0), torch.where(active, end, 0), n_max_func)


def intervals_to_hstu(
    starts: torch.Tensor, ends: torch.Tensor, n_max_func: int | None = None
) -> torch.Tensor:
    """Encode per-row KV intervals ``[starts, ends)`` as an HSTU func tensor.

    ``starts``/``ends`` are ``[seqlen_q, m]``; an interval with start >= end is
    empty. Each row's intervals are sorted and merged (touching intervals merge);
    ``n_func`` is the largest count used by any row, at least 1.
    """
    if starts.shape != ends.shape or starts.ndim != 2:
        raise ValueError("starts and ends must both be [seqlen_q, m]")
    seqlen_q, num_slices = starts.shape
    device = starts.device
    starts, ends = starts.long(), ends.long()
    active = starts < ends
    big = torch.iinfo(torch.int64).max
    start = torch.where(active, starts, big)
    order = torch.argsort(start, dim=1, stable=True)
    start = torch.gather(start, 1, order)
    end = torch.gather(torch.where(active, ends, -1), 1, order)
    merged_start = torch.zeros(seqlen_q, num_slices, dtype=torch.int64, device=device)
    merged_end = torch.zeros_like(merged_start)
    count = torch.zeros(seqlen_q, dtype=torch.int64, device=device)
    rows = torch.arange(seqlen_q, device=device)
    for i in range(num_slices):
        s, e = start[:, i], end[:, i]
        valid = s != big
        last = (count - 1).clamp(min=0)
        opens = valid & ((count == 0) | (s > merged_end[rows, last]))
        grows = valid & ~opens
        merged_end[rows[grows], last[grows]] = torch.maximum(
            merged_end[rows[grows], last[grows]], e[grows]
        )
        merged_start[rows[opens], count[opens]] = s[opens]
        merged_end[rows[opens], count[opens]] = e[opens]
        count += opens.long()
    if n_max_func is not None and int(count.max()) > (n_max_func + 1) // 2:
        raise ValueError(f"a query row needs more than n_max_func={n_max_func} functions")
    # Rows whose first interval starts at 0 use [0, F0); others keep [0, 0).
    lead = (count > 0) & (merged_start[:, 0] == 0)
    used = torch.where(count == 0, 0, 2 * count + 1 - 2 * lead.long())
    n_func = max(int(used.max()), 1)
    func = torch.zeros(n_func, seqlen_q, dtype=torch.int32, device=device)
    func[0] = torch.where(lead, merged_end[:, 0], 0).to(torch.int32)
    for j in range(num_slices):
        slot = 2 * j + 1 - 2 * lead.long()
        keep = (j < count) & ~(lead & (j == 0))
        if not bool(keep.any()):
            continue
        func[slot[keep], rows[keep]] = merged_start[keep, j].to(torch.int32)
        func[slot[keep] + 1, rows[keep]] = merged_end[keep, j].to(torch.int32)
    return func


def hstu_dense_mask(func: torch.Tensor, seqlen_q: int, seqlen_k: int) -> torch.Tensor:
    """Boolean ``[seqlen_q, seqlen_k]`` visibility of an HSTU func tensor."""
    func = _check_func(func, seqlen_q).long()
    cols = torch.arange(seqlen_k, device=func.device)[None, :]
    valid = cols < func[0][:, None]
    for i in range(1, func.shape[0], 2):
        valid |= (cols >= func[i][:, None]) & (cols < func[i + 1][:, None])
    return valid


def hstu_aux_tensors(func: torch.Tensor, seqlen_q: int) -> list[torch.Tensor]:
    """One row-interleaved int32 vector: bound ``j`` of row ``q`` at ``q * n_func + j``.

    Interleaving keeps a row's bounds in one cache line; the masks read every
    bound of a row per element in backward.
    """
    func = _check_func(func, seqlen_q)
    padded = torch.zeros(
        seqlen_q + HSTU_ROW_PADDING, func.shape[0], dtype=torch.int32, device=func.device
    )
    padded[:seqlen_q] = func.T
    return [padded.view(-1)]


def hstu_mask_mod(n_func: int, vec_size: int = 1) -> Callable:
    """CuTe ``mask_mod`` for ``hstu_aux_tensors`` of an ``n_func`` func tensor.

    ``vec_size=32`` (SM100/SM110 forward) returns one packed keep mask per 32
    contiguous KV columns: each interval ``[lo, hi)`` is ``below(hi) & ~below(lo)``
    and the OR of all intervals lowers to R2P, like a native interval mask.
    Backward and SM90 always call the scalar form.
    """
    if n_func <= 0 or n_func % 2 != 1:
        raise ValueError("n_func must be a positive odd number")
    if vec_size not in (1, 32):
        raise ValueError("vec_size must be 1 or 32")
    key = (n_func, vec_size)
    if key in _MASK_MODS:
        return _MASK_MODS[key]
    import cutlass
    import cutlass.cute as cute

    from flash_attn.cute import utils
    from flash_attn.cute.mask import r2p_bitmask_below

    num_intervals = (n_func - 1) // 2

    @cute.jit
    def hstu(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        bounds = aux_tensors[0]
        row = m_idx[0] * n_func
        if cutlass.const_expr(cute.size(n_idx.shape) == 1):
            valid = n_idx < utils.scalar_to_ssa(bounds[row], cutlass.Int32)
            for i in cutlass.range_constexpr(num_intervals):
                lo = utils.scalar_to_ssa(bounds[row + 2 * i + 1], cutlass.Int32)
                hi = utils.scalar_to_ssa(bounds[row + 2 * i + 2], cutlass.Int32)
                valid = valid | ((n_idx >= lo) & (n_idx < hi))
            return valid
        else:
            assert cute.size(n_idx.shape) == 32
            # The kernel passes 32 contiguous KV indices starting at n_idx[0].
            base = n_idx[0]
            keep = r2p_bitmask_below(bounds[row] - base, 0)
            for i in cutlass.range_constexpr(num_intervals):
                below_lo = r2p_bitmask_below(bounds[row + 2 * i + 1] - base, 0)
                below_hi = r2p_bitmask_below(bounds[row + 2 * i + 2] - base, 0)
                keep = keep | (below_hi & (below_lo ^ cutlass.Uint32(0xFFFFFFFF)))
            result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
            result[0] = keep
            return result.load()

    hstu.__vec_size__ = vec_size
    _MASK_MODS[key] = hstu
    return hstu


def _block_state(
    valid: torch.Tensor, seqlen_q: int, seqlen_k: int, block_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(partial, full)`` bool ``[num_q_blocks, num_kv_blocks]`` from dense visibility.

    Blocks that cross the Q or KV boundary are never full: the kernel's tile
    masking still has to run for them.
    """
    q_block, kv_block = block_size
    num_q, num_kv = -(-seqlen_q // q_block), -(-seqlen_k // kv_block)
    padded = torch.zeros(num_q * q_block, num_kv * kv_block, dtype=torch.bool, device=valid.device)
    padded[:seqlen_q, :seqlen_k] = valid
    tiles = padded.view(num_q, q_block, num_kv, kv_block)
    any_valid = tiles.any(dim=3).any(dim=1)
    all_valid = tiles.all(dim=3).all(dim=1)
    in_bounds = (torch.arange(1, num_q + 1, device=valid.device) * q_block <= seqlen_q)[:, None] & (
        torch.arange(1, num_kv + 1, device=valid.device) * kv_block <= seqlen_k
    )[None, :]
    full = all_valid & in_bounds
    return any_valid & ~full, full


def _to_sparse(
    partial: torch.Tensor, full: torch.Tensor, block_size: tuple[int, int]
) -> BlockSparseTensorsTorch:
    def pack(active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cols = active.shape[1]
        # Stable sort keeps active indices ascending ahead of inactive ones.
        key = torch.where(active, 0, 1) * cols + torch.arange(cols, device=active.device)
        idx = torch.argsort(key, dim=1).to(torch.int32)
        return active.sum(dim=1).to(torch.int32)[None, None], idx[None, None].contiguous()

    mask_cnt, mask_idx = pack(partial)
    full_cnt, full_idx = pack(full)
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=block_size,
    )


def hstu_block_sparse_tensors(
    func: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    *,
    deterministic: bool = False,
    q_rows_per_chunk: int = 4096,
) -> tuple[BlockSparseTensorsTorch, BlockSparseTensorsTorch]:
    """Forward (Q-outer) and backward (KV-outer) block-sparse tensors.

    Both use ``block_size=(q_block, kv_block)``, head/batch broadcast
    dimensions of 1, and ascending block indices. ``deterministic`` adds the
    ascending dQ write-order tickets deterministic backward requires. Dense
    visibility is built ``q_rows_per_chunk`` rows at a time to bound memory.
    """
    func = _check_func(func, seqlen_q)
    q_block = block_size[0]
    step = max(q_block, q_rows_per_chunk // q_block * q_block)
    partial_rows, full_rows = [], []
    for begin in range(0, seqlen_q, step):
        rows = min(step, seqlen_q - begin)
        valid = hstu_dense_mask(func[:, begin : begin + rows], rows, seqlen_k)
        # Every chunk but the last spans whole Q blocks.
        partial, full = _block_state(valid, rows, seqlen_k, block_size)
        partial_rows.append(partial)
        full_rows.append(full)
    partial = torch.cat(partial_rows)
    full = torch.cat(full_rows)
    fwd = _to_sparse(partial, full, block_size)
    bwd = _to_sparse(partial.T, full.T, block_size)
    if deterministic:
        order, full_order = compute_dq_write_order(
            fwd.mask_block_cnt,
            fwd.mask_block_idx,
            fwd.full_block_cnt,
            fwd.full_block_idx,
            bwd.mask_block_cnt,
            bwd.mask_block_idx,
            bwd.full_block_cnt,
            bwd.full_block_idx,
            spt=False,
        )
        bwd = bwd._replace(dq_write_order=order, dq_write_order_full=full_order, spt=False)
    return fwd, bwd


def hstu_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    func: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    block_size: tuple[int, int] | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """Autograd FA4 attention restricted by an HSTU func mask.

    ``block_size`` defaults to (256, 128) on SM100/SM110 and (128, 128) on
    SM90, matching the sparse tile each backend consumes.
    """
    from flash_attn.cute.interface import flash_attn_func

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    func = _check_func(func, seqlen_q)
    major = torch.cuda.get_device_capability(q.device)[0]
    blackwell = major in (10, 11)
    if block_size is None:
        block_size = (256, 128) if blackwell else (128, 128)
    fwd, bwd = hstu_block_sparse_tensors(
        func, seqlen_q, seqlen_k, block_size, deterministic=deterministic
    )
    return flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        # SM100/SM110 forward masks partial tiles 32 columns at a time with R2P.
        mask_mod=hstu_mask_mod(func.shape[0], vec_size=32 if blackwell else 1),
        aux_tensors=hstu_aux_tensors(func, seqlen_q),
        block_sparse_tensors=fwd,
        block_sparse_tensors_bwd=bwd,
        deterministic=deterministic,
        return_lse=return_lse,
    )
