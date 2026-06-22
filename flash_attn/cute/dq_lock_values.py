# Ported from byted flexible-flash-attention interface.py: deterministic bwd dQ lock-values.
# Provides flash_attn_cute.interface._compute_bwd_dQ_lock_values for magi_attention glue.
import os
import math
from typing import Optional, Tuple
import torch

# Optional Triton import: the deterministic ``_compute_bwd_dQ_lock_values``
# fast path uses two Triton kernels (per-segment in-kernel sort +
# uniqueness-aware histogram, plus a small lock-value kernel). The
# import is guarded so environments without Triton fall back to the
# pure-PyTorch implementation transparently. The Triton path is
# bit-identical to the PyTorch fallback (see ``test_dq_lock_values.py``
# for a cross-check on 700+ shapes).
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton always present in prod
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


# Fixed sentinel for padding slots in the per-segment in-kernel sort.
# Chosen as a constant well above any realistic ``num_m`` (which is at most
# ``ceil(seqlen_q / m_block_size) * B * H``; ``1 << 30`` ~ 1e9 covers all
# production and test inputs by ~6 orders of magnitude). Pinning this to a
# module-level constant lets ``INVALID`` stay a Triton ``constexpr`` (so the
# ``tl.where(..., INVALID)`` / ``tl.full((BLOCK_K,), INVALID, ...)`` stay
# compile-time scalars) without keying the cache on ``num_m`` -- which
# would otherwise re-introduce per-shape recompile churn.
_DQ_LOCK_INVALID_SENTINEL = 1 << 30


# Minimum BLOCK_K bucket. ``tl.sort`` requires the sorted dim to be a
# pow2 >= 2; we floor to 128 so the bucket set across all production+test
# shapes is just ``{128, 256, 512, ...}`` -- a small bounded cache. Counter-
# intuitively this is also the *fastest* setting: BLOCK_K=128 = 4 warps/CTA
# fully saturates the SM warp scheduler, whereas BLOCK_K<=64 leaves it
# under-occupied (<=2 warps). The marginal bitonic-sort stages added by
# padding small ``max_combined`` cases up to 128 are more than recovered by
# the higher warp utilisation. INVALID-padded lanes are masked off on store,
# so correctness is unaffected.
_DQ_LOCK_MIN_BLOCK_K = 128


# Maximum ``num_n`` for the Triton fast path. Above this, the per-segment
# Triton kernel's grid + (num_n, num_m) histogram allocation cost dominates
# and the (already-fast) PyTorch fallback wins. Empirically derived from
# ``exps/microbench/bench_dq_lock_values.py`` (see comment in
# ``_compute_bwd_dQ_lock_values`` for the exact crossover analysis).
# DEVIATION: the threshold is hardware-tuned (measured on Blackwell B200);
# on different GPUs the crossover may shift, but PyTorch baseline is
# launch-bound (~1.3 ms constant) and Triton cost is linear in num_n, so
# the threshold is robust across SM counts.
# Recovery: tweak this constant; it only changes which path is used, never
# the output (the two paths are bit-identical).
_DQ_LOCK_TRITON_NUM_N_THRESHOLD = 4096


# Opt-in flag for the Triton fast path. The Triton path is bit-identical
# to the PyTorch fallback (700+ shape unit tests + fp64 reference,
# see ``tests/test_attn/test_dq_lock_values.py``) and consistently
# 3-6x faster (see ``exps/microbench/bench_dq_lock_values.py``), but we
# keep the production default OFF so that:
#
# 1. Existing users see exactly the pre-optimisation behaviour unless
#    they explicitly opt-in -- no surprise behaviour shifts on upgrade.
# 2. The Triton path's first-call JIT compile (~200-400ms) doesn't get
#    silently triggered in latency-sensitive setup paths.
# 3. Sites with deterministic CI that compares bit-equal output across
#    PyTorch versions get a clean fallback path while validating the
#    Triton kernel on their shape distribution before flipping.
#
# Truthy values: ``1``, ``true``, ``yes``, ``on``, ``y``, ``t`` (case
# insensitive). Anything else (including unset, ``0``, ``false``) =
# PyTorch fallback. Read ONCE at module import to avoid per-call env
# lookups; tests/benchmarks toggle the cached value directly via
# ``flash_attn_cute.interface._DQ_LOCK_USE_TRITON_ENABLED = True``.
_DQ_LOCK_USE_TRITON_ENV = "MAGI_FA4_DQ_LOCK_USE_TRITON"
_DQ_LOCK_USE_TRITON_ENABLED = os.environ.get(
    _DQ_LOCK_USE_TRITON_ENV, ""
).strip().lower() in ("1", "true", "yes", "on", "y", "t")


# Cap on the BLOCK_K bucket reachable via the ``max_combined_hint`` no-sync
# path. Without a cap, a caller passing a loose hint (e.g. ``num_m`` instead
# of the tight ``ceil((wl+wr)/n_block)+2`` for SWA) would push BLOCK_K into
# a brand-new bucket (e.g. 4096, 8192) and recompile the Triton kernels --
# defeating the recompile-avoidance work in ``_DQ_LOCK_MIN_BLOCK_K``.
#
# Behaviour when ``_dq_lock_next_pow2(hint) > _DQ_LOCK_MAX_HINT_BLOCK_K``:
# we transparently fall back to the synced ``max_combined`` derivation
# (one ``.item()``). This preserves the existing 3 active buckets
# ``{128, 256, 512, 1024, 2048}`` (the typical 60B SWA/dense range)
# and any user with a loose hint just gets the old behaviour back,
# never a recompile storm.
#
# Tuning: 2048 supports ``max_combined`` up to 2048 entries per n_block.
# For Magi's largest measured production shapes this is comfortable
# headroom (the densest dense_1024x1024 case has max_combined=1024). If
# in the future a real workload needs > 2048, bump this AND the bucket
# table comment in ``_DQ_LOCK_MIN_BLOCK_K`` together.
_DQ_LOCK_MAX_HINT_BLOCK_K = 2048


def _dq_lock_next_pow2(n: int) -> int:
    """Smallest power-of-two ``>= max(n, _DQ_LOCK_MIN_BLOCK_K)``.

    Floored at ``_DQ_LOCK_MIN_BLOCK_K`` (=128) so the BLOCK_K bucket set
    used as a constexpr cache key is just a small bounded set across all
    production+test shapes; see ``_DQ_LOCK_MIN_BLOCK_K`` for the rationale.
    ``tl.sort`` also requires the sorted dim to be a pow2 >= 2, so any
    floor >= 2 is safe.
    """
    p = 1
    while p < max(n, _DQ_LOCK_MIN_BLOCK_K):
        p <<= 1
    return p



if _HAS_TRITON:

    # ``do_not_specialize`` is critical here: by default Triton specializes
    # scalar int args on (a) divisible-by-16 alignment and (b) equality to 1,
    # which silently extends the cache key beyond the visible ``tl.constexpr``
    # set. ``NUM_N`` / ``NUM_M`` vary across every call (each (B, H, n_block)
    # shape combo); ANY implicit specialisation here would re-introduce the
    # recompile churn this kernel is designed to avoid. We keep ``HAS_FULL``
    # / ``INVALID`` / ``BLOCK_K`` as ``tl.constexpr`` (they intentionally
    # drive the specialisation set: {HAS_FULL_true, HAS_FULL_false} x
    # {BLOCK_K_128, BLOCK_K_256, ...}) -- everything else is runtime.
    @triton.jit(do_not_specialize=["NUM_N", "NUM_M"])
    def _dq_lock_sort_kernel_linear(
        mask_idx_ptr,             # int32 (mask_len,)
        mask_off_ptr,             # int32 (num_n + 1,) prefix-sum
        full_idx_ptr,             # int32 (full_len,)  or unused if HAS_FULL=False
        full_off_ptr,             # int32 (num_n + 1,) or unused if HAS_FULL=False
        combined_off_ptr,         # int32 (num_n + 1,) prefix-sum of (mask_cnt + full_cnt)
        sorted_block_idx_ptr,     # int32 (total,) out (INVALID -> 0)
        sorted_is_full_ptr,       # int32 (total,) out
        m_count_per_seg_ptr,      # int32 (num_n, NUM_M) out (pre-zeroed by host)
        NUM_N,                    # runtime int
        NUM_M,                    # runtime int
        HAS_FULL: tl.constexpr,
        INVALID: tl.constexpr,    # fixed sentinel (see ``_DQ_LOCK_INVALID_SENTINEL``)
        BLOCK_K: tl.constexpr,
    ):
        """Per-segment merge + sort + histogram.

        One program per n_block. Loads the mask + full m_block ids for this
        segment (slicing into the flat ``mask_idx`` / ``full_idx`` arrays via
        the prefix-sum offsets), sorts them with a packed
        ``(value << 1) | is_full`` key so mask sorts before full on the rare
        INVALID tie, and writes both the sorted output and the per-segment
        m_block histogram. The histogram scatter is race-free because, within
        a segment, each m_block id appears at most once (mask and full lists
        are each internally unique and mutually disjoint).
        """
        pid_n = tl.program_id(0)

        mask_start = tl.load(mask_off_ptr + pid_n).to(tl.int32)
        mask_end = tl.load(mask_off_ptr + pid_n + 1).to(tl.int32)
        mask_cnt = mask_end - mask_start

        if HAS_FULL:
            full_start = tl.load(full_off_ptr + pid_n).to(tl.int32)
            full_end = tl.load(full_off_ptr + pid_n + 1).to(tl.int32)
            full_cnt = full_end - full_start
        else:
            full_start = 0
            full_cnt = 0

        combined_start = tl.load(combined_off_ptr + pid_n).to(tl.int32)
        combined_cnt = mask_cnt + full_cnt

        offs_k = tl.arange(0, BLOCK_K)
        in_mask_half = offs_k < mask_cnt
        in_full_half = (offs_k >= mask_cnt) & (offs_k < combined_cnt)
        pos_in_mask = offs_k
        pos_in_full = offs_k - mask_cnt

        mask_vals = tl.load(
            mask_idx_ptr + mask_start + pos_in_mask,
            mask=in_mask_half,
            other=0,
        ).to(tl.int32)
        # Clamp into [0, NUM_M) so out-of-range ids (shouldn't happen on valid
        # input, but we mirror the PyTorch fallback's defensive clamp) don't
        # blow up the histogram scatter address arithmetic.
        mask_safe = tl.maximum(tl.minimum(mask_vals, NUM_M - 1), 0)
        val_mask_half = tl.where(in_mask_half, mask_safe, INVALID)

        if HAS_FULL:
            full_vals = tl.load(
                full_idx_ptr + full_start + pos_in_full,
                mask=in_full_half,
                other=0,
            ).to(tl.int32)
            full_safe = tl.maximum(tl.minimum(full_vals, NUM_M - 1), 0)
            val_full_half = tl.where(in_full_half, full_safe, INVALID)
        else:
            val_full_half = tl.full((BLOCK_K,), INVALID, dtype=tl.int32)

        val = tl.where(in_mask_half, val_mask_half, val_full_half)
        is_full = tl.where(in_mask_half, 0, 1).to(tl.int32)
        valid = in_mask_half | in_full_half

        # Packed sort key: high bits = m_block value, low bit = is_full source.
        # Equivalent to the PyTorch fallback's ``flat * 2 + is_full`` key.
        key = (val.to(tl.int64) << 1) | is_full.to(tl.int64)
        sorted_key = tl.sort(key, dim=0)
        sorted_val = (sorted_key >> 1).to(tl.int32)
        sorted_isf = (sorted_key & 1).to(tl.int32)

        # Mask INVALID -> 0 on the m_block id output (matches the PyTorch
        # fallback's behaviour for the (non-existent in linear layout) tail
        # padding slots; the kernel never reads sorted_block_idx past
        # combined_cnt anyway, so this is purely for tensor hashability /
        # debug consistency).
        invalid_mask = sorted_val == INVALID
        sorted_idx_out = tl.where(invalid_mask, 0, sorted_val)
        mask_store = offs_k < combined_cnt
        tl.store(
            sorted_block_idx_ptr + combined_start + offs_k,
            sorted_idx_out,
            mask=mask_store,
        )
        tl.store(
            sorted_is_full_ptr + combined_start + offs_k,
            sorted_isf,
            mask=mask_store,
        )

        # Histogram scatter: race-free because per-segment uniqueness
        # guarantees no two threads write the same (pid_n, val) cell.
        # ``m_count_per_seg`` is pre-zeroed by the host so we only need to
        # write the 1s.
        val_idx_for_scatter = tl.where(valid, val.to(tl.int64), 0)
        tl.store(
            m_count_per_seg_ptr + pid_n.to(tl.int64) * NUM_M + val_idx_for_scatter,
            tl.full((BLOCK_K,), 1, dtype=tl.int32),
            mask=valid,
        )

    @triton.jit(do_not_specialize=["NUM_N", "NUM_M"])
    def _dq_lock_value_kernel_linear(
        sorted_block_idx_ptr,    # int32 (total,)
        combined_off_ptr,        # int32 (num_n + 1,)
        cum_count_ptr,           # int32 (num_n, NUM_M) inclusive cumsum along n
        m_count_lane_total_ptr,  # int32 (NUM_M,)  == cum_count[-1, :]
        dq_lock_values_ptr,      # int32 (total,) out
        NUM_N,
        NUM_M,
        BLOCK_K: tl.constexpr,
    ):
        """Per-slot lock-value gather using the uniqueness simplification.

        For position p in segment n with m_block ``m``, the per-segment
        uniqueness invariant collapses
            fwd_lock[p]  =  cum_count[n, m] - 1
        and the (always-reverse for Magi LPT scheduling) lock value
            lock[p] = m_count_lane_total[m] - 1 - fwd_lock[p]
                    = m_count_lane_total[m] - cum_count[n, m]
        which is just two int32 gathers and a sub per slot.
        """
        pid_n = tl.program_id(0)
        combined_start = tl.load(combined_off_ptr + pid_n).to(tl.int32)
        combined_end = tl.load(combined_off_ptr + pid_n + 1).to(tl.int32)
        combined_cnt = combined_end - combined_start

        offs_k = tl.arange(0, BLOCK_K)
        active = offs_k < combined_cnt

        sorted_m = tl.load(
            sorted_block_idx_ptr + combined_start + offs_k,
            mask=active,
            other=0,
        ).to(tl.int32)
        safe_m = tl.where(
            active, tl.maximum(tl.minimum(sorted_m, NUM_M - 1), 0), 0,
        ).to(tl.int64)

        cum = tl.load(
            cum_count_ptr + pid_n.to(tl.int64) * NUM_M + safe_m,
            mask=active,
            other=0,
        )
        total = tl.load(
            m_count_lane_total_ptr + safe_m, mask=active, other=0,
        )
        lock_val = total - cum
        lock_val = tl.where(active, lock_val, 0).to(tl.int32)
        tl.store(
            dq_lock_values_ptr + combined_start + offs_k,
            lock_val,
            mask=active,
        )


def _compute_dq_lock_values_triton_linear(
    mask_off: torch.Tensor,
    mask_idx: torch.Tensor,
    full_off: Optional[torch.Tensor],
    full_idx: Optional[torch.Tensor],
    combined_offset: torch.Tensor,
    num_n: int,
    num_m: int,
    total: int,
    max_combined: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-fused implementation of ``_compute_bwd_dQ_lock_values``.

    Replaces the prior ~10 small PyTorch launches
    (arange/searchsorted/scatter/argsort/bincount/cumsum/gather/...) with two
    Triton kernels (sort + histogram, then lock value) plus a single PyTorch
    cumsum along n.

    Bit-identical to the PyTorch fallback on every slot (the linear layout
    has no padding, so every output slot is active).

    Returns ``(sorted_block_idx, sorted_is_full, dq_lock_values)`` as
    ``int32`` tensors of shape ``(total,)``.

    Caller contract:
    * ``total = mask_len + (full_len if has_full else 0)`` -- pre-computed
      by the caller to avoid a host-sync on ``combined_offset[-1]``.
    * ``num_m`` and ``max_combined`` are pre-computed by the caller too;
      they drive the histogram tensor size and the BLOCK_K bucket.
    * ``mask_idx`` / ``full_idx`` must be contiguous int32 on the same CUDA
      device.
    """
    device = mask_idx.device
    has_full = full_idx is not None

    sorted_block_idx = torch.empty(total, dtype=torch.int32, device=device)
    sorted_is_full = torch.empty(total, dtype=torch.int32, device=device)
    # Pre-zero the histogram tensor: the kernel only writes 1s at valid
    # (segment, m) cells via a race-free scatter (per-segment uniqueness).
    m_count_per_seg = torch.zeros(
        (num_n, num_m), dtype=torch.int32, device=device,
    )

    BLOCK_K = _dq_lock_next_pow2(max_combined)
    grid = (num_n,)

    if has_full:
        full_idx_arg = full_idx
        full_off_arg = full_off
    else:
        # Dummy 1-element tensors satisfy the kernel's ptr args; the kernel
        # never reads them because HAS_FULL=False short-circuits all loads.
        full_idx_arg = mask_idx
        full_off_arg = mask_off

    # NUM_N / NUM_M travel as runtime ints so the cache key is just
    # (HAS_FULL, BLOCK_K). INVALID is fixed by ``_DQ_LOCK_INVALID_SENTINEL``
    # (still constexpr, but only one value across all callers).
    _dq_lock_sort_kernel_linear[grid](
        mask_idx, mask_off, full_idx_arg, full_off_arg, combined_offset,
        sorted_block_idx, sorted_is_full, m_count_per_seg,
        num_n,
        num_m,
        HAS_FULL=has_full,
        INVALID=_DQ_LOCK_INVALID_SENTINEL,
        BLOCK_K=BLOCK_K,
    )

    # Single PyTorch cumsum across n -- the only cross-segment dependency
    # in the lock-value derivation. Kept in PyTorch where it's already
    # fast and benefits from cuDNN-tuned reduction kernels. Output is
    # naturally contiguous (cumsum allocates a fresh output).
    cum_count = m_count_per_seg.cumsum(dim=0, dtype=torch.int32)
    # ``cum_count[-1, :]`` is the last row of a C-contiguous 2-D tensor,
    # which is contiguous-by-construction (no copy needed). Likewise
    # ``cum_count`` itself doesn't need ``.contiguous()`` because it came
    # from a fresh ``cumsum``.
    m_count_lane_total = cum_count[-1, :]

    dq_lock_values = torch.empty(total, dtype=torch.int32, device=device)
    _dq_lock_value_kernel_linear[grid](
        sorted_block_idx, combined_offset,
        cum_count, m_count_lane_total,
        dq_lock_values,
        num_n,
        num_m,
        BLOCK_K=BLOCK_K,
    )

    return sorted_block_idx, sorted_is_full, dq_lock_values


def _compute_bwd_dQ_lock_values(
    block_sparse_tensors,
    *,
    num_m_blocks_actual: Optional[int] = None,
    max_combined_hint: Optional[int] = None,
):
    """Precompute sorted iteration metadata for deterministic backward.

    Shape symbols used below (all 1-D unless noted):
      num_n    – number of n_blocks (kv tiles)
      mask_len – total mask entries across all n_blocks
                 (== ``block_sparse_tensors.mask_block_idx.shape[0]``)
      full_len – total full entries across all n_blocks
                 (== ``block_sparse_tensors.full_block_idx.shape[0]``, 0 if no full)
      total    – mask_len + full_len
      num_m    – number of distinct m_blocks, i.e. ``max(num_n, sorted_flat.max()+1)``

    Returns a dict with:

    * ``dQ_lock_values``  — int32 (total,): semaphore lock value per entry
    * ``dQ_lock_combined_offset`` — int32 (num_n + 1,): prefix-sum offset
    * ``sorted_block_idx``  — int32 (total,): m_block indices sorted by
      m_block within each n_block (merge of mask + full)
    * ``sorted_block_is_full`` — int32 (total,): 0 = mask, 1 = full

    The sorted order ensures CTAs encounter the same m_block at similar
    iteration indices, minimising semaphore wait bubbles.

    Prefer calling this once at ``FA4AttnArg`` construction time.

    Parameters
    ----------
    block_sparse_tensors : LinearBlockSparseTensorsTorch
        The block-sparse CSR-like inputs.
    num_m_blocks_actual : int, optional (keyword-only)
        Static upper bound on ``num_m`` (= number of distinct query
        tiles). When provided, skips the ``.item()`` sync that would
        otherwise derive ``num_m`` from
        ``max(mask_idx.max(), full_idx.max()) + 1``. Callers that know
        ``num_m = ceil(seqlen_q / m_block_size)`` (or the equivalent
        flattened ``(B, H, num_m_per_bh)`` for
        ``bhqk_to_linear_sparse_tensors`` consumers) should pass this
        for fully async execution. When ``None`` (default), one
        host-sync is performed -- bit-identical to the original
        implementation's behaviour.

        Also lets the PyTorch fallback path skip its ``seg_multiplier``
        sync (the multiplier becomes ``2 * num_m_blocks_actual``, a
        safe upper bound since ``flat.max() < num_m_blocks_actual``).
    max_combined_hint : int, optional (keyword-only)
        Tight upper bound on ``max(mask_cnt + full_cnt)`` -- i.e. the
        maximum number of entries any single n_block contributes. When
        provided AND the rounded ``BLOCK_K`` falls within
        ``_DQ_LOCK_MAX_HINT_BLOCK_K`` (=2048), the Triton path skips
        the LAST remaining host-sync (the ``max_combined`` derivation),
        making the function fully asynchronous w.r.t. the GPU stream.

        Recommended tight bounds (computed entirely from static config):
          * causal:        ``num_n`` (or smaller if seqlen_q < seqlen_k)
          * SWA symmetric: ``ceil((wl + wr) / n_block_size) + 2``
          * SWA causal:    ``ceil(wl / n_block_size) + 2``
          * block_diag:    ``ceil(doc_size / max(m_block, n_block)) + 1``
          * dense:         ``num_m``

        Correctness contract: ``max_combined_hint >= max(mask_cnt[i] +
        full_cnt[i])`` for all i. Violating this corrupts the in-kernel
        sort (silent buffer truncation). The kernel does NOT validate
        the hint at runtime -- this is intentional, since validation
        would require the very sync we are trying to avoid.

        Recompile safety: the hint is rounded to a pow2 BLOCK_K bucket
        (``_dq_lock_next_pow2``, floored at 128). If the rounded bucket
        exceeds ``_DQ_LOCK_MAX_HINT_BLOCK_K`` (=2048), the hint is
        silently ignored and we fall back to the synced derivation --
        this caps the Triton kernel cache key space at the active
        bucket set ``{128, 256, 512, 1024, 2048}`` regardless of how
        loose the caller's hint is. A loose hint is therefore safe in
        the worst case (reverts to default behaviour) but a tight hint
        is strictly better (avoids the sync AND keeps the hot
        ``BLOCK_K=128/256`` cache entries).
    """
    device = block_sparse_tensors.mask_block_idx.device

    # Inputs from LinearBlockSparseTensorsTorch. `*_block_offset` is a prefix-sum
    # so it has one extra entry (leading 0, trailing total count).
    mask_cnt = block_sparse_tensors.mask_block_cnt        # int32 (num_n,)
    mask_off = block_sparse_tensors.mask_block_offset     # int32 (num_n + 1,)
    mask_idx = block_sparse_tensors.mask_block_idx        # int32 (mask_len,)
    has_full = block_sparse_tensors.full_block_cnt is not None

    mask_len = mask_idx.shape[0]
    num_n = mask_cnt.shape[0]

    if has_full:
        full_cnt = block_sparse_tensors.full_block_cnt        # int32 (num_n,)
        full_off = block_sparse_tensors.full_block_offset     # int32 (num_n + 1,)
        full_idx = block_sparse_tensors.full_block_idx        # int32 (full_len,)
        full_len = full_idx.shape[0]
    else:
        full_cnt = None
        full_off = None
        full_idx = None
        full_len = 0

    total = mask_len + full_len

    # -- combined offset (prefix-sum of per-n_block total counts) ---------
    # int32 directly: per-segment counts are tiny (typical <= 256), and the
    # cumulative total is bounded by ``total`` which fits in int32 for any
    # realistic input (production largest ~5M entries). Keeping int32 saves
    # 2-3 PyTorch op launches over the int64-then-cast form (each save is
    # ~30us on the host-overhead-bound path -- worth ~10% of the whole
    # Triton path latency).
    total_per_n = mask_cnt  # int32 (num_n,)
    if has_full:
        total_per_n = total_per_n + full_cnt  # int32 (num_n,)
    # ``F.pad`` is one op (vs zeros + slice-assign which is 2). Output is
    # bit-identical: leading 0 followed by inclusive cumsum.
    combined_offset = torch.nn.functional.pad(
        torch.cumsum(total_per_n, dim=0, dtype=torch.int32),
        (1, 0),
    )

    empty = dict(
        dQ_lock_values=torch.zeros(0, dtype=torch.int32, device=device),
        dQ_lock_combined_offset=combined_offset,
        sorted_block_idx=torch.zeros(0, dtype=torch.int32, device=device),
        sorted_block_is_full=torch.zeros(0, dtype=torch.int32, device=device),
    )
    if total == 0:
        return empty

    # ------------------------------------------------------------------
    # FAST PATH (Triton, CUDA only): fuse the per-segment merge + sort +
    # histogram + lock-value computation into two Triton kernels (plus a
    # single PyTorch cumsum along n). See ``_dq_lock_sort_kernel_linear``
    # and ``_dq_lock_value_kernel_linear`` for the algorithm; we exploit
    # the per-segment uniqueness invariant (mask/full m_block ids within
    # a single n_block are pairwise distinct -- ``torch.nonzero`` returns
    # sorted-unique positions and mask/full classifications are mutually
    # exclusive) to collapse the lock-value formula into a single
    # ``cum_count`` gather, dodging the per-segment rank computation
    # entirely.
    #
    # Bit-identical to the PyTorch fallback on every slot (the linear
    # layout has no padding so every output slot is active and gets the
    # same value either way).
    #
    # Heuristic ``num_n <= _DQ_LOCK_TRITON_NUM_N_THRESHOLD``: the Triton
    # path's grid is ``(num_n,)`` (one program per segment) and its
    # histogram tensor is ``(num_n, num_m)``. When ``num_n`` blows up
    # past ~5k (the BH-flattened ``bhqk_to_linear_sparse_tensors`` path
    # with H=80) the per-program work becomes too small to amortise the
    # kernel launch overhead AND the histogram alloc/zero/cumsum starts
    # dominating. In that regime the PyTorch path (which uses one big
    # global argsort + bincount + cumsum on 1-D tensors of size
    # ``total``) wins because its launch count is small and constant.
    # See ``exps/microbench/bench_dq_lock_values.py`` for the data; the
    # threshold of 4096 falls in the steady "Triton wins ≥2x" zone for
    # the cases we measured and conservatively dodges the
    # ``bh=80`` regression (Triton was 5x slower at ``num_n=20480``).
    # ------------------------------------------------------------------
    # Triton path is opt-in. Production default is OFF so the function's
    # behaviour and JIT-compile latency remain exactly as before the
    # optimisation; set ``MAGI_FA4_DQ_LOCK_USE_TRITON=1`` to enable.
    # All four guards must hold for Triton: feature flag on, kernels
    # available, inputs on CUDA, and ``num_n`` within the threshold
    # where Triton beats the PyTorch fallback (see comment on
    # ``_DQ_LOCK_TRITON_NUM_N_THRESHOLD`` for the crossover analysis).
    use_triton = (
        _DQ_LOCK_USE_TRITON_ENABLED
        and _HAS_TRITON
        and mask_idx.is_cuda
        and (not has_full or full_idx.is_cuda)
        and num_n <= _DQ_LOCK_TRITON_NUM_N_THRESHOLD
    )
    if use_triton:
        # ``num_m`` and ``max_combined`` are the only host-visible scalars
        # the Triton kernels need: the former sizes the histogram tensor,
        # the latter chooses the ``BLOCK_K`` constexpr bucket. We have a
        # 3-way dispatch on caller-provided hints:
        #
        # 1. Both hints supplied + ``max_combined_hint`` fits the cache:
        #    fully async, ZERO host-syncs. This matches blackstone's
        #    no-sync property -- the gap was purely caller-side info,
        #    not algorithmic. The hint cap (``_DQ_LOCK_MAX_HINT_BLOCK_K``)
        #    guards against recompile churn from loose hints.
        # 2. ``num_m_blocks_actual`` only (or hint too loose): one sync
        #    on ``max_combined`` (a single int).
        # 3. Neither: one coalesced sync via stack+tolist() pulling both
        #    scalars in one cudaStreamSync (bit-identical to the prior
        #    two-sync form but half the host stalls).
        #
        # Loose-hint detection MUST happen before any GPU work so that
        # the fallback's sync stays a single sync, not two (one for
        # validation + one for derivation).
        hint_usable = (
            max_combined_hint is not None
            and _dq_lock_next_pow2(max_combined_hint) <= _DQ_LOCK_MAX_HINT_BLOCK_K
        )

        if hint_usable and num_m_blocks_actual is not None:
            # FULLY ASYNC: no host transfers in this branch. The kernel
            # cache key is (HAS_FULL, BLOCK_K) where BLOCK_K is derived
            # purely from the caller-provided hint -- so two callers
            # with the same static config hit the same compiled kernel
            # without any GPU-side measurement.
            max_combined = int(max_combined_hint)
            num_m = max(num_n, num_m_blocks_actual)
        elif num_m_blocks_actual is not None:
            # Partial: caller knows num_m statically, but not (or only
            # loosely) max_combined. One small sync.
            max_combined = int(total_per_n.max().item())
            num_m = max(num_n, num_m_blocks_actual)
        else:
            # Coalesced sync: stack max_combined and max_m into a 2-elem
            # tensor and do ONE host transfer via .tolist(). This is
            # bit-identical to the prior two-sync form but cuts the
            # cudaStreamSync count in half (each sync was ~30us on the
            # observed Triton path; eliminating one is ~10% speedup).
            if mask_len > 0 and has_full and full_len > 0:
                max_m_t = torch.maximum(mask_idx.max(), full_idx.max())
            elif mask_len > 0:
                max_m_t = mask_idx.max()
            else:
                max_m_t = full_idx.max()
            stacked = torch.stack([total_per_n.max(), max_m_t.to(total_per_n.dtype)])
            mc, mm = stacked.tolist()
            # If the caller passed a hint that was just *too loose* (got
            # capped above), we can still honour ``num_m`` from the
            # synced ``mm`` -- but ``mc`` is the actual measured value,
            # which guarantees a tight ``BLOCK_K`` for this call. We do
            # NOT trust the (capped) hint for ``max_combined`` here
            # because using the larger hint would push us into a bigger
            # bucket than necessary, undoing the cap's protection.
            max_combined = int(mc)
            num_m = max(num_n, int(mm) + 1)

        # Ensure mask_idx / full_idx are int32 (the kernel's ptr arithmetic
        # assumes int32 layout). Inputs from LinearBlockSparseTensorsTorch
        # are already int32 in all callers we know of, so these are no-ops
        # (we skip the redundant ``.contiguous()`` calls too -- the
        # source tensors come from ``torch.cat`` / ``torch.cumsum`` /
        # ``torch.zeros`` which all return contiguous).
        if mask_idx.dtype != torch.int32:
            mask_idx = mask_idx.to(torch.int32)
        if mask_off.dtype != torch.int32:
            mask_off = mask_off.to(torch.int32)
        if has_full:
            if full_idx.dtype != torch.int32:
                full_idx = full_idx.to(torch.int32)
            if full_off.dtype != torch.int32:
                full_off = full_off.to(torch.int32)

        sorted_block_idx, sorted_is_full, lock_values = _compute_dq_lock_values_triton_linear(
            mask_off=mask_off,
            mask_idx=mask_idx,
            full_off=full_off,
            full_idx=full_idx,
            combined_offset=combined_offset,
            num_n=num_n,
            num_m=num_m,
            total=total,
            max_combined=max_combined,
        )
        return dict(
            dQ_lock_values=lock_values,
            dQ_lock_combined_offset=combined_offset,
            sorted_block_idx=sorted_block_idx,
            sorted_block_is_full=sorted_is_full,
        )

    # ------------------------------------------------------------------
    # FALLBACK PATH (pure PyTorch): kept for CPU tensors and environments
    # without Triton. Bit-identical to the original pre-optimisation
    # implementation -- this is the reference behaviour the Triton path
    # is validated against.
    # ------------------------------------------------------------------
    # -- flat m_block & is_full arrays (mask-first order) -----------------
    cum64 = combined_offset.to(torch.int64)               # int64 (num_n + 1,)

    mp = torch.arange(mask_len, dtype=torch.int64, device=device)   # int64 (mask_len,)
    mo = mask_off.to(torch.int64)                                   # int64 (num_n + 1,)
    # mn[i] = which n_block the i-th mask entry belongs to, in [0, num_n).
    mn = torch.searchsorted(mo, mp, right=True) - 1                 # int64 (mask_len,)
    # Destination of the i-th mask entry in the combined (mask-first) layout.
    mask_dst = cum64[mn] + (mp - mo[mn])                            # int64 (mask_len,)

    # ``flat`` is the merged m_block-id array that unifies mask_idx and full_idx
    # into a single per-n_block-segmented buffer, laid out as:
    #   [ n0.mask | n0.full | n1.mask | n1.full | ... | n_{num_n-1}.mask | n_{num_n-1}.full ]
    # Segment boundaries follow ``cum64`` (combined prefix-sum), and inside each
    # n_block segment mask entries precede full entries. This unified layout is
    # what the downstream per-segment sort (see below) operates on.
    #
    # Concrete example (num_n = 3):
    #   mask_cnt = [2, 3, 1],  full_cnt = [1, 0, 2]   => total = 9
    #   mask_off = [0, 2, 5, 6]       (prefix sum of mask_cnt)
    #   cum64    = [0, 3, 6, 9]       (prefix sum of mask_cnt + full_cnt)
    #   mask_idx = [a0, a1, b0, b1, b2, c0]                  (flat, by n_block)
    #   full_idx = [A0,         C0, C1]                      (flat, by n_block)
    #
    # After the two scatters below fill both mask and full slots, ``flat`` becomes:
    #   index:     0   1   2   3   4   5   6   7   8
    #   flat:    [ a0  a1  A0  b0  b1  b2  c0  C0  C1 ]
    #   is_full: [  0   0   1   0   0   0   0   1   1 ]
    #            └─ n_block0 ─┘└ n_block1 ┘└ n_block2 ┘
    # (lowercase = mask entries, uppercase = full entries; each cell holds the
    # m_block id that the corresponding mask/full entry points at.)
    flat = torch.zeros(total, dtype=torch.int64, device=device)     # int64 (total,) — m_block ids
    # Scatter mask entries into their mask-first slots computed above; full
    # slots remain 0 for now and are filled in the ``has_full`` branch below.
    # In the example above this writes [a0, a1, 0, b0, b1, b2, c0, 0, 0].
    flat.scatter_(0, mask_dst, mask_idx.to(torch.int64))

    is_full = torch.zeros(total, dtype=torch.int32, device=device)  # int32 (total,) — 0 mask / 1 full

    if has_full and full_len > 0:
        fp = torch.arange(full_len, dtype=torch.int64, device=device)   # int64 (full_len,)
        fo = full_off.to(torch.int64)                                   # int64 (num_n + 1,)
        fn = torch.searchsorted(fo, fp, right=True) - 1                 # int64 (full_len,) in [0, num_n)
        # full entries come right after this n_block's mask entries in the combined layout.
        full_dst = cum64[fn] + mask_cnt[fn].to(torch.int64) + (fp - fo[fn])  # int64 (full_len,)
        flat.scatter_(0, full_dst, full_idx.to(torch.int64))
        is_full.scatter_(0, full_dst, torch.ones(full_len, dtype=torch.int32, device=device))

    # -- per-n_block sort by m_block (stable, mask entries before full) ---
    # Sort key = m_block * 2 + is_full so ties break mask-before-full.
    sort_key = flat * 2 + is_full.to(torch.int64)         # int64 (total,)
    # Segment-sort: offset each n_block's keys to keep segments separate.
    # The multiplier MUST strictly exceed the largest intra-segment sort key,
    # otherwise argsort interleaves entries across n_blocks and the resulting
    # ``sorted_block_idx`` / ``dQ_lock_values`` become incoherent with
    # ``dQ_lock_combined_offset`` — which deadlocks the deterministic reduce
    # warpgroup on wait_eq. max(sort_key_in_segment) = 2 * max(flat) + 1, so
    # a multiplier of 2 * (max(flat) + 1) is sufficient.
    #
    # Sync-avoidance: when ``num_m_blocks_actual`` is supplied, we use
    # it as the static upper bound for ``max(flat) + 1`` (since
    # ``flat`` only contains m_block ids in ``[0, num_m_blocks_actual)``).
    # This is slightly looser than ``max(flat) + 1`` but still correct
    # by construction -- it just yields a larger ``seg_multiplier``,
    # which has no semantic effect (only inflates the int64 keys, all
    # comparisons remain consistent). Cost: zero. Benefit: removes the
    # last sync from the PyTorch fallback path.
    if num_m_blocks_actual is not None:
        max_flat_plus_one = max(num_m_blocks_actual, 1)
    else:
        max_flat_plus_one = int(flat.max().item()) + 1 if total > 0 else 1
    seg_multiplier = max(num_n * 4, 2 * max_flat_plus_one)
    seg_bias = torch.repeat_interleave(
        torch.arange(num_n, dtype=torch.int64, device=device) * seg_multiplier,
        total_per_n,
    )                                                      # int64 (total,)
    sort_key = sort_key + seg_bias                         # int64 (total,)

    perm = torch.argsort(sort_key, stable=True)            # int64 (total,)
    sorted_flat = flat[perm]                               # int64 (total,) — sorted m_block ids
    sorted_is_full = is_full[perm]                         # int32 (total,)

    # --------------------------------------------------------------------
    # Lock values for deterministic dQ accumulation
    # --------------------------------------------------------------------
    # Background: multiple CTAs produce partial dQ contributions that all land
    # on the same ``m_block`` (query tile). To make the reduction bitwise
    # deterministic we force a fixed accumulation order via per-m_block
    # semaphores: each entry gets a ``lock_value`` and the kernel executes
    # ``wait_eq(dQ_sem[m], lock_value)`` before adding its contribution, then
    # ``atomic_add(dQ_sem[m], 1)`` to hand off to the next one.
    #
    # We assign lock values in REVERSE order of the sorted sequence, so that
    # the entry scheduled to run FIRST on the GPU carries ``lock=0`` (it
    # never waits) and subsequent entries carry 1, 2, ... Combined with the
    # LPT scheduler's block reversal (``spt=True``), CTA 0 maps to
    # ``n_block = num_n - 1``, which lines up with the lock=0 slots so the
    # chain never stalls on a CTA that hasn't launched yet.
    #
    # Running example (num_n = 3, total = 6):
    #   sorted_flat = [ 0   5   2   5   0   5 ]     m_block ids
    #   position        0   1   2   3   4   5
    #                   └─ n0 ┘└─ n1 ┘└─ n2 ─┘
    # m_block occurrences: m=0 twice, m=2 once, m=5 three times.

    positions = torch.arange(total, dtype=torch.int64, device=device)  # int64 (total,)
    # ``num_m`` must cover every m_block id that appears, and also be at least
    # ``num_n`` since the kernel indexes the semaphore array by m_block.
    # Honour the optional ``num_m_blocks_actual`` parameter to skip the
    # ``.item()`` sync when the caller already knows the static bound.
    if num_m_blocks_actual is not None:
        num_m = max(num_n, num_m_blocks_actual)
    else:
        num_m = max(num_n, int(sorted_flat.max().item()) + 1) if total > 0 else num_n

    # ---- Step 1: forward arrival index per m_block -------------------------
    # ``fwd_lock[p]`` = how many earlier entries (p' < p) in ``sorted_flat``
    # already target the same m_block as entry p. I.e. "I am the k-th one to
    # hit this m_block, counting from 0".
    #
    # Memory-efficient sort-based computation (O(total) memory vs the
    # O(num_m * total) one-hot+cumsum approach, which blows up when ``total``
    # and ``num_m`` both scale with num_n):
    #   1. Stable argsort ``sorted_flat`` by m_block value. Entries sharing
    #      the same m_block become consecutive in this order, and stability
    #      preserves their original p-order within each group.
    #   2. For the i-th entry in the sorted order, its rank within its group
    #      equals ``i - group_start[m]``, where ``group_start`` is the
    #      prefix-sum of per-m_block occurrence counts (i.e. ``bincount``).
    #   3. Scatter the per-sorted-position ranks back to the original
    #      positions via the argsort permutation.
    #
    # For the example (sorted_flat = [0, 5, 2, 5, 0, 5], num_m = 6):
    #   order         = [0, 4, 2, 1, 3, 5]    (stable argsort)
    #   sorted_m      = [0, 0, 2, 5, 5, 5]
    #   m_block_count = [2, 0, 1, 0, 0, 3]
    #   group_starts  = [0, 2, 2, 3, 3, 3]    (exclusive prefix-sum)
    #   ranks_in_ord  = [0, 1, 0, 0, 1, 2]    (i - group_starts[sorted_m[i]])
    #   fwd_lock[order[i]] = ranks_in_ord[i]
    #                 = [0, 0, 0, 1, 1, 2]
    order = torch.argsort(sorted_flat, stable=True)                        # int64 (total,)
    sorted_m = sorted_flat[order]                                          # int64 (total,)
    m_block_count = torch.bincount(sorted_flat, minlength=num_m).to(torch.int32)  # int32 (num_m,)
    group_starts = torch.zeros(num_m, dtype=torch.int64, device=device)    # int64 (num_m,)
    group_starts[1:] = m_block_count[:-1].to(torch.int64).cumsum(0)
    ranks_in_order = positions - group_starts[sorted_m]                    # int64 (total,)
    fwd_lock = torch.empty(total, dtype=torch.int32, device=device)        # int32 (total,)
    fwd_lock[order] = ranks_in_order.to(torch.int32)

    # ---- Step 2: reverse into lock values ----------------------------------
    # ``m_block_count[m]`` = total number of entries that target m_block m
    # (computed above via bincount, reused here).
    # Reversed lock: the LAST arrival (fwd_lock = count-1) gets lock 0, the
    # second-to-last gets lock 1, etc., so the reduction chain completes in
    # the reverse of the sorted sequence.
    #
    #   lock_values[p] = m_block_count[sorted_flat[p]] - 1 - fwd_lock[p]
    #
    # For the example:
    #   m_block_count = [ 2, _, 1, _, _, 3 ]   (m=1,3,4 unused → 0)
    #   lock_values   = [ 2-1-0, 3-1-0, 1-1-0, 3-1-1, 2-1-1, 3-1-2 ]
    #                 = [   1,     2,     0,     1,     0,     0   ]
    #
    # Kernel-side usage of these lock values (per CTA / per n_block):
    #   for i in range(combined_offset[n], combined_offset[n+1]):
    #       m    = sorted_block_idx[i]
    #       full = sorted_block_is_full[i]
    #       # ... compute partial dQ for (n, m) ...
    #       wait_eq(dQ_sem[m], dQ_lock_values[i])   # fixed predecessor chain
    #       dQ[m] += partial_dQ
    #       atomic_add(dQ_sem[m], 1)                # hand off to successor
    lock_values = (m_block_count[sorted_flat] - 1 - fwd_lock)              # int32 (total,)

    # --------------------------------------------------------------------
    # Returned tensors (the "deterministic schedule table" consumed by the
    # backward kernel). All four are aligned entry-by-entry along ``total``
    # except for ``dQ_lock_combined_offset`` which partitions ``total`` by
    # n_block:
    #
    #   dQ_lock_values         — int32 (total,)
    #       Per-entry semaphore value the CTA must ``wait_eq`` on before
    #       accumulating its partial dQ into ``dQ[m_block]``. Ordering is
    #       REVERSED so the earliest-scheduled CTA (mapped to the last
    #       n_block by LPT reversal) carries lock=0 and never blocks.
    #
    #   dQ_lock_combined_offset — int32 (num_n + 1,)
    #       Prefix-sum boundaries. CTA handling n_block ``n`` reads the
    #       slice ``[combined_offset[n], combined_offset[n+1])`` of the
    #       three per-entry arrays below.
    #
    #   sorted_block_idx       — int32 (total,)
    #       The sorted sequence of m_block ids each (n_block, entry) pair
    #       targets. Within an n_block segment: sorted by m_block ascending,
    #       mask entries before full entries on ties.
    #
    #   sorted_block_is_full   — int32 (total,)
    #       0 = mask block (apply sparsity/causal mask in the kernel),
    #       1 = full block (dense fast-path). Index-aligned with
    #       ``sorted_block_idx``.
    # --------------------------------------------------------------------
    return dict(
        dQ_lock_values=lock_values,
        dQ_lock_combined_offset=combined_offset,
        sorted_block_idx=sorted_flat.to(torch.int32),
        sorted_block_is_full=sorted_is_full,
    )


