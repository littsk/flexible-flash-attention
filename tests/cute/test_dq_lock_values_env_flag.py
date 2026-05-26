"""Submodule-local test: ``MAGI_FA4_DQ_LOCK_USE_TRITON`` equivalence guard.

Scope (deliberately narrow — the full correctness sweep lives upstream in
``Magi_Attention/tests/test_attn/test_dq_lock_values.py`` with 700+ shape
combinations):

This file is the submodule's *self-contained* contract test. It verifies
the SINGLE invariant the env-var introduces, in isolation from any
``magi_attention`` package layer:

    ``_compute_bwd_dQ_lock_values(...)`` produces bit-identical output
    regardless of whether ``_DQ_LOCK_USE_TRITON_ENABLED`` is True or False,
    for every supported caller-hint combination
    (no hints / num_m_only / max_combined_only / both / loose-hint-over-cap).

Why a separate, slim test here:

1. The submodule may be vendored / consumed standalone by other repos
   (the canonical entry point ``flash_attn_cute.interface`` is in this
   submodule, not in the parent project). Pinning the env-var contract
   inside the submodule lets downstream consumers run *just*
   ``pytest tests/cute/test_dq_lock_values_env_flag.py`` to validate
   the flip without setting up the parent project's deeper testing
   infra (``magi_attention.testing.parameterize`` etc.).
2. CI in the submodule can guard against accidental regressions to the
   env-var default or the dispatch wiring. The upstream test does the
   same check but the submodule shouldn't rely on the parent's CI to
   catch its own contract violations.
3. Keeps the blast radius small — if ``LinearBlockSparseTensorsTorch``
   ever moves modules inside the submodule, this test breaks first and
   loudly (since it imports the actual symbol), serving as a tripwire.

If you need broader correctness coverage (mask patterns, irregular seqlen,
hint cap edges, dtype variants, semantic invariants, etc.), add tests to
the upstream ``test_dq_lock_values.py`` instead — duplicating the full
sweep here would be unmaintainable.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pytest
import torch

try:
    from flash_attn_cute.interface import _compute_bwd_dQ_lock_values
    from flash_attn_cute.block_sparsity import LinearBlockSparseTensorsTorch
    import flash_attn_cute.interface as _iface

    _CAN_IMPORT = True
except Exception:  # pragma: no cover - import errors should be impossible in-tree
    _CAN_IMPORT = False


# ---------------------------------------------------------------------------
# Sparse-tensor factories — one per mask pattern. Each builds a
# ``LinearBlockSparseTensorsTorch`` for a specific attention pattern so
# the env-flag equivalence test can exercise every code path the Triton
# kernel handles (mask-only / full-only / mixed; short / long segments;
# dense / sparse) without depending on the parent project's test helpers.
# ---------------------------------------------------------------------------


def _build_sparse(
    mask_lists: List[List[int]],
    full_lists: Optional[List[List[int]]],
    *,
    device: torch.device,
) -> LinearBlockSparseTensorsTorch:
    """Pack per-n_block (mask, full) ID lists into the LinearBlockSparse
    CSR-like layout the kernel consumes.

    Per-segment uniqueness invariant: ``mask_lists[n]`` and ``full_lists[n]``
    must each be sorted-unique and their intersection must be empty. The
    factories below enforce this by construction. ``full_lists=None``
    produces a mask-only CSR (``has_full=False``) -- this exercises the
    full-list-absent branch in the Triton kernel's conditional load.
    """
    num_n = len(mask_lists)
    mask_cnt = torch.tensor(
        [len(l) for l in mask_lists], dtype=torch.int32, device=device,
    )
    mask_off = torch.zeros(num_n + 1, dtype=torch.int32, device=device)
    mask_off[1:] = torch.cumsum(mask_cnt, dim=0)
    mask_idx = torch.tensor(
        [m for l in mask_lists for m in l] or [0],
        dtype=torch.int32, device=device,
    )[: mask_cnt.sum().item()]  # trim the dummy [0] when total == 0

    if full_lists is None:
        return LinearBlockSparseTensorsTorch(
            mask_block_cnt=mask_cnt,
            mask_block_offset=mask_off,
            mask_block_idx=mask_idx,
            full_block_cnt=None,
            full_block_offset=None,
            full_block_idx=None,
        )

    full_cnt = torch.tensor(
        [len(l) for l in full_lists], dtype=torch.int32, device=device,
    )
    full_off = torch.zeros(num_n + 1, dtype=torch.int32, device=device)
    full_off[1:] = torch.cumsum(full_cnt, dim=0)
    full_idx = torch.tensor(
        [m for l in full_lists for m in l] or [0],
        dtype=torch.int32, device=device,
    )[: full_cnt.sum().item()]

    return LinearBlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_offset=mask_off,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_offset=full_off,
        full_block_idx=full_idx,
    )


def _causal_lists(
    seqlen: int, m_block: int, n_block: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Causal mask: each n_block N=[kv_lo, kv_hi) sees all m_blocks M
    where the M-N rectangle overlaps the lower triangle. Splits into
    ``full`` (entirely below the diagonal) and ``mask`` (straddles it).

    Coverage profile: max_combined grows linearly with n_block index
    (last n_block sees all m_blocks). This is the dominant production
    pattern; exercises both mask+full present and the typical
    BLOCK_K=128 path (since max_combined ≈ num_m for max-n_block).
    """
    num_m = seqlen // m_block
    num_n = seqlen // n_block
    mask_lists: list[list[int]] = []
    full_lists: list[list[int]] = []
    for n in range(num_n):
        kv_lo = n * n_block
        kv_hi = kv_lo + n_block
        partial, full = [], []
        for m in range(num_m):
            q_lo = m * m_block
            q_hi = q_lo + m_block
            if q_hi <= kv_lo:
                continue
            if q_lo >= kv_hi - 1:
                full.append(m)
            else:
                partial.append(m)
        mask_lists.append(partial)
        full_lists.append(full)
    return mask_lists, full_lists


def _swa_lists(
    seqlen: int, m_block: int, n_block: int,
    window_left: int, window_right: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Sliding-window mask: each n_block sees only the m_blocks within
    ``[kv - window_left, kv + window_right]``. Yields short, uniform
    segments (max_combined ≈ ceil((wl+wr)/n_block)+2) regardless of
    seqlen -- exercises the tight-bucket Triton path.
    """
    num_m = seqlen // m_block
    num_n = seqlen // n_block
    mask_lists: list[list[int]] = []
    full_lists: list[list[int]] = []
    for n in range(num_n):
        kv_lo = n * n_block
        kv_hi = kv_lo + n_block
        partial, full = [], []
        for m in range(num_m):
            q_lo = m * m_block
            q_hi = q_lo + m_block
            # Window intersects [q_lo, q_hi) with [kv_lo - wl, kv_hi + wr).
            wstart = kv_lo - window_left
            wend = kv_hi + window_right
            if q_hi <= wstart or q_lo >= wend:
                continue
            if q_lo >= wstart and q_hi <= wend:
                full.append(m)
            else:
                partial.append(m)
        mask_lists.append(partial)
        full_lists.append(full)
    return mask_lists, full_lists


def _block_diag_lists(
    seqlen: int, m_block: int, n_block: int, doc_size: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Block-diagonal (document) mask: each n_block sees only m_blocks
    within the same document. ``max_combined`` ≈ ceil(doc_size/m_block);
    exercises a different sparsity distribution than causal (uniform
    per segment vs increasing)."""
    num_m = seqlen // m_block
    num_n = seqlen // n_block
    mask_lists: list[list[int]] = []
    full_lists: list[list[int]] = []
    for n in range(num_n):
        kv_lo = n * n_block
        kv_hi = kv_lo + n_block
        doc_id = kv_lo // doc_size
        doc_lo = doc_id * doc_size
        doc_hi = doc_lo + doc_size
        partial, full = [], []
        for m in range(num_m):
            q_lo = m * m_block
            q_hi = q_lo + m_block
            if q_hi <= doc_lo or q_lo >= doc_hi:
                continue
            if q_lo >= doc_lo and q_hi <= doc_hi \
                    and q_lo >= kv_lo - doc_size and q_hi <= kv_hi + doc_size:
                full.append(m)
            else:
                partial.append(m)
        mask_lists.append(partial)
        full_lists.append(full)
    return mask_lists, full_lists


def _dense_lists(
    num_n: int, num_m: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Dense pattern: every n_block touches every m_block (alternating
    mask/full to keep the 50/50 split). Exercises the maximum-density
    case where ``max_combined == num_m``, pushing BLOCK_K into larger
    buckets (256/512/1024/2048) — important for verifying the env-flag
    equivalence in the *non-default* BLOCK_K bucket caches."""
    mask_lists: list[list[int]] = []
    full_lists: list[list[int]] = []
    for n in range(num_n):
        partial = [m for m in range(num_m) if m % 2 == 0]
        full = [m for m in range(num_m) if m % 2 == 1]
        mask_lists.append(partial)
        full_lists.append(full)
    return mask_lists, full_lists


def _single_entry_lists(num_n: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Each segment has exactly one entry — exercises the smallest
    valid input (max_combined=1, BLOCK_K still floored to 128 by the
    min-bucket policy) and catches off-by-one in the per-segment loop."""
    return [[n] for n in range(num_n)], [[] for _ in range(num_n)]


def _make_sparse_for_pattern(
    pattern: str,
    *,
    seqlen: Optional[int] = None,
    m_block: Optional[int] = None,
    n_block: Optional[int] = None,
    num_n: Optional[int] = None,
    num_m: Optional[int] = None,
    window_left: Optional[int] = None,
    window_right: Optional[int] = None,
    doc_size: Optional[int] = None,
    drop_full: bool = False,
    device: Optional[torch.device] = None,
) -> LinearBlockSparseTensorsTorch:
    """Single entry point that dispatches to the right ``*_lists`` factory.

    Keeps the test parametrisation table compact (each case is just
    ``(pattern_name, kwargs)``) while letting each factory keep its
    natural signature. ``drop_full=True`` rebuilds the sparse with
    ``full_lists=None`` to exercise the ``has_full=False`` code path
    on the same shape -- useful for catching bugs that only surface
    when the kernel's full-load conditional is False.
    """
    assert device is not None
    if pattern == "causal":
        mask_lists, full_lists = _causal_lists(seqlen, m_block, n_block)
    elif pattern == "swa":
        mask_lists, full_lists = _swa_lists(
            seqlen, m_block, n_block, window_left, window_right,
        )
    elif pattern == "block_diag":
        mask_lists, full_lists = _block_diag_lists(
            seqlen, m_block, n_block, doc_size,
        )
    elif pattern == "dense":
        mask_lists, full_lists = _dense_lists(num_n, num_m)
    elif pattern == "single_entry":
        mask_lists, full_lists = _single_entry_lists(num_n)
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    if drop_full:
        # Fold full IDs back into mask, keeping the same per-segment
        # uniqueness invariant. This is *not* semantically the same
        # mask (a kernel can't tell mask from full without is_full) --
        # we use it only to exercise the ``has_full=False`` branch.
        merged = []
        for m_list, f_list in zip(mask_lists, full_lists):
            merged.append(sorted(set(m_list) | set(f_list)))
        return _build_sparse(merged, None, device=device)

    return _build_sparse(mask_lists, full_lists, device=device)


def _make_causal_sparse(
    seqlen: int,
    m_block: int,
    n_block: int,
    device: torch.device,
) -> LinearBlockSparseTensorsTorch:
    """Legacy helper retained for ``test_env_flag_routes_dispatch`` —
    new tests should use ``_make_sparse_for_pattern`` for consistency."""
    return _make_sparse_for_pattern(
        "causal",
        seqlen=seqlen, m_block=m_block, n_block=n_block,
        device=device,
    )


def _max_per_segment(sparse: LinearBlockSparseTensorsTorch) -> Tuple[int, int]:
    """Compute ``(num_m_upper_bound, max_combined)`` from the sparse tensors.

    Done CPU-side via shape/host info (already-cumsummed offsets) to keep
    this test deterministic and dependency-free; a real caller would have
    these as static config (see ``_compute_bwd_dQ_lock_values`` docstring
    for the closed-form bounds per mask pattern).
    """
    mask_cnt = sparse.mask_block_cnt.cpu()
    full_cnt = sparse.full_block_cnt.cpu() if sparse.full_block_cnt is not None \
        else torch.zeros_like(mask_cnt)
    max_combined = int((mask_cnt + full_cnt).max().item())
    # Guard against the empty-CSR case (total=0): max_combined drops to
    # 0 but the kernel expects max_combined >= 1 for its BLOCK_K floor.
    max_combined = max(max_combined, 1)
    max_m = 0
    if sparse.mask_block_idx.numel() > 0:
        max_m = max(max_m, int(sparse.mask_block_idx.max().item()) + 1)
    if (
        sparse.full_block_idx is not None
        and sparse.full_block_idx.numel() > 0
    ):
        max_m = max(max_m, int(sparse.full_block_idx.max().item()) + 1)
    num_n = mask_cnt.shape[0]
    num_m = max(num_n, max_m, 1)
    return num_m, max_combined


def _assert_outputs_bit_identical(
    a: dict, b: dict, *, tag: str,
) -> None:
    """Bit-equal check on all four output tensors of
    ``_compute_bwd_dQ_lock_values``. We use ``torch.equal`` (NOT
    ``allclose``) because the contract is bitwise — any drift indicates
    a non-determinism bug in either path."""
    for key in (
        "dQ_lock_values",
        "dQ_lock_combined_offset",
        "sorted_block_idx",
        "sorted_block_is_full",
    ):
        ta, tb = a[key], b[key]
        assert ta.dtype == tb.dtype, f"[{tag}] {key} dtype mismatch: {ta.dtype} vs {tb.dtype}"
        assert ta.shape == tb.shape, f"[{tag}] {key} shape mismatch: {ta.shape} vs {tb.shape}"
        # Move both to CPU then equality-check. ``torch.equal`` works on
        # GPU too but CPU keeps the assertion message reproducible.
        if not torch.equal(ta.cpu(), tb.cpu()):
            diff_mask = ta.cpu() != tb.cpu()
            n_diff = int(diff_mask.sum().item())
            first_diff = int(diff_mask.nonzero(as_tuple=False)[0].item()) \
                if n_diff > 0 else -1
            raise AssertionError(
                f"[{tag}] {key} bit-mismatch: {n_diff}/{ta.numel()} elems differ, "
                f"first @ {first_diff}: triton={ta.cpu().flatten()[first_diff].item()}, "
                f"pytorch={tb.cpu().flatten()[first_diff].item()}"
            )


# ---------------------------------------------------------------------------
# The contract test
# ---------------------------------------------------------------------------


HINT_MODES = ("none", "num_m", "max_combined", "both", "loose_cap")
"""All caller-hint combinations the env-flag dispatches through.

Each mode exercises a distinct branch of the Triton fast-path's
3-way ``num_m_blocks_actual`` / ``max_combined_hint`` dispatch
(plus the PyTorch fallback which honours ``num_m_blocks_actual``
for its own sync-elision). Every test that varies over patterns
runs the full mode matrix to make sure the env-flag equivalence
holds at *every* call shape, not just the default one.

  "none"        : no hints (1 coalesced sync on Triton path)
  "num_m"       : num_m_blocks_actual only (1 sync max_combined)
  "max_combined": max_combined_hint only (1 sync num_m)
  "both"        : both hints (0 sync — fully async Triton path)
  "loose_cap"   : max_combined_hint over the cap (Triton path silently
                  falls back to the synced derivation)
"""


def _build_hint_kwargs(
    hint_mode: str,
    *,
    num_m: int,
    max_combined: int,
) -> dict:
    """Return the kwargs dict ``_compute_bwd_dQ_lock_values`` should be
    called with for the given hint mode. Centralised so every test that
    varies hint_mode applies the same semantics consistently."""
    if hint_mode == "none":
        return {}
    if hint_mode == "num_m":
        return {"num_m_blocks_actual": num_m}
    if hint_mode == "max_combined":
        return {"max_combined_hint": max_combined}
    if hint_mode == "both":
        return {
            "num_m_blocks_actual": num_m,
            "max_combined_hint": max_combined,
        }
    if hint_mode == "loose_cap":
        return {
            "num_m_blocks_actual": num_m,
            # Push max_combined_hint past _DQ_LOCK_MAX_HINT_BLOCK_K to
            # trigger the silent fallback-to-synced-derivation branch.
            # Output must still be bit-identical; only the perf
            # characteristic changes.
            "max_combined_hint": _iface._DQ_LOCK_MAX_HINT_BLOCK_K * 4,
        }
    raise ValueError(hint_mode)


def _flip_flag_and_compare(
    sparse: LinearBlockSparseTensorsTorch,
    *,
    hint_mode: str,
    tag: str,
) -> None:
    """Common test body: run with flag OFF, run with flag ON, assert
    bit-equal. Restored flag is preserved across exceptions."""
    if not _iface._HAS_TRITON:
        pytest.skip("Triton not installed -- nothing to compare")

    num_m, max_combined = _max_per_segment(sparse)
    kwargs = _build_hint_kwargs(
        hint_mode, num_m=num_m, max_combined=max_combined,
    )

    saved = _iface._DQ_LOCK_USE_TRITON_ENABLED
    try:
        _iface._DQ_LOCK_USE_TRITON_ENABLED = False
        out_pytorch = _compute_bwd_dQ_lock_values(sparse, **kwargs)

        _iface._DQ_LOCK_USE_TRITON_ENABLED = True
        out_triton = _compute_bwd_dQ_lock_values(sparse, **kwargs)
    finally:
        _iface._DQ_LOCK_USE_TRITON_ENABLED = saved

    _assert_outputs_bit_identical(
        out_triton, out_pytorch, tag=f"{tag} hint={hint_mode}",
    )


# ---------------------------------------------------------------------------
# Pattern: causal — primary production pattern. Sweeps shape × hint mode.
# ---------------------------------------------------------------------------

CAUSAL_SHAPES = [
    # Common pow-of-2 production sizes
    (2048, 64, 128),
    (2048, 128, 128),
    (8192, 128, 128),
    (32768, 128, 128),
    # Cross-tile (m_block != n_block) -- catches assumptions that the
    # two block sizes are equal.
    (8192, 64, 128),
    (8192, 128, 64),
    # Hopper sm90 tile (m_block=80, not pow2) -- exercises ceildiv
    # rounding in the m-axis arithmetic and a less-common BLOCK_K bucket.
    (8192, 80, 128),
    # Irregular seqlen -- catches off-by-one in num_m / max_combined.
    (1500, 128, 64),
    (4097, 128, 128),
    (12345, 128, 128),
    # Long-seqlen production scale -- exercises the larger BLOCK_K
    # bucket (max_combined ≈ num_m = 1024 -> BLOCK_K=1024).
    (131072, 128, 128),
]


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape", CAUSAL_SHAPES,
    ids=[f"sq{s[0]}_mb{s[1]}_nb{s[2]}" for s in CAUSAL_SHAPES],
)
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_causal(
    shape: Tuple[int, int, int], hint_mode: str,
) -> None:
    """Causal × all 5 hint modes × all shapes. With 11 shapes × 5 modes
    this is 55 sub-cases; covers BLOCK_K buckets {128, 256, 512, 1024}
    via different ``max_combined`` per shape."""
    seqlen, m_block, n_block = shape
    sparse = _make_sparse_for_pattern(
        "causal", seqlen=seqlen, m_block=m_block, n_block=n_block,
        device=torch.device("cuda"),
    )
    _flip_flag_and_compare(
        sparse, hint_mode=hint_mode,
        tag=f"causal sq={seqlen} mb={m_block} nb={n_block}",
    )


# ---------------------------------------------------------------------------
# Pattern: sliding-window (left / right / sym)
# ---------------------------------------------------------------------------

SWA_CASES = [
    # (seqlen, m_block, n_block, wl, wr)
    (2048, 64, 128, 512, 512),       # symmetric small window
    (8192, 128, 128, 2048, 2048),
    (32768, 128, 128, 4096, 4096),
    (8192, 128, 128, 2048, 0),       # causal-flavoured
    (32768, 128, 128, 8192, 0),
    (8192, 128, 128, 256, 256),      # tiny window (max_combined ≈ 3)
    (131072, 128, 128, 4096, 4096),  # long context inference
]


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case", SWA_CASES,
    ids=[f"sq{c[0]}_mb{c[1]}_nb{c[2]}_wl{c[3]}_wr{c[4]}" for c in SWA_CASES],
)
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_swa(
    case: Tuple[int, int, int, int, int], hint_mode: str,
) -> None:
    """SWA: tight uniform segments exercise the smallest BLOCK_K bucket
    (128) at varying num_n. ``loose_cap`` is interesting here because
    the natural ``max_combined`` is much smaller than the cap -- catches
    any logic that mishandles tight-vs-loose-hint selection when the
    hint is artificially inflated."""
    seqlen, m_block, n_block, wl, wr = case
    sparse = _make_sparse_for_pattern(
        "swa", seqlen=seqlen, m_block=m_block, n_block=n_block,
        window_left=wl, window_right=wr,
        device=torch.device("cuda"),
    )
    _flip_flag_and_compare(
        sparse, hint_mode=hint_mode,
        tag=f"swa sq={seqlen} wl={wl} wr={wr}",
    )


# ---------------------------------------------------------------------------
# Pattern: block-diagonal (document mask)
# ---------------------------------------------------------------------------

BLOCK_DIAG_CASES = [
    # (seqlen, m_block, n_block, doc_size)
    (8192, 128, 128, 1024),
    (8192, 128, 128, 2048),
    (32768, 128, 128, 4096),
    (32768, 128, 128, 8192),
    # doc_size NOT a multiple of n_block -- exercises off-diagonal
    # straddling segments with varying mix of mask + full.
    (8192, 128, 128, 1500),
]


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case", BLOCK_DIAG_CASES,
    ids=[f"sq{c[0]}_doc{c[3]}" for c in BLOCK_DIAG_CASES],
)
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_block_diag(
    case: Tuple[int, int, int, int], hint_mode: str,
) -> None:
    """Block-diagonal: per-segment counts cluster around doc boundaries,
    giving a different distribution than causal. Exercises the kernel's
    per-segment loop with non-monotone segment sizes."""
    seqlen, m_block, n_block, doc_size = case
    sparse = _make_sparse_for_pattern(
        "block_diag",
        seqlen=seqlen, m_block=m_block, n_block=n_block,
        doc_size=doc_size,
        device=torch.device("cuda"),
    )
    _flip_flag_and_compare(
        sparse, hint_mode=hint_mode,
        tag=f"block_diag sq={seqlen} doc={doc_size}",
    )


# ---------------------------------------------------------------------------
# Pattern: dense — every n touches every m. Exercises max BLOCK_K bucket.
# ---------------------------------------------------------------------------

DENSE_CASES = [
    # (num_n, num_m) — direct matrix shape control. Larger num_m
    # pushes BLOCK_K into bigger buckets, validating env-flag
    # equivalence at every BLOCK_K constexpr value the kernel JITs.
    (16, 16),      # BLOCK_K=128 (floor)
    (64, 64),      # BLOCK_K=128 (floor)
    (128, 128),    # BLOCK_K=128 (floor)
    (256, 256),    # BLOCK_K=256 (first non-floor bucket)
    (512, 512),    # BLOCK_K=512
    (1024, 1024),  # BLOCK_K=1024
]


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case", DENSE_CASES, ids=[f"n{c[0]}_m{c[1]}" for c in DENSE_CASES],
)
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_dense(
    case: Tuple[int, int], hint_mode: str,
) -> None:
    """Dense stress-tests the large-BLOCK_K kernels. With num_m=1024
    the Triton sort fits in a 1024-wide bucket; bit-equal across the
    flag flip ensures the larger-bucket kernel variants (JIT-compiled
    separately from BLOCK_K=128) are correct -- a non-trivial extension
    of coverage."""
    num_n, num_m = case
    sparse = _make_sparse_for_pattern(
        "dense", num_n=num_n, num_m=num_m,
        device=torch.device("cuda"),
    )
    _flip_flag_and_compare(
        sparse, hint_mode=hint_mode, tag=f"dense n{num_n} m{num_m}",
    )


# ---------------------------------------------------------------------------
# Pattern: mask-only (``has_full=False``) — exercises the conditional
# ``full`` load branch in the Triton kernel.
# ---------------------------------------------------------------------------

NO_FULL_SHAPES = [
    (2048, 128, 128),
    (8192, 128, 128),
    (32768, 128, 128),
]


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape", NO_FULL_SHAPES,
    ids=[f"sq{s[0]}_mb{s[1]}_nb{s[2]}" for s in NO_FULL_SHAPES],
)
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_no_full_blocks(
    shape: Tuple[int, int, int], hint_mode: str,
) -> None:
    """``has_full=False`` -- the kernel's full-block load is skipped
    via conditional execution. The dispatch logic's ``has_full`` checks
    differ from the main path, so this exercise the alternate branch
    across all hint modes."""
    seqlen, m_block, n_block = shape
    sparse = _make_sparse_for_pattern(
        "causal", seqlen=seqlen, m_block=m_block, n_block=n_block,
        drop_full=True, device=torch.device("cuda"),
    )
    _flip_flag_and_compare(
        sparse, hint_mode=hint_mode,
        tag=f"no_full sq={seqlen} mb={m_block} nb={n_block}",
    )


# ---------------------------------------------------------------------------
# Edge cases — small / degenerate inputs.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_single_n_block(hint_mode: str) -> None:
    """``num_n=1`` -- the kernel grid is just one program; ensures the
    single-segment path agrees across the flag flip. Catches bugs that
    only surface without inter-segment boundary handling."""
    sparse = _make_sparse_for_pattern(
        "dense", num_n=1, num_m=64, device=torch.device("cuda"),
    )
    _flip_flag_and_compare(sparse, hint_mode=hint_mode, tag="single_n_block")


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_single_entry_per_segment(hint_mode: str) -> None:
    """``max_combined=1`` -- the smallest non-trivial bucket. Pads up
    to BLOCK_K=128 (min-bucket floor) and verifies the per-segment sort
    handles 127 INVALID-padded entries correctly across the flag flip."""
    sparse = _make_sparse_for_pattern(
        "single_entry", num_n=64, device=torch.device("cuda"),
    )
    _flip_flag_and_compare(
        sparse, hint_mode=hint_mode, tag="single_entry_per_segment",
    )


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hint_mode", HINT_MODES)
def test_env_flag_equivalence_empty_csr(hint_mode: str) -> None:
    """``total=0`` -- both paths hit the early-return branch. Validates
    the early-return is wired identically on both flag settings (would
    catch a divergence where one path returns sentinel tensors and the
    other returns ``None``)."""
    num_n = 8
    mask_lists = [[] for _ in range(num_n)]
    full_lists = [[] for _ in range(num_n)]
    sparse = _build_sparse(mask_lists, full_lists, device=torch.device("cuda"))
    _flip_flag_and_compare(sparse, hint_mode=hint_mode, tag="empty_csr")


# ---------------------------------------------------------------------------
# Heuristic boundary -- num_n straddling _DQ_LOCK_TRITON_NUM_N_THRESHOLD.
# When num_n > threshold the flag-True path *transparently* falls back
# to PyTorch (Triton is slower there); guards that this transparent
# fallback still produces bit-identical output to the flag-False path.
# ---------------------------------------------------------------------------

THRESHOLD_CASES = [
    1000,    # well below
    4095,    # just below
    4096,    # exactly at (still <= threshold)
    4097,    # just above (flag-on path falls back to PyTorch)
    8000,    # well above (typical bh-flattened large-num_n)
]


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_n", THRESHOLD_CASES)
def test_env_flag_equivalence_at_threshold(num_n: int) -> None:
    """Heuristic boundary: both the "below-threshold Triton" and
    "above-threshold PyTorch-fallback" branches must yield bit-identical
    output when the flag is flipped. We use a sparse block-diag-like
    pattern to keep memory under control even at num_n=8000."""
    num_m = max(64, num_n // 16)
    ms_per_n = 4
    mask_lists = [
        [(n * 3 + j) % num_m for j in range(ms_per_n // 2)]
        for n in range(num_n)
    ]
    full_lists = [
        [(n * 7 + j + ms_per_n // 2) % num_m for j in range(ms_per_n // 2)]
        for n in range(num_n)
    ]
    for n in range(num_n):
        combined = sorted(set(mask_lists[n]) | set(full_lists[n]))
        half = len(combined) // 2
        mask_lists[n] = combined[:half]
        full_lists[n] = combined[half:]

    sparse = _build_sparse(mask_lists, full_lists, device=torch.device("cuda"))
    _flip_flag_and_compare(
        sparse, hint_mode="both",
        tag=f"threshold num_n={num_n}",
    )


# ---------------------------------------------------------------------------
# Determinism guard -- repeated runs under each flag setting must agree.
# Catches non-determinism (e.g., a race in the Triton path) that the
# single-shot bit-equal test would miss.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_env_flag_repeated_calls_deterministic() -> None:
    """4 calls (2 flag settings × 2 reps) must all match bit-equal.
    Catches non-determinism the single-shot equivalence test would miss
    -- e.g., a Triton race that happens to give the same output as
    PyTorch most of the time but flickers."""
    if not _iface._HAS_TRITON:
        pytest.skip("Triton not installed")

    device = torch.device("cuda")
    sparse = _make_sparse_for_pattern(
        "causal", seqlen=8192, m_block=128, n_block=128, device=device,
    )

    saved = _iface._DQ_LOCK_USE_TRITON_ENABLED
    try:
        _iface._DQ_LOCK_USE_TRITON_ENABLED = False
        py_a = _compute_bwd_dQ_lock_values(sparse)
        py_b = _compute_bwd_dQ_lock_values(sparse)
        _assert_outputs_bit_identical(py_a, py_b, tag="pytorch_repeat")

        _iface._DQ_LOCK_USE_TRITON_ENABLED = True
        tr_a = _compute_bwd_dQ_lock_values(sparse)
        tr_b = _compute_bwd_dQ_lock_values(sparse)
        _assert_outputs_bit_identical(tr_a, tr_b, tag="triton_repeat")
        _assert_outputs_bit_identical(py_a, tr_a, tag="cross_flag")
    finally:
        _iface._DQ_LOCK_USE_TRITON_ENABLED = saved


# ---------------------------------------------------------------------------
# Dtype variant -- int64 idx tensors (defensive cast path in Triton).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_env_flag_with_int64_idx() -> None:
    """``mask_block_idx`` / ``full_block_idx`` as int64. The Triton
    path defensively casts to int32; the PyTorch fallback handles int64
    natively. Output must agree across the flag flip."""
    if not _iface._HAS_TRITON:
        pytest.skip("Triton not installed")

    device = torch.device("cuda")
    mask_lists, full_lists = _causal_lists(4096, 128, 128)
    num_n = len(mask_lists)
    mask_cnt = torch.tensor(
        [len(l) for l in mask_lists], dtype=torch.int32, device=device,
    )
    mask_off = torch.zeros(num_n + 1, dtype=torch.int32, device=device)
    mask_off[1:] = torch.cumsum(mask_cnt, dim=0)
    mask_idx_i64 = torch.tensor(
        [m for l in mask_lists for m in l], dtype=torch.int64, device=device,
    )
    full_cnt = torch.tensor(
        [len(l) for l in full_lists], dtype=torch.int32, device=device,
    )
    full_off = torch.zeros(num_n + 1, dtype=torch.int32, device=device)
    full_off[1:] = torch.cumsum(full_cnt, dim=0)
    full_idx_i64 = torch.tensor(
        [m for l in full_lists for m in l], dtype=torch.int64, device=device,
    )

    sparse_i64 = LinearBlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt, mask_block_offset=mask_off,
        mask_block_idx=mask_idx_i64,
        full_block_cnt=full_cnt, full_block_offset=full_off,
        full_block_idx=full_idx_i64,
    )
    _flip_flag_and_compare(sparse_i64, hint_mode="both", tag="int64_idx")


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
def test_env_flag_default_is_off() -> None:
    """Production default contract: ``MAGI_FA4_DQ_LOCK_USE_TRITON`` is OFF
    unless explicitly set. This guards against an accidental code change
    flipping the default — which would silently shift production
    behaviour (and JIT-compile latency profile) on upgrade.

    Note: this asserts the *module-import-time* default, which is
    derived from the env var. If the test runner sets the env var, this
    test is meaningful only when run with the env var unset, so we
    inspect what the parser *would* yield for an unset env, not the
    cached attribute (which may already be True if CI set the env)."""
    import os
    # Recompute the default the same way the module does at import.
    raw = os.environ.get("MAGI_FA4_DQ_LOCK_USE_TRITON", "")
    truthy = raw.strip().lower() in ("1", "true", "yes", "on", "y", "t")
    if not truthy:
        # In a clean env, the cached flag must agree with our recomputation.
        assert _iface._DQ_LOCK_USE_TRITON_ENABLED is False, (
            "Production default for MAGI_FA4_DQ_LOCK_USE_TRITON must be "
            "OFF when the env var is unset/false. Got True -- check the "
            "module-level parser in flash_attn/cute/interface.py."
        )


@pytest.mark.skipif(not _CAN_IMPORT, reason="flash_attn_cute import failed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_env_flag_routes_dispatch() -> None:
    """The flag must actually route dispatch, not just be a vestigial
    boolean. We probe by monkey-patching the Triton helper to record
    whether it was called -- more robust than relying on timing."""
    if not _iface._HAS_TRITON:
        pytest.skip("Triton not installed")

    device = torch.device("cuda")
    sparse = _make_causal_sparse(2048, 64, 128, device)

    saved_flag = _iface._DQ_LOCK_USE_TRITON_ENABLED
    saved_helper = _iface._compute_dq_lock_values_triton_linear
    triton_called = {"n": 0}

    def _probe(*args, **kwargs):
        triton_called["n"] += 1
        return saved_helper(*args, **kwargs)

    try:
        _iface._compute_dq_lock_values_triton_linear = _probe

        _iface._DQ_LOCK_USE_TRITON_ENABLED = False
        _compute_bwd_dQ_lock_values(sparse)
        assert triton_called["n"] == 0, (
            "Triton helper called even though flag is False -- dispatch "
            "is not honouring _DQ_LOCK_USE_TRITON_ENABLED."
        )

        _iface._DQ_LOCK_USE_TRITON_ENABLED = True
        _compute_bwd_dQ_lock_values(sparse)
        assert triton_called["n"] == 1, (
            "Triton helper NOT called when flag is True -- dispatch is "
            "not honouring _DQ_LOCK_USE_TRITON_ENABLED."
        )
    finally:
        _iface._compute_dq_lock_values_triton_linear = saved_helper
        _iface._DQ_LOCK_USE_TRITON_ENABLED = saved_flag
