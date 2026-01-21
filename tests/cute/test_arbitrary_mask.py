# mask mod test script
# REFACTORED to use _flash_attn_fwd as the kernel entrypoint
#
# Test Organization:
# - test_static_masks: Fast tests for masks that don't need per-seqlen compilation
#   (identity, document, block_diagonal, etc.) with comprehensive seqlen coverage
# - test_parameterized_masks: Slower tests for masks that require recompilation per
#   seqlen pair (causal, block_causal, sliding_window) with reduced seqlen coverage
#
# Usage:
#   pytest test_mask_mod.py::test_static_masks         # Run only fast tests
#   pytest test_mask_mod.py::test_parameterized_masks  # Run only slow tests
#   pytest test_mask_mod.py                            # Run all tests

import math
from typing import Optional
from einops import rearrange, repeat

import torch
import pytest
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch, bhqk_to_linear_sparse_tensors, LinearBlockSparseTensorsTorch

# Import CUDA kernel for create_block_mask
try:
    import create_block_mask_cuda
    HAS_CREATE_BLOCK_MASK_CUDA = True
except ImportError:
    # Try to build and install create_block_mask_cuda
    import subprocess
    import os
    print("create_block_mask_cuda not found. Attempting to build and install...")
    utils_dir = os.path.join(os.path.dirname(__file__), "../../csrc/utils")
    utils_dir = os.path.abspath(utils_dir)
    try:
        # Run make to build the module
        result = subprocess.run(
            ["make", "create_block_mask"],
            cwd=utils_dir,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Build successful. Importing create_block_mask_cuda...")
            import create_block_mask_cuda
            HAS_CREATE_BLOCK_MASK_CUDA = True
        else:
            print(f"Build failed: {result.stderr}")
            create_block_mask_cuda = None
            HAS_CREATE_BLOCK_MASK_CUDA = False
    except Exception as e:
        print(f"Failed to build create_block_mask_cuda: {e}")
        create_block_mask_cuda = None
        HAS_CREATE_BLOCK_MASK_CUDA = False
        print("Warning: create_block_mask_cuda not found. Using PyTorch reference implementation.")
from flash_attn.cute.mask_definitions import (
    get_mask_pair,
    STATIC_MASKS,
    arbitrary_func_tensor,
)
COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]

def apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k):
    """
    qk_attn: [B, H, seqlen_q, seqlen_k]
    arbitrary_func: [b, h, func_num, seqlen_q + 256] where b=1 or B, h=1 or H
    Only the first `seqlen_q` positions of the last dim are used.
    """
    device = qk_attn.device
    B, H, sq, sk = qk_attn.shape
    assert sq == seqlen_q and sk == seqlen_k

    b_size, h_size = arbitrary_func.shape[0], arbitrary_func.shape[1]

    # Use only the first seqlen_q entries along last dim
    af = arbitrary_func[..., :seqlen_q]   # [b, h, func_num, seqlen_q]
    func_num = af.shape[2]

    # j grid: [1, 1, 1, seqlen_k]
    j = torch.arange(seqlen_k, device=device, dtype=af.dtype).view(1, 1, 1, seqlen_k)

    # Initialize valid mask as all False
    valid = torch.zeros(B, H, seqlen_q, seqlen_k, dtype=torch.bool, device=device)

    for b in range(B):
        for h in range(H):
            # Support broadcasting
            b_idx = 0 if b_size == 1 else b
            h_idx = 0 if h_size == 1 else h

            # base cutoff: af[b_idx, h_idx, 0, :] -> [seqlen_q, 1]
            base_cut = af[b_idx, h_idx, 0, :].view(seqlen_q, 1)
            base_valid = j.squeeze(0).squeeze(0) < base_cut  # [seqlen_q, seqlen_k]

            # intervals
            num_intervals = func_num // 2
            if num_intervals > 0:
                starts = af[b_idx, h_idx, 1:func_num:2, :]  # [num_intervals, seqlen_q]
                ends = af[b_idx, h_idx, 2:func_num:2, :]    # [num_intervals, seqlen_q]

                # reshape for broadcasting: [num_intervals, seqlen_q, 1]
                starts = starts.view(num_intervals, seqlen_q, 1)
                ends = ends.view(num_intervals, seqlen_q, 1)

                # j expanded: [num_intervals, seqlen_q, seqlen_k]
                j_exp = j.squeeze(0).squeeze(0).expand(num_intervals, seqlen_q, seqlen_k)

                in_interval = (j_exp >= starts) & (j_exp < ends)  # [num_intervals, seqlen_q, seqlen_k]
                interval_valid = in_interval.any(dim=0)  # [seqlen_q, seqlen_k]
            else:
                interval_valid = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=device)

            valid[b, h] = base_valid | interval_valid

    # set invalid positions to -inf
    qk_attn = torch.where(valid, qk_attn, torch.full_like(qk_attn, -float("inf")))

    return qk_attn


def create_tensors(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
):
    device = "cuda"
    q = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
    k = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    ).uniform_(-1, 1).requires_grad_(True)
    v = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    ).uniform_(-1, 1).requires_grad_(True)
    out = torch.empty(
        batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, nheads, seqlen_q, device=device, dtype=torch.float32)

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "out": out.contiguous(),
        "lse": lse.contiguous(),
        "cu_seqlens_q": None,
        "cu_seqlens_k": None,
    }

def compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False):
    """Compute reference of arbitrary mask"""
    q = tensors["q"] if not up_cast else tensors["q"].float()
    k = tensors["k"] if not up_cast else tensors["k"].float()
    v = tensors["v"] if not up_cast else tensors["v"].float()
    batch_size = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    nheads = q.shape[2]
    nheads_kv = k.shape[2]
    headdim = q.shape[3]
    scale = 1.0 / math.sqrt(headdim)

    if nheads_kv == nheads:
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q * scale,
            k,
        )
    else:
        k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q * scale,
            k,
        )

    qk_attn = apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k)

    all_inf_mask = torch.all(qk_attn == float('-inf'), dim=-1, keepdim=True)  # [B, H, seqlen_q, 1]
    softmax_attn = F.softmax(qk_attn, dim=-1)
    out = torch.einsum(
        "bhnm,bmhd->bnhd",
        softmax_attn,
        v,
    )

    is_all_zero = torch.count_nonzero(arbitrary_func) == 0
    out.fill_(0.0) if is_all_zero else out

    return out

def _run_mask_test(
    seqlen_q,
    seqlen_k,
    batch_size,
    nheads,
    arb_multi_batch,
    arb_multi_heads,
    func_num,
    kv_mode,
    headdim,
    dtype,
    tile_m,
    tile_n,
    use_block_sparsity,
    run_benchmark,
):
    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
    elif kv_mode == "gqa":
        nheads_kv = max(1, nheads // 2)
    elif kv_mode == "mqa":
        nheads_kv = 1
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    headdim_v = headdim

    # aux_tensors_arg = None
    # mask_mod_cute, mask_mod_flex = get_mask_pair("causal", seqlen_q, seqlen_k)
    mask_mod_cute, mask_mod_flex = get_mask_pair("arbitrary")
    arb_batch_size = batch_size if arb_multi_batch else 1
    arb_nheads = nheads if arb_multi_heads else 1
    arbitrary_func = arbitrary_func_tensor(arb_batch_size, arb_nheads, func_num, seqlen_q, seqlen_k, device="cuda", pattern="causal")
    original_flex_mask = mask_mod_flex

    def mask_mod_flex(b, h, q_idx, kv_idx, arbitrary_func=arbitrary_func):
        return original_flex_mask(b, h, q_idx, kv_idx, arbitrary_func)

    aux_tensors_arg = [arbitrary_func]
    causal = False

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )
    softmax_scale = 1.0 / math.sqrt(headdim)

    # Compute block sparsity for mask_mod
    if COMPUTE_CAPABILITY == 10:
        sparse_tile_m = 2 * tile_m
    else:
        sparse_tile_m = tile_m

    # =========================================================================
    # Method 1: Use PyTorch's create_block_mask (reference)
    # =========================================================================
    bm = create_block_mask(
        mask_mod_flex,
        arb_batch_size,
        arb_nheads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx = None, None, None, None
    if isinstance(bm.as_tuple()[0], int):
        _, _, k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm.as_tuple()
    else:
        k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm.as_tuple()

    k_block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=k_mask_cnt,
        mask_block_idx=k_mask_idx,
        full_block_cnt=k_full_cnt,
        full_block_idx=k_full_idx,
    )
    ref_linear_k_block_sparse_mask = bhqk_to_linear_sparse_tensors(k_block_sparse_mask)

    bm_bwd = create_block_mask(
        mask_mod_flex,
        arb_batch_size,
        arb_nheads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(128, 128) if COMPUTE_CAPABILITY == 10 else (64, 128), # casual 64 128, non causal 80 128
    )
    q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx = None, None, None, None
    if isinstance(bm_bwd.as_tuple()[0], int):
        _, _, _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
    else:
        _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
    q_block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=q_mask_cnt,
        mask_block_idx=q_mask_idx,
        full_block_cnt=q_full_cnt,
        full_block_idx=q_full_idx,
    )
    ref_linear_q_block_sparse_mask = bhqk_to_linear_sparse_tensors(q_block_sparse_mask)

    # =========================================================================
    # Method 2: Use CUDA kernel (create_block_mask_cuda)
    # =========================================================================
    if HAS_CREATE_BLOCK_MASK_CUDA:
        # nvtx tag
        with torch.cuda.nvtx.range("create_q2k_csr_sparse_from_func"):
            # Q2K (Forward): fix q_block, loop kv_blocks
            (cuda_k_mask_cnt, cuda_k_mask_offset, cuda_k_mask_idx,
            cuda_k_full_cnt, cuda_k_full_offset, cuda_k_full_idx) = \
                create_block_mask_cuda.create_q2k_csr_sparse_from_func(
                    arbitrary_func,
                    seqlen_q,
                    seqlen_k,
                    Q_BLOCK_SIZE=sparse_tile_m,
                    KV_BLOCK_SIZE=tile_n,
                    check_q_boundary=False  # True just to align to flex attention, change to False for performance
                )

            # Convert to LinearBlockSparseTensorsTorch format
            # CUDA kernel returns:
            #   - cnt: [B, H, num_blocks] - directly the counts
            #   - offset: [B * H * num_blocks + 1] - flattened exclusive prefix sum (starts with 0)
            #   - idx: [total_blocks] - compact indices
            cuda_linear_k_block_sparse_mask = LinearBlockSparseTensorsTorch(
                mask_block_cnt=cuda_k_mask_cnt,
                mask_block_offset=cuda_k_mask_offset,
                mask_block_idx=cuda_k_mask_idx,
                full_block_cnt=cuda_k_full_cnt,
                full_block_offset=cuda_k_full_offset,
                full_block_idx=cuda_k_full_idx,
            )
        with torch.cuda.nvtx.range("create_k2q_csr_sparse_from_func"):
            # K2Q (Backward): fix kv_block, loop q_blocks
            # Note: Q_BLOCK_SIZE must match PyTorch's bm_bwd BLOCK_SIZE[0]
            k2q_q_block_size = 128 if COMPUTE_CAPABILITY == 10 else 64
            (cuda_q_mask_cnt, cuda_q_mask_offset, cuda_q_mask_idx,
            cuda_q_full_cnt, cuda_q_full_offset, cuda_q_full_idx) = \
                create_block_mask_cuda.create_k2q_csr_sparse_from_func(
                    arbitrary_func,
                    seqlen_q,
                    seqlen_k,
                    Q_BLOCK_SIZE=k2q_q_block_size,
                    KV_BLOCK_SIZE=128
                )

            cuda_linear_q_block_sparse_mask = LinearBlockSparseTensorsTorch(
                mask_block_cnt=cuda_q_mask_cnt,
                mask_block_offset=cuda_q_mask_offset,
                mask_block_idx=cuda_q_mask_idx,
                full_block_cnt=cuda_q_full_cnt,
                full_block_offset=cuda_q_full_offset,
                full_block_idx=cuda_q_full_idx,
            )

        # =====================================================================
        # Verify CUDA kernel output matches PyTorch reference
        # =====================================================================
        def compare_linear_sparse_tensors(cuda_tensors, ref_tensors, name):
            """Compare CUDA kernel output with PyTorch reference."""
            all_match = True

            # Compare mask_block_cnt
            if not torch.equal(cuda_tensors.mask_block_cnt, ref_tensors.mask_block_cnt):
                print(f"  {name} mask_block_cnt MISMATCH!")
                print(f"    CUDA shape: {cuda_tensors.mask_block_cnt.shape}, ref shape: {ref_tensors.mask_block_cnt.shape}")
                print(f"    CUDA sum: {cuda_tensors.mask_block_cnt.sum().item()}, ref sum: {ref_tensors.mask_block_cnt.sum().item()}")
                all_match = False

            # Compare mask_block_offset
            if not torch.equal(cuda_tensors.mask_block_offset, ref_tensors.mask_block_offset):
                print(f"  {name} mask_block_offset MISMATCH!")
                print(f"    CUDA shape: {cuda_tensors.mask_block_offset.shape}, ref shape: {ref_tensors.mask_block_offset.shape}")
                all_match = False

            # Compare mask_block_idx (values should match, order may differ within each block)
            if cuda_tensors.mask_block_idx.shape != ref_tensors.mask_block_idx.shape:
                print(f"  {name} mask_block_idx shape MISMATCH!")
                print(f"    CUDA: {cuda_tensors.mask_block_idx.shape}, ref: {ref_tensors.mask_block_idx.shape}")
                all_match = False
            elif not torch.equal(cuda_tensors.mask_block_idx, ref_tensors.mask_block_idx):
                # Check if values are the same but in different order
                if set(cuda_tensors.mask_block_idx.tolist()) != set(ref_tensors.mask_block_idx.tolist()):
                    print(f"  {name} mask_block_idx values MISMATCH!")
                    all_match = False

            # Compare full_block_cnt
            if cuda_tensors.full_block_cnt is not None and ref_tensors.full_block_cnt is not None:
                if not torch.equal(cuda_tensors.full_block_cnt, ref_tensors.full_block_cnt):
                    print(f"  {name} full_block_cnt MISMATCH!")
                    all_match = False

            # Compare full_block_offset
            if cuda_tensors.full_block_offset is not None and ref_tensors.full_block_offset is not None:
                if not torch.equal(cuda_tensors.full_block_offset, ref_tensors.full_block_offset):
                    print(f"  {name} full_block_offset MISMATCH!")
                    all_match = False

            # Compare full_block_idx
            if cuda_tensors.full_block_idx is not None and ref_tensors.full_block_idx is not None:
                if cuda_tensors.full_block_idx.shape != ref_tensors.full_block_idx.shape:
                    print(f"  {name} full_block_idx shape MISMATCH!")
                    all_match = False
                elif not torch.equal(cuda_tensors.full_block_idx, ref_tensors.full_block_idx):
                    if set(cuda_tensors.full_block_idx.tolist()) != set(ref_tensors.full_block_idx.tolist()):
                        print(f"  {name} full_block_idx values MISMATCH!")
                        all_match = False

            return all_match

        print("\n" + "=" * 60)
        print("Comparing CUDA kernel vs PyTorch reference:")
        print("=" * 60)

        k_match = compare_linear_sparse_tensors(cuda_linear_k_block_sparse_mask, ref_linear_k_block_sparse_mask, "Q2K")
        q_match = compare_linear_sparse_tensors(cuda_linear_q_block_sparse_mask, ref_linear_q_block_sparse_mask, "K2Q")

        if k_match and q_match:
            print("✓ All CUDA kernel outputs match PyTorch reference!")
        else:
            print("✗ Some outputs do not match!")

        # Use CUDA kernel output for the attention computation
        linear_k_block_sparse_mask = cuda_linear_k_block_sparse_mask
        linear_q_block_sparse_mask = cuda_linear_q_block_sparse_mask
    else:
        # Fallback to PyTorch reference
        linear_k_block_sparse_mask = ref_linear_k_block_sparse_mask
        linear_q_block_sparse_mask = ref_linear_q_block_sparse_mask

    # =========================================================================
    # Benchmarking mode
    # =========================================================================
    if run_benchmark:
        num_runs = 21

        # Pre-create dout for backward pass
        dout = torch.rand(batch_size, seqlen_q, nheads, headdim_v, device="cuda", dtype=dtype)

        # Create CUDA events for accurate GPU timing
        fwd_start_event = torch.cuda.Event(enable_timing=True)
        fwd_end_event = torch.cuda.Event(enable_timing=True)
        bwd_start_event = torch.cuda.Event(enable_timing=True)
        bwd_end_event = torch.cuda.Event(enable_timing=True)

        # Calculate FLOPS
        # Forward: 4 * B * H * seqlen_q * seqlen_k * headdim (Q*K^T and attn*V)
        # Backward: approximately 2.5x forward (dP*V^T, dP^T*Q, dP*K, Q*K^T)
        fwd_flops = 4 * batch_size * nheads * seqlen_q * seqlen_k * headdim
        bwd_flops = 2.5 * fwd_flops  # Backward is roughly 2.5x forward

        # =====================================================================
        # Benchmark 1: Baseline causal attention (no arbitrary mask)
        # =====================================================================
        baseline_fwd_times = []
        baseline_bwd_times = []

        for i in range(num_runs):
            q = tensors["q"].detach().clone().requires_grad_(True)
            k = tensors["k"].detach().clone().requires_grad_(True)
            v = tensors["v"].detach().clone().requires_grad_(True)

            # Forward pass timing
            fwd_start_event.record()

            out_baseline, lse_baseline = flash_attn_func(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                causal=True,
                arbitrary=False,
                window_size=(None, None),
                learnable_sink=None,
                softcap=0.0,
                num_splits=1,
                pack_gqa=False,
                deterministic=False,
                mask_mod=None,
                linear_k_block_sparse_tensors=None,
                linear_q_block_sparse_tensors=None,
                aux_tensors=None,
            )

            fwd_end_event.record()

            # Backward pass timing
            bwd_start_event.record()

            dq, dk, dv = torch.autograd.grad(
                out_baseline, (q, k, v), dout
            )

            bwd_end_event.record()

            torch.cuda.synchronize()

            # Skip the first run (warmup)
            if i > 0:
                baseline_fwd_times.append(fwd_start_event.elapsed_time(fwd_end_event))
                baseline_bwd_times.append(bwd_start_event.elapsed_time(bwd_end_event))

        # =====================================================================
        # Benchmark 2: Arbitrary mask attention
        # =====================================================================
        arb_fwd_times = []
        arb_bwd_times = []

        for i in range(num_runs):
            q = tensors["q"].detach().clone().requires_grad_(True)
            k = tensors["k"].detach().clone().requires_grad_(True)
            v = tensors["v"].detach().clone().requires_grad_(True)

            # Forward pass timing
            fwd_start_event.record()

            out_cute, lse_cute = flash_attn_func(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                causal=causal,
                arbitrary=True,
                window_size=(None, None),
                learnable_sink=None,
                softcap=0.0,
                num_splits=1,
                pack_gqa=False,
                deterministic=False,
                mask_mod=None,
                linear_k_block_sparse_tensors=linear_k_block_sparse_mask,
                linear_q_block_sparse_tensors=linear_q_block_sparse_mask,
                aux_tensors=aux_tensors_arg,
            )

            fwd_end_event.record()

            # Backward pass timing
            bwd_start_event.record()

            dq, dk, dv = torch.autograd.grad(
                out_cute, (q, k, v), dout
            )

            bwd_end_event.record()

            torch.cuda.synchronize()

            # Skip the first run (warmup)
            if i > 0:
                arb_fwd_times.append(fwd_start_event.elapsed_time(fwd_end_event))
                arb_bwd_times.append(bwd_start_event.elapsed_time(bwd_end_event))

        # =====================================================================
        # Calculate and print results
        # =====================================================================
        baseline_avg_fwd_ms = sum(baseline_fwd_times) / len(baseline_fwd_times)
        baseline_avg_bwd_ms = sum(baseline_bwd_times) / len(baseline_bwd_times)
        arb_avg_fwd_ms = sum(arb_fwd_times) / len(arb_fwd_times)
        arb_avg_bwd_ms = sum(arb_bwd_times) / len(arb_bwd_times)

        baseline_fwd_tflops = fwd_flops / (baseline_avg_fwd_ms * 1e-3) / 1e12
        baseline_bwd_tflops = bwd_flops / (baseline_avg_bwd_ms * 1e-3) / 1e12
        arb_fwd_tflops = fwd_flops / (arb_avg_fwd_ms * 1e-3) / 1e12
        arb_bwd_tflops = bwd_flops / (arb_avg_bwd_ms * 1e-3) / 1e12

        print("\n" + "=" * 70)
        print("Benchmark Results:")
        print("=" * 70)
        print(f"Configuration: B={batch_size}, H={nheads}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, headdim={headdim}")
        print(f"Number of runs: {num_runs} (first run excluded for warmup)")
        print("-" * 70)
        print(f"{'Kernel':<20} {'Forward (ms)':<15} {'Fwd TFLOPS*':<12} {'Backward (ms)':<15} {'Bwd TFLOPS*':<12}")
        print("-" * 70)
        print(f"{'Baseline (causal)':<20} {baseline_avg_fwd_ms:<15.3f} {baseline_fwd_tflops:<12.2f} {baseline_avg_bwd_ms:<15.3f} {baseline_bwd_tflops:<12.2f}")
        print(f"{'Arbitrary mask':<20} {arb_avg_fwd_ms:<15.3f} {arb_fwd_tflops:<12.2f} {arb_avg_bwd_ms:<15.3f} {arb_bwd_tflops:<12.2f}")
        print("-" * 70)
        fwd_overhead = (arb_avg_fwd_ms - baseline_avg_fwd_ms) / baseline_avg_fwd_ms * 100
        bwd_overhead = (arb_avg_bwd_ms - baseline_avg_bwd_ms) / baseline_avg_bwd_ms * 100
        total_baseline = baseline_avg_fwd_ms + baseline_avg_bwd_ms
        total_arb = arb_avg_fwd_ms + arb_avg_bwd_ms
        total_overhead = (total_arb - total_baseline) / total_baseline * 100
        print(f"{'Overhead':<20} {fwd_overhead:+.1f}%{'':<10} {'':<12} {bwd_overhead:+.1f}%{'':<10}")
        print("-" * 70)
        print(f"{'Total time':<20} {total_baseline:<15.3f} {'':<12} {total_arb:<15.3f} {'':12} overhead: {total_overhead:+.1f}%")
        print("=" * 70)
        print("* TFLOPS is calculated based on full mask, not actual mask")

        return  # Skip correctness checks in benchmark mode

    out_cute, lse_cute = flash_attn_func(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        softmax_scale=softmax_scale,
        causal=causal,
        arbitrary=True,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=False,
        deterministic=False,
        mask_mod=None,
        linear_k_block_sparse_tensors=linear_k_block_sparse_mask,
        linear_q_block_sparse_tensors=linear_q_block_sparse_mask,
        aux_tensors=aux_tensors_arg,
    )

    out_ref_fp32 = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=True)
    out_ref = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False)

    print(f"Output max diff: {(out_cute - out_ref_fp32).abs().max().item()}")
    print(f"Pytorch max diff: {(out_ref - out_ref_fp32).abs().max().item()}")

    # Check for invalid values
    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert (out_cute - out_ref_fp32).abs().max().item() <= 2 * (out_ref - out_ref_fp32).abs().max().item()

    dout = torch.rand_like(out_cute)

    dq, dk, dv = torch.autograd.grad(
        out_cute, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref_fp32, dk_ref_fp32, dv_ref_fp32) = torch.autograd.grad(
        out_ref_fp32, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
        out_ref, (tensors["q"], tensors["k"], tensors["v"]), dout
    )

    print(f"dV max diff: {(dv - dv_ref_fp32).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_ref - dv_ref_fp32).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref_fp32).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_ref - dk_ref_fp32).abs().max().item()}")
    print(f"dQ max diff: {(dq - dq_ref_fp32).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_ref - dq_ref_fp32).abs().max().item()}")

    assert (dv - dv_ref_fp32).abs().max().item() <= 5 * (dv_ref - dv_ref_fp32).abs().max().item()
    assert (dk - dk_ref_fp32).abs().max().item() <= 5 * (dk_ref - dk_ref_fp32).abs().max().item()
    assert (dq - dq_ref_fp32).abs().max().item() <= 5 * (dq_ref - dq_ref_fp32).abs().max().item()


@pytest.mark.parametrize("seqlen_q,seqlen_k", [
  (128, 128),
  (256, 256),
  (511, 511),
  (1057, 1057),
  (2123, 2123),
  (4259, 4259),
  (8521, 8521)
])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("arb_multi_batch", [True, False])
@pytest.mark.parametrize("arb_multi_heads", [True, False])
@pytest.mark.parametrize("func_num", [3, 9])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_block_sparsity", [True])
@pytest.mark.parametrize("tile_m,tile_n", [(128, 128)])
@pytest.mark.parametrize("run_benchmark", [False])
def test_arbitrary_mask(
    seqlen_q,
    seqlen_k,
    batch_size,
    nheads,
    arb_multi_batch,
    arb_multi_heads,
    func_num,
    kv_mode,
    headdim,
    dtype,
    use_block_sparsity,
    tile_m,
    tile_n,
    run_benchmark,
):
    """Test arbitrary mask
    """
    if COMPUTE_CAPABILITY == 10 and (tile_m, tile_n) != (128, 128):
        pytest.skip("TODO: Non-128x128 tiles currently not supported on SM 10.0.")

    if COMPUTE_CAPABILITY == 9 and headdim < 128:
        pytest.skip("TODO: hdim > 128 currently not supported on SM 9.0 backward")

    if COMPUTE_CAPABILITY == 9 and kv_mode != "mha":
        pytest.skip("TODO: Non-mha kv_mode currently not supported on SM 9.0 backward")

    _run_mask_test(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        batch_size=batch_size,
        nheads=nheads,
        arb_multi_batch=arb_multi_batch,
        arb_multi_heads=arb_multi_heads,
        func_num=func_num,
        kv_mode=kv_mode,
        headdim=headdim,
        dtype=dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        use_block_sparsity=use_block_sparsity,
        run_benchmark=run_benchmark,
    )


if __name__ == "__main__":
    test_arbitrary_mask(
        seqlen_q=8192,
        seqlen_k=8192,
        batch_size=1,
        nheads=32,
        arb_multi_batch=False,
        arb_multi_heads=True,
        func_num=3,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=True,
        tile_m=128,
        tile_n=128,
        run_benchmark=True,
    )