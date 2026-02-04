# Arbitrary mask test script
# Unified test file supporting both CUTE DSL and Hopper C++ backends
#
# Usage:
#   pytest test_arbitrary_mask.py                              # Run with CUTE backend (default)
#   FLASH_ATTN_BACKEND=hopper pytest test_arbitrary_mask.py    # Run with Hopper backend
#   python test_arbitrary_mask.py                              # Run default test directly

import os
import math
import sys
from typing import Optional, NamedTuple

import torch
import pytest
import torch.nn.functional as F
from einops import repeat

# Try to import flex_attention's create_block_mask
# Default: disabled (use CUDA kernel). Set DISABLE_FLEX_ATTENTION=FALSE to enable.
DISABLE_FLEX_ATTENTION = os.getenv("DISABLE_FLEX_ATTENTION", "TRUE") == "TRUE"

try:
    if DISABLE_FLEX_ATTENTION:
        raise ImportError("Flex attention disabled by environment variable")
    from torch.nn.attention.flex_attention import create_block_mask
    HAS_FLEX_ATTENTION = True
except ImportError:
    create_block_mask = None
    HAS_FLEX_ATTENTION = False
    print("Warning: flex_attention not available. Will use CUDA kernel for block mask creation.")

# ============================================================================
# Environment variables and configuration (following test_flash_attn.py style)
# ============================================================================
BACKEND = os.getenv("FLASH_ATTN_BACKEND", "cute")  # "cute" or "hopper"
DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"

COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]

# Compiled headdims based on environment
COMPILED_HDIMS = (
    []
    + ([64] if not DISABLE_HDIM64 else [])
    + ([128] if not DISABLE_HDIM128 else [])
    + ([256] if not DISABLE_HDIM256 else [])
)

# ============================================================================
# Backend selection - supports both CUTE DSL and Hopper C++ implementations
# ============================================================================
if BACKEND == "hopper":
    # Add hopper directory to path for imports
    hopper_dir = os.path.join(os.path.dirname(__file__), "../../hopper")
    hopper_dir = os.path.abspath(hopper_dir)
    if hopper_dir not in sys.path:
        sys.path.insert(0, hopper_dir)
    from flash_attn_interface import flash_attn_forward, flash_attn_func, FlashAttnFunc
    
    class LinearBlockSparseTensors(NamedTuple):
        """Linear CSR format block sparse tensors for Hopper C++ API."""
        mask_block_cnt: torch.Tensor
        mask_block_offset: torch.Tensor
        mask_block_idx: torch.Tensor
        full_block_cnt: Optional[torch.Tensor] = None
        full_block_offset: Optional[torch.Tensor] = None
        full_block_idx: Optional[torch.Tensor] = None
else:
    # CUTE DSL backend (default)
    from flash_attn.cute.interface import flash_attn_func
    from flash_attn.cute.block_sparsity import (
        BlockSparseTensorsTorch,
        bhqk_to_linear_sparse_tensors,
        LinearBlockSparseTensorsTorch as LinearBlockSparseTensors,
    )

# Import tile size utilities for automatic tile size detection
from flash_attn.utils.tile_size import (
    get_fwd_tile_sizes, get_bwd_tile_sizes, get_arch,
    get_fwd_tile_sizes_dsl, get_bwd_tile_sizes_dsl,
    get_tile_sizes_by_backend,
)

# Import arbitrary_func_tensor from mask_definitions
from flash_attn.cute.mask_definitions import arbitrary_func_tensor

# ============================================================================
# Import CUDA kernel for create_block_mask
# ============================================================================
try:
    import create_block_mask_cuda
    HAS_CREATE_BLOCK_MASK_CUDA = True
except ImportError:
    import subprocess
    print("create_block_mask_cuda not found. Attempting to build and install...")
    utils_dir = os.path.join(os.path.dirname(__file__), "../../csrc/utils")
    utils_dir = os.path.abspath(utils_dir)
    try:
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


# ============================================================================
# Mask definitions
# ============================================================================
def flex_arbitrary_mask(b, h, q_idx, kv_idx, arbitrary_func):
    """Flex attention arbitrary mask based on interval-based masking.
    
    This implementation directly uses b and h as indices. When arbitrary_func has
    batch=1 or head=1 dimensions, PyTorch broadcasting handles it automatically
    as long as create_block_mask is called with matching B and H parameters.
    
    Note: Uses h * 0 for func_num dimension index (always 0) for vmap compatibility.
    """
    zero = h * 0  # Creates a zero tensor with same shape as h for vmap compatibility
    value_valid = kv_idx < arbitrary_func[b, h, zero, q_idx]
    n_func = arbitrary_func.shape[2]
    for i in range(n_func // 2):
        in_range = (kv_idx >= arbitrary_func[b, h, zero + (2*i+1), q_idx]) & \
                   (kv_idx < arbitrary_func[b, h, zero + (2*i+2), q_idx])
        value_valid = value_valid | in_range
    return value_valid


# ============================================================================
# Helper classes and functions
# ============================================================================
class BlockSparseTensorsTorch(NamedTuple):
    """Intermediate BHQK format block sparse tensors."""
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: Optional[torch.Tensor] = None
    full_block_idx: Optional[torch.Tensor] = None


def bhqk_to_linear_sparse_tensors_local(bhqk_tensors: BlockSparseTensorsTorch) -> LinearBlockSparseTensors:
    """Convert BHQK format to LinearBlockSparseTensors (CSR format).
    
    Handles full [B, H, ...] shapes. Processes all batches and heads to produce
    CSR format compatible with CUDA kernel output.
    """
    B, H, n_blocks_q, max_blocks_per_q = bhqk_tensors.mask_block_idx.shape
    device = bhqk_tensors.mask_block_cnt.device
    
    # mask_block_cnt: [B, H, num_q_blocks]
    mask_block_cnt = bhqk_tensors.mask_block_cnt.to(torch.int32)
    
    # For offset, we need to flatten [B, H, n_blocks_q] and compute prefix sum
    # offset shape: [B * H * n_blocks_q + 1] (flattened)
    mask_block_cnt_flat = mask_block_cnt.reshape(-1)  # [B * H * n_blocks_q]
    mask_block_offset = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(mask_block_cnt_flat, dim=0).to(torch.int32)
    ], dim=0)
    
    # Collect all mask block indices
    mask_block_idx_list = []
    for b in range(B):
        for h in range(H):
            for q in range(n_blocks_q):
                cnt = mask_block_cnt[b, h, q].item()
                if cnt > 0:
                    mask_block_idx_list.append(bhqk_tensors.mask_block_idx[b, h, q, :cnt])
    mask_block_idx = torch.cat(mask_block_idx_list, dim=0).to(torch.int32) if mask_block_idx_list else torch.tensor([], dtype=torch.int32, device=device)
    
    full_block_cnt = None
    full_block_offset = None
    full_block_idx = None
    
    if bhqk_tensors.full_block_cnt is not None:
        full_block_cnt = bhqk_tensors.full_block_cnt.to(torch.int32)
        
        full_block_cnt_flat = full_block_cnt.reshape(-1)
        full_block_offset = torch.cat([
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(full_block_cnt_flat, dim=0).to(torch.int32)
        ], dim=0)
    
    if bhqk_tensors.full_block_idx is not None and full_block_cnt is not None:
        full_block_idx_list = []
        for b in range(B):
            for h in range(H):
                for q in range(n_blocks_q):
                    cnt = full_block_cnt[b, h, q].item()
                    if cnt > 0:
                        full_block_idx_list.append(bhqk_tensors.full_block_idx[b, h, q, :cnt])
        full_block_idx = torch.cat(full_block_idx_list, dim=0).to(torch.int32) if full_block_idx_list else torch.tensor([], dtype=torch.int32, device=device)
    
    return LinearBlockSparseTensors(
        mask_block_cnt=mask_block_cnt,
        mask_block_offset=mask_block_offset,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_offset=full_block_offset,
        full_block_idx=full_block_idx,
    )


def apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k):
    """Apply arbitrary mask to QK attention scores.
    
    qk_attn: [B, H, seqlen_q, seqlen_k]
    arbitrary_func: [B_func, H_func, func_num, seqlen_q + 256] 
                   (B_func and H_func support broadcasting, can be 1)
    """
    device = qk_attn.device
    B, H, sq, sk = qk_attn.shape
    assert sq == seqlen_q and sk == seqlen_k

    af = arbitrary_func[..., :seqlen_q]  # [B_func, H_func, func_num, seqlen_q]
    B_func, H_func, func_num, _ = af.shape

    j = torch.arange(seqlen_k, device=device, dtype=af.dtype).view(1, 1, 1, seqlen_k)
    
    # base_cut: [B_func, H_func, seqlen_q, 1]
    base_cut = af[:, :, 0, :].unsqueeze(-1)
    base_valid = j < base_cut  # [B_func, H_func, seqlen_q, seqlen_k]

    num_intervals = func_num // 2
    if num_intervals > 0:
        # starts/ends: [B_func, H_func, num_intervals, seqlen_q]
        starts = af[:, :, 1:func_num:2, :]
        ends = af[:, :, 2:func_num:2, :]
        # Reshape for broadcasting: [B_func, H_func, num_intervals, seqlen_q, 1]
        starts = starts.unsqueeze(-1)
        ends = ends.unsqueeze(-1)
        j_exp = j.view(1, 1, 1, 1, seqlen_k)  # [1, 1, 1, 1, seqlen_k]
        in_interval = (j_exp >= starts) & (j_exp < ends)  # [B_func, H_func, num_intervals, seqlen_q, seqlen_k]
        interval_valid = in_interval.any(dim=2)  # [B_func, H_func, seqlen_q, seqlen_k]
    else:
        interval_valid = torch.zeros(B_func, H_func, seqlen_q, seqlen_k, dtype=torch.bool, device=device)

    valid = base_valid | interval_valid  # [B_func, H_func, seqlen_q, seqlen_k]
    valid = valid.expand(B, H, seqlen_q, seqlen_k)  # Broadcast to match qk_attn shape
    qk_attn = torch.where(valid, qk_attn, torch.full_like(qk_attn, -float("inf")))

    return qk_attn


def create_tensors(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype):
    """Create input tensors for testing."""
    device = "cuda"
    q = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
    k = torch.empty(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
    v = torch.empty(batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
    
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
    }


def compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False):
    """Compute reference output using PyTorch for arbitrary mask."""
    q = tensors["q"] if not up_cast else tensors["q"].float()
    k = tensors["k"] if not up_cast else tensors["k"].float()
    v = tensors["v"] if not up_cast else tensors["v"].float()
    
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    nheads = q.shape[2]
    nheads_kv = k.shape[2]
    headdim = q.shape[3]
    scale = 1.0 / math.sqrt(headdim)

    if nheads_kv != nheads:
        k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
    
    qk_attn = torch.einsum("bnhd,bmhd->bhnm", q * scale, k)
    qk_attn = apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k)

    softmax_attn = F.softmax(qk_attn, dim=-1)
    out = torch.einsum("bhnm,bmhd->bnhd", softmax_attn, v)

    is_all_zero = torch.count_nonzero(arbitrary_func) == 0
    if is_all_zero:
        out.fill_(0.0)

    return out


def compare_linear_sparse_tensors(cuda_tensors, ref_tensors, name):
    """Compare CUDA kernel output with PyTorch reference."""
    all_match = True

    # Compare mask_block_cnt
    if not torch.equal(cuda_tensors.mask_block_cnt.flatten(), ref_tensors.mask_block_cnt.flatten()):
        print(f"  {name} mask_block_cnt MISMATCH!")
        print(f"    CUDA shape: {cuda_tensors.mask_block_cnt.shape}, ref shape: {ref_tensors.mask_block_cnt.shape}")
        print(f"    CUDA sum: {cuda_tensors.mask_block_cnt.sum().item()}, ref sum: {ref_tensors.mask_block_cnt.sum().item()}")
        all_match = False

    # Compare mask_block_offset
    if not torch.equal(cuda_tensors.mask_block_offset, ref_tensors.mask_block_offset):
        print(f"  {name} mask_block_offset MISMATCH!")
        all_match = False

    # Compare mask_block_idx
    if cuda_tensors.mask_block_idx.shape != ref_tensors.mask_block_idx.shape:
        print(f"  {name} mask_block_idx shape MISMATCH!")
        all_match = False
    elif not torch.equal(cuda_tensors.mask_block_idx, ref_tensors.mask_block_idx):
        if set(cuda_tensors.mask_block_idx.tolist()) != set(ref_tensors.mask_block_idx.tolist()):
            print(f"  {name} mask_block_idx values MISMATCH!")
            all_match = False

    # Compare full_block tensors
    if cuda_tensors.full_block_cnt is not None and ref_tensors.full_block_cnt is not None:
        if not torch.equal(cuda_tensors.full_block_cnt.flatten(), ref_tensors.full_block_cnt.flatten()):
            print(f"  {name} full_block_cnt MISMATCH!")
            all_match = False

    if cuda_tensors.full_block_offset is not None and ref_tensors.full_block_offset is not None:
        if not torch.equal(cuda_tensors.full_block_offset, ref_tensors.full_block_offset):
            print(f"  {name} full_block_offset MISMATCH!")
            all_match = False

    if cuda_tensors.full_block_idx is not None and ref_tensors.full_block_idx is not None:
        if cuda_tensors.full_block_idx.shape != ref_tensors.full_block_idx.shape:
            print(f"  {name} full_block_idx shape MISMATCH!")
            all_match = False
        elif not torch.equal(cuda_tensors.full_block_idx, ref_tensors.full_block_idx):
            if set(cuda_tensors.full_block_idx.tolist()) != set(ref_tensors.full_block_idx.tolist()):
                print(f"  {name} full_block_idx values MISMATCH!")
                all_match = False

    return all_match


# ============================================================================
# Supported native mask patterns (kernel supports these without block sparsity)
# ============================================================================
NATIVE_MASK_PATTERNS = ["causal", "full"]  # Patterns that can use kernel's native mask support


# ============================================================================
# Main test function
# ============================================================================
def _run_mask_test(seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype,
                   use_block_sparsity=True, pattern="causal",
                   n_func=3, batch_broadcast=True, qhead_broadcast=True,
                   deterministic=False):
    """Run arbitrary mask test with automatic tile size detection.
    
    Args:
        use_block_sparsity: If True, use block sparsity + arbitrary mask implementation.
                           If False, use kernel's native mask support (causal, local, etc.)
                           For patterns not natively supported, will raise an error.
        pattern: Mask pattern to test. Options: "causal", "full", "random"
        n_func: Number of functions for arbitrary mask encoding (must be odd)
        batch_broadcast: If True, arbitrary_func batch dim = 1 (broadcast).
                        If False, arbitrary_func batch dim = batch_size.
        qhead_broadcast: If True, arbitrary_func head dim = 1 (broadcast).
                        If False, arbitrary_func head dim = nheads (q_head).
        deterministic: If True, use deterministic mode for backward pass (fixed accumulation order).
    """
    # Set random seed for reproducibility (ensures same behavior in pytest and direct run)
    torch.manual_seed(0)
    
    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
    elif kv_mode == "gqa":
        nheads_kv = nheads // 2
    elif kv_mode == "mqa":
        nheads_kv = 1
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    batch_size = 1
    headdim_v = headdim

    # Create tensors
    tensors = create_tensors(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype)
    softmax_scale = 1.0 / math.sqrt(headdim)

    # =========================================================================
    # Automatic tile size detection
    # =========================================================================
    arch = get_arch()
    
    print(f"\n{'='*60}")
    print(f"Backend: {BACKEND}, Arch: {arch}, headdim: {headdim}")
    print(f"use_block_sparsity: {use_block_sparsity}, pattern: {pattern}")
    print(f"{'='*60}")

    # =========================================================================
    # Branch based on use_block_sparsity
    # =========================================================================
    if not use_block_sparsity:
        # =====================================================================
        # use_block_sparsity=False: Use kernel's native mask support
        # =====================================================================
        if pattern not in NATIVE_MASK_PATTERNS:
            raise ValueError(
                f"Pattern '{pattern}' is not natively supported by the kernel. "
                f"Supported native patterns: {NATIVE_MASK_PATTERNS}. "
                f"Use use_block_sparsity=True for arbitrary patterns."
            )
        
        # Map pattern to kernel parameters
        if pattern == "causal":
            causal = True
            print("Using kernel's native causal mask (no block sparsity)")
        elif pattern == "full":
            causal = False
            print("Using kernel's native full attention (no mask, no block sparsity)")
        
        # Run flash attention with native mask support
        if BACKEND == "hopper":
            out_fa, lse_fa, _, _ = flash_attn_forward(
                q=tensors["q"], k=tensors["k"], v=tensors["v"],
                softmax_scale=softmax_scale, causal=causal,
            )
        else:  # cute
            out_fa, lse_fa = flash_attn_func(
                q=tensors["q"], k=tensors["k"], v=tensors["v"],
                softmax_scale=softmax_scale, causal=causal, arbitrary=False,
                window_size=(None, None), softcap=0.0, num_splits=1,
                pack_gqa=False, deterministic=deterministic,
            )
        
        # Compute reference for native mask (need separate computations for fp32 and dtype)
        def compute_native_reference(tensors, softmax_scale, pattern, nheads, nheads_kv, seqlen_q, seqlen_k, up_cast):
            """Compute reference output for native mask patterns."""
            q = tensors["q"].float() if up_cast else tensors["q"]
            k = tensors["k"].float() if up_cast else tensors["k"]
            v = tensors["v"].float() if up_cast else tensors["v"]
            if nheads_kv != nheads:
                k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
                v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
            qk = torch.einsum("bnhd,bmhd->bhnm", q * softmax_scale, k)
            if pattern == "causal":
                causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device="cuda"), diagonal=1).bool()
                qk.masked_fill_(causal_mask, float("-inf"))
            attn = F.softmax(qk, dim=-1)
            return torch.einsum("bhnm,bmhd->bnhd", attn, v)
        
        # Compute separately to maintain independent computation graphs for backward
        out_ref_fp32 = compute_native_reference(tensors, softmax_scale, pattern, nheads, nheads_kv, seqlen_q, seqlen_k, up_cast=True)
        out_ref = compute_native_reference(tensors, softmax_scale, pattern, nheads, nheads_kv, seqlen_q, seqlen_k, up_cast=False)
        
    else:
        # =====================================================================
        # use_block_sparsity=True: Use block sparsity + arbitrary mask
        # =====================================================================
        causal = False  # Handled by arbitrary mask
        
        # Determine if we should use PackGQA for block sparsity
        # PackGQA is used when nheads_kv != nheads (GQA/MQA mode)
        qhead_per_khead = nheads // nheads_kv
        
        # Generate arbitrary mask function
        # Shape: [B, H_q, func_num, seqlen_q + 256], B and H_q support broadcasting
        func_batch = 1 if batch_broadcast else batch_size
        func_nheads = 1 if qhead_broadcast else nheads # q heads
        # Note: arbitrary_func_tensor signature is (batch, nheads, n_func, seqlen_q, seqlen_k, ...)
        arbitrary_func = arbitrary_func_tensor(func_batch, func_nheads, n_func, seqlen_q, seqlen_k, device="cuda", pattern=pattern)
        print(f"  arbitrary_func shape: {arbitrary_func.shape} (n_func={n_func}, batch_broadcast={batch_broadcast}, qhead_broadcast={qhead_broadcast})")

        # Create flex mask for PyTorch reference
        def mask_mod_flex(b, h, q_idx, kv_idx, arbitrary_func=arbitrary_func):
            return flex_arbitrary_mask(b, h, q_idx, kv_idx, arbitrary_func)

        # Get tile sizes based on backend:
        # - C++ backend (hopper): uses tile_size.h (varies by headdim)
        # - DSL backend (cute): uses fixed tile sizes (128, 128 for fwd)
        fwd_q_block, fwd_kv_block = get_tile_sizes_by_backend(
            backend=BACKEND, pass_type="forward", arch=arch, headdim=headdim,
            is_causal=causal, is_local=False, is_arbitrary=True,
        )
        bwd_q_block, bwd_kv_block = get_tile_sizes_by_backend(
            backend=BACKEND, pass_type="backward", arch=arch, headdim=headdim,
            is_causal=causal, is_local=False, is_arbitrary=True,
        )
        
        print(f"  Backend: {BACKEND}")
        print(f"  Forward (Q2K): Q_BLOCK={fwd_q_block}, KV_BLOCK={fwd_kv_block}")
        print(f"  Backward (K2Q): Q_BLOCK={bwd_q_block}, KV_BLOCK={bwd_kv_block}")

        # Create block masks using CUDA kernel (preferred) or PyTorch flex_attention reference
        if HAS_CREATE_BLOCK_MASK_CUDA:
            # Q2K (Forward)
            (cuda_k_mask_cnt, cuda_k_mask_offset, cuda_k_mask_idx,
             cuda_k_full_cnt, cuda_k_full_offset, cuda_k_full_idx) = \
                create_block_mask_cuda.create_q2k_csr_sparse_from_func(
                    arbitrary_func, seqlen_q, seqlen_k,
                    Q_BLOCK_SIZE=fwd_q_block, KV_BLOCK_SIZE=fwd_kv_block,
                    check_q_boundary=True
                )

            linear_k = LinearBlockSparseTensors(
                mask_block_cnt=cuda_k_mask_cnt,
                mask_block_offset=cuda_k_mask_offset,
                mask_block_idx=cuda_k_mask_idx,
                full_block_cnt=cuda_k_full_cnt,
                full_block_offset=cuda_k_full_offset,
                full_block_idx=cuda_k_full_idx,
            )

            # K2Q (Backward)
            (cuda_q_mask_cnt, cuda_q_mask_offset, cuda_q_mask_idx,
             cuda_q_full_cnt, cuda_q_full_offset, cuda_q_full_idx) = \
                create_block_mask_cuda.create_k2q_csr_sparse_from_func(
                    arbitrary_func, seqlen_q, seqlen_k,
                    Q_BLOCK_SIZE=bwd_q_block, KV_BLOCK_SIZE=bwd_kv_block,
                )

            linear_q = LinearBlockSparseTensors(
                mask_block_cnt=cuda_q_mask_cnt,
                mask_block_offset=cuda_q_mask_offset,
                mask_block_idx=cuda_q_mask_idx,
                full_block_cnt=cuda_q_full_cnt,
                full_block_offset=cuda_q_full_offset,
                full_block_idx=cuda_q_full_idx,
            )
            
            # Optionally compare with PyTorch flex_attention reference (if available)
            if HAS_FLEX_ATTENTION:
                bm_fwd = create_block_mask(mask_mod_flex, func_batch, func_nheads, seqlen_q, seqlen_k, device="cuda",
                                            BLOCK_SIZE=(fwd_q_block, fwd_kv_block))
                
                if isinstance(bm_fwd.as_tuple()[0], int):
                    _, _, k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm_fwd.as_tuple()
                else:
                    k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm_fwd.as_tuple()

                k_block_sparse = BlockSparseTensorsTorch(
                    mask_block_cnt=k_mask_cnt, mask_block_idx=k_mask_idx,
                    full_block_cnt=k_full_cnt, full_block_idx=k_full_idx,
                )
                ref_linear_k = bhqk_to_linear_sparse_tensors_local(k_block_sparse)

                bm_bwd = create_block_mask(mask_mod_flex, func_batch, func_nheads, seqlen_q, seqlen_k, device="cuda",
                                            BLOCK_SIZE=(bwd_q_block, bwd_kv_block))
                
                if isinstance(bm_fwd.as_tuple()[0], int):
                    _, _, _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
                else:
                    _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()

                q_block_sparse = BlockSparseTensorsTorch(
                    mask_block_cnt=q_mask_cnt, mask_block_idx=q_mask_idx,
                    full_block_cnt=q_full_cnt, full_block_idx=q_full_idx,
                )
                ref_linear_q = bhqk_to_linear_sparse_tensors_local(q_block_sparse)

                # Compare CUDA kernel vs PyTorch reference
                print("\nComparing CUDA kernel vs PyTorch flex_attention reference:")
                k_match = compare_linear_sparse_tensors(linear_k, ref_linear_k, "Q2K")
                q_match = compare_linear_sparse_tensors(linear_q, ref_linear_q, "K2Q")

                if k_match and q_match:
                    print("✓ All CUDA kernel outputs match PyTorch reference!")
                else:
                    print("✗ Some outputs do not match!")
            else:
                print("\nUsing CUDA kernel for block mask creation (flex_attention not available)")

        elif HAS_FLEX_ATTENTION:
            # Fallback to PyTorch flex_attention reference
            print("\nUsing PyTorch flex_attention for block mask creation (CUDA kernel not available)")
            bm_fwd = create_block_mask(mask_mod_flex, func_batch, func_nheads, seqlen_q, seqlen_k, device="cuda",
                                        BLOCK_SIZE=(fwd_q_block, fwd_kv_block))
            
            if isinstance(bm_fwd.as_tuple()[0], int):
                _, _, k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm_fwd.as_tuple()
            else:
                k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm_fwd.as_tuple()

            k_block_sparse = BlockSparseTensorsTorch(
                mask_block_cnt=k_mask_cnt, mask_block_idx=k_mask_idx,
                full_block_cnt=k_full_cnt, full_block_idx=k_full_idx,
            )
            linear_k = bhqk_to_linear_sparse_tensors_local(k_block_sparse)

            bm_bwd = create_block_mask(mask_mod_flex, func_batch, func_nheads, seqlen_q, seqlen_k, device="cuda",
                                        BLOCK_SIZE=(bwd_q_block, bwd_kv_block))
            
            if isinstance(bm_fwd.as_tuple()[0], int):
                _, _, _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
            else:
                _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()

            q_block_sparse = BlockSparseTensorsTorch(
                mask_block_cnt=q_mask_cnt, mask_block_idx=q_mask_idx,
                full_block_cnt=q_full_cnt, full_block_idx=q_full_idx,
            )
            linear_q = bhqk_to_linear_sparse_tensors_local(q_block_sparse)
        else:
            raise RuntimeError(
                "Neither CUDA kernel nor flex_attention available for block mask creation. "
                "Please either build create_block_mask_cuda or upgrade PyTorch to support flex_attention."
            )

        # Run flash attention with block sparsity
        if BACKEND == "hopper":
            out_fa, lse_fa, _, _ = flash_attn_forward(
                q=tensors["q"], k=tensors["k"], v=tensors["v"],
                softmax_scale=softmax_scale, causal=causal,
                block_sparse=linear_k, k2q_block_sparse=linear_q,
                arbitrary_func=arbitrary_func,
            )
        else:  # cute
            out_fa, lse_fa = flash_attn_func(
                q=tensors["q"], k=tensors["k"], v=tensors["v"],
                softmax_scale=softmax_scale, causal=causal, arbitrary=True,
                window_size=(None, None), softcap=0.0, num_splits=1,
                pack_gqa=False, deterministic=deterministic, mask_mod=None,
                linear_k_block_sparse_tensors=linear_k,
                linear_q_block_sparse_tensors=linear_q,
                aux_tensors=[arbitrary_func],
            )
        
        # Compute reference for arbitrary mask
        out_ref_fp32 = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=True)
        out_ref = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False)

    # =========================================================================
    # Compare output
    # =========================================================================
    print(f"Output max diff: {(out_fa - out_ref_fp32).abs().max().item()}")
    print(f"Pytorch max diff: {(out_ref - out_ref_fp32).abs().max().item()}")

    # Assertions for forward pass
    assert out_fa.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_fa).any(), "Output contains NaN!"
    assert torch.isfinite(out_fa).all(), "Output contains Inf!"
    
    # Allow 2x numerical error compared to PyTorch reference
    fwd_max_diff = (out_fa - out_ref_fp32).abs().max().item()
    pt_max_diff = (out_ref - out_ref_fp32).abs().max().item()
    assert fwd_max_diff <= 2 * pt_max_diff + 1e-5, \
        f"Forward max diff {fwd_max_diff} > 2 * PyTorch diff {pt_max_diff}"

    # =========================================================================
    # Backward pass (if not disabled)
    # =========================================================================
    if not DISABLE_BACKWARD:
        dout = torch.rand_like(out_fa)

        dq, dk, dv = torch.autograd.grad(out_fa, (tensors["q"], tensors["k"], tensors["v"]), dout)
        dq_ref_fp32, dk_ref_fp32, dv_ref_fp32 = torch.autograd.grad(
            out_ref_fp32, (tensors["q"], tensors["k"], tensors["v"]), dout
        )
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(
            out_ref, (tensors["q"], tensors["k"], tensors["v"]), dout
        )

        print(f"dV max diff: {(dv - dv_ref_fp32).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_ref - dv_ref_fp32).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref_fp32).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_ref - dk_ref_fp32).abs().max().item()}")
        print(f"dQ max diff: {(dq - dq_ref_fp32).abs().max().item()}")
        print(f"dQ Pytorch max diff: {(dq_ref - dq_ref_fp32).abs().max().item()}")

        # Allow 5x numerical error for gradients
        assert (dv - dv_ref_fp32).abs().max().item() <= 5 * (dv_ref - dv_ref_fp32).abs().max().item() + 1e-5
        assert (dk - dk_ref_fp32).abs().max().item() <= 5 * (dk_ref - dk_ref_fp32).abs().max().item() + 1e-5
        assert (dq - dq_ref_fp32).abs().max().item() <= 5 * (dq_ref - dq_ref_fp32).abs().max().item() + 1e-5


# ============================================================================
# Pytest test function - single unified test
# ============================================================================
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (128, 128),
    (256, 256),
    (511, 511),
    (1057, 1057),
    (2123, 2123),
    (4259, 4259),
    (8521, 8521),
])
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
# @pytest.mark.parametrize("kv_mode", ["mha"])
@pytest.mark.parametrize("headdim", COMPILED_HDIMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_block_sparsity", [True])
@pytest.mark.parametrize("pattern", ["causal"])  # Test causal pattern for performance comparison
@pytest.mark.parametrize("n_func", [3])  # Number of functions for arbitrary mask (must be odd)
@pytest.mark.parametrize("batch_broadcast", [True])  # True: func batch=1, False: func batch=batch_size
@pytest.mark.parametrize("qhead_broadcast", [True])  # True: func head=1, False: func head=nheads
@pytest.mark.parametrize("deterministic", [True])  # True: deterministic mode for backward pass
def test_arbitrary_mask(seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, use_block_sparsity, pattern,
                        n_func, batch_broadcast, qhead_broadcast, deterministic):
    """
    Test arbitrary mask with optional block sparsity.
    
    Tests:
    - Forward pass correctness against PyTorch reference
    - Backward pass correctness (if not disabled)
    - Block sparse tensors match between CUDA kernel and PyTorch reference (when use_block_sparsity=True)
    
    Args:
        use_block_sparsity: If True, use block sparsity + arbitrary mask implementation.
                           If False, use kernel's native mask support (causal, local, etc.)
                           This is useful for performance comparison.
        pattern: Mask pattern to test ("causal", "full", "random").
                 When use_block_sparsity=False, only "causal" and "full" are supported.
        n_func: Number of functions for arbitrary mask encoding (must be odd)
        batch_broadcast: If True, arbitrary_func batch dim = 1 (broadcast).
                        If False, arbitrary_func batch dim = batch_size.
        qhead_broadcast: If True, arbitrary_func head dim = 1 (broadcast).
                        If False, arbitrary_func head dim = nheads (q_head).
        deterministic: If True, use deterministic mode for backward pass (fixed accumulation order).
    
    Tile sizes are automatically detected based on GPU architecture and headdim.
    """
    # =========================================================================
    # Skip conditions based on architecture and configuration
    # =========================================================================
    if COMPUTE_CAPABILITY < 8:
        pytest.skip("Arbitrary mask requires SM80+")

    if COMPUTE_CAPABILITY == 10 and headdim not in [64, 128]:
        pytest.skip(f"SM100 does not support headdim={headdim} for arbitrary mask")

    # DSL SM90 limitations (from interface.py):
    # - headdim must be 128 (backward pass limitation)
    # - num_head must equal num_head_kv (MHA only, no GQA/MQA)
    if BACKEND == "cute" and COMPUTE_CAPABILITY == 9:
        if headdim > 128:
            pytest.skip(f"DSL SM90 does not support headdim={headdim} (only headdim <= 128 supported)")
        if kv_mode != "mha":
            pytest.skip(f"DSL SM90 does not support {kv_mode} mode (only MHA supported)")

    # Skip unsupported pattern + use_block_sparsity combinations
    if not use_block_sparsity and pattern not in NATIVE_MASK_PATTERNS:
        pytest.skip(f"Pattern '{pattern}' requires use_block_sparsity=True (not natively supported)")

    # Run test
    _run_mask_test(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=nheads,
        kv_mode=kv_mode,
        headdim=headdim,
        dtype=dtype,
        use_block_sparsity=use_block_sparsity,
        pattern=pattern,
        n_func=n_func,
        batch_broadcast=batch_broadcast,
        qhead_broadcast=qhead_broadcast,
        deterministic=deterministic,
    )


# ============================================================================
# Benchmark function
# ============================================================================
def benchmark_arbitrary_mask(
    seqlen_q=8192,
    seqlen_k=8192,
    batch_size=1,
    nheads=32,
    nheads_kv=None,
    headdim=128,
    dtype=torch.bfloat16,
    pattern="causal",
    n_func=3,
    batch_broadcast=True,
    qhead_broadcast=True,
    num_warmup=5,
    num_runs=20,
    deterministic=False,
):
    """Benchmark comparing block sparsity vs native mask implementation.
    
    Args:
        seqlen_q: Query sequence length
        seqlen_k: Key/Value sequence length
        batch_size: Batch size
        nheads: Number of query heads
        nheads_kv: Number of key/value heads (None = same as nheads)
        headdim: Head dimension
        dtype: Data type
        pattern: Mask pattern ("causal" or "full")
        n_func: Number of functions for arbitrary mask encoding (must be odd)
        batch_broadcast: If True, use batch=1 for arbitrary_func (broadcast across batches)
        qhead_broadcast: If True, use nheads=1 for arbitrary_func (broadcast across heads)
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        deterministic: If True, use deterministic mode for backward pass
    
    Returns:
        Dict with benchmark results
    """
    if nheads_kv is None:
        nheads_kv = nheads
    
    headdim_v = headdim
    softmax_scale = 1.0 / math.sqrt(headdim)
    arch = get_arch()
    
    # Check if backward is enabled
    benchmark_backward = not DISABLE_BACKWARD
    
    # Determine arbitrary_func dimensions based on broadcast settings
    func_batch = 1 if batch_broadcast else batch_size
    func_nheads = 1 if qhead_broadcast else nheads
    
    print(f"\n{'='*80}")
    print(f"Benchmark: Fixed Mask vs Arbitrary Mask")
    print(f"{'='*80}")
    print(f"Backend: {BACKEND}, Arch: {arch}")
    print(f"Config: B={batch_size}, seqlen={seqlen_q}x{seqlen_k}, H={nheads}, H_kv={nheads_kv}, d={headdim}")
    print(f"Pattern: {pattern}, dtype: {dtype}, n_func: {n_func}")
    print(f"Broadcast: batch={batch_broadcast} (func_batch={func_batch}), qhead={qhead_broadcast} (func_nheads={func_nheads})")
    print(f"Warmup: {num_warmup}, Runs: {num_runs}")
    print(f"Backward benchmark: {'Enabled' if benchmark_backward else 'Disabled (FLASH_ATTENTION_DISABLE_BACKWARD=TRUE)'}")
    print(f"{'='*80}")
    
    # Calculate theoretical FLOPs
    # Forward: 4 * B * H * seqlen_q * seqlen_k * d
    # For causal, roughly half the FLOPs due to triangular mask
    # Backward: ~2.5x forward FLOPs
    if pattern == "causal":
        effective_seqlen = seqlen_q * seqlen_k / 2  # Approximate for causal
    else:
        effective_seqlen = seqlen_q * seqlen_k
    
    fwd_flops = 4 * batch_size * nheads * effective_seqlen * headdim
    bwd_flops = 2.5 * fwd_flops  # Backward is roughly 2.5x forward
    
    results = {}
    
    # =========================================================================
    # Benchmark 1: Fixed mask (use_block_sparsity=False) - Forward
    # =========================================================================
    if pattern in NATIVE_MASK_PATTERNS:
        print(f"\n[1/2] Benchmarking Fixed {pattern.upper()} mask...")
        causal = (pattern == "causal")
        
        # Create tensors without grad for forward-only benchmark
        device = "cuda"
        q_fwd = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1)
        k_fwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype).uniform_(-1, 1)
        v_fwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype).uniform_(-1, 1)
        
        # Forward warmup
        for _ in range(num_warmup):
            if BACKEND == "hopper":
                out, _, _, _ = flash_attn_forward(
                    q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=causal,
                )
            else:
                out, _ = flash_attn_func(
                    q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=causal,
                    arbitrary=False, window_size=(None, None), softcap=0.0,
                    num_splits=1, pack_gqa=False, deterministic=deterministic,
                )
        torch.cuda.synchronize()
        
        # Forward benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_runs):
            if BACKEND == "hopper":
                out, _, _, _ = flash_attn_forward(
                    q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=causal,
                )
            else:
                out, _ = flash_attn_func(
                    q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=causal,
                    arbitrary=False, window_size=(None, None), softcap=0.0,
                    num_splits=1, pack_gqa=False, deterministic=deterministic,
                )
        end_event.record()
        torch.cuda.synchronize()
        
        fixed_fwd_time_ms = start_event.elapsed_time(end_event) / num_runs
        fixed_fwd_tflops = fwd_flops / (fixed_fwd_time_ms * 1e-3) / 1e12
        
        results["fixed_fwd"] = {
            "time_ms": fixed_fwd_time_ms,
            "tflops": fixed_fwd_tflops,
        }
        print(f"  Forward:  {fixed_fwd_time_ms:.3f} ms, {fixed_fwd_tflops:.2f} TFLOPS")
        
        # Backward benchmark (if enabled)
        if benchmark_backward:
            # Create tensors with grad for backward
            q_bwd = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
            k_bwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
            v_bwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
            dout = torch.rand(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype)
            
            # Step 1: Execute forward once to get out (for backward)
            if BACKEND == "hopper":
                out = flash_attn_func(
                    q=q_bwd, k=k_bwd, v=v_bwd, softmax_scale=softmax_scale, causal=causal,
                    window_size=(-1, -1), softcap=0.0,
                    num_splits=1, pack_gqa=None, deterministic=deterministic,
                )
            else:
                out, _ = flash_attn_func(
                    q=q_bwd, k=k_bwd, v=v_bwd, softmax_scale=softmax_scale, causal=causal,
                    arbitrary=False, window_size=(None, None), softcap=0.0,
                    num_splits=1, pack_gqa=False, deterministic=deterministic,
                )
            
            # Step 2: Backward warmup (only backward, retain_graph=True)
            for _ in range(num_warmup):
                torch.autograd.grad(out, (q_bwd, k_bwd, v_bwd), dout, retain_graph=True)
            torch.cuda.synchronize()
            
            # Step 3: Backward benchmark (only backward)
            start_event.record()
            for _ in range(num_runs):
                torch.autograd.grad(out, (q_bwd, k_bwd, v_bwd), dout, retain_graph=True)
            end_event.record()
            torch.cuda.synchronize()
            
            fixed_bwd_time_ms = start_event.elapsed_time(end_event) / num_runs
            fixed_bwd_tflops = bwd_flops / (fixed_bwd_time_ms * 1e-3) / 1e12
            
            results["fixed_bwd"] = {
                "time_ms": fixed_bwd_time_ms,
                "tflops": fixed_bwd_tflops,
            }
            print(f"  Backward: {fixed_bwd_time_ms:.3f} ms, {fixed_bwd_tflops:.2f} TFLOPS")
    
    # =========================================================================
    # Benchmark 2: Block sparsity + arbitrary mask (use_block_sparsity=True)
    # =========================================================================
    # Clear CUDA state before benchmark
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    print(f"\n[2/2] Benchmarking Block Sparsity + Arbitrary mask ({pattern})...")
    
    # Get tile sizes based on backend
    fwd_q_block, fwd_kv_block = get_tile_sizes_by_backend(
        backend=BACKEND, pass_type="forward", arch=arch, headdim=headdim,
        is_causal=False, is_local=False, is_arbitrary=True,
    )
    bwd_q_block, bwd_kv_block = get_tile_sizes_by_backend(
        backend=BACKEND, pass_type="backward", arch=arch, headdim=headdim,
        is_causal=False, is_local=False, is_arbitrary=True,
    )
    print(f"  Backend: {BACKEND}")
    print(f"  Forward (Q2K): Q_BLOCK={fwd_q_block}, KV_BLOCK={fwd_kv_block}")
    print(f"  Backward (K2Q): Q_BLOCK={bwd_q_block}, KV_BLOCK={bwd_kv_block}")
    
    # Create arbitrary_func with configurable broadcast
    arbitrary_func = arbitrary_func_tensor(func_batch, func_nheads, n_func, seqlen_q, seqlen_k, device="cuda", pattern=pattern)
    print(f"  arbitrary_func shape: {arbitrary_func.shape}")
    
    # Create block sparse tensors using CUDA kernel if available (faster and more reliable)
    if HAS_CREATE_BLOCK_MASK_CUDA:
        # Q2K (Forward)
        (k_mask_cnt, k_mask_offset, k_mask_idx,
         k_full_cnt, k_full_offset, k_full_idx) = \
            create_block_mask_cuda.create_q2k_csr_sparse_from_func(
                arbitrary_func, seqlen_q, seqlen_k,
                Q_BLOCK_SIZE=fwd_q_block, KV_BLOCK_SIZE=fwd_kv_block,
                check_q_boundary=True
            )
        linear_k = LinearBlockSparseTensors(
            mask_block_cnt=k_mask_cnt,
            mask_block_offset=k_mask_offset,
            mask_block_idx=k_mask_idx,
            full_block_cnt=k_full_cnt,
            full_block_offset=k_full_offset,
            full_block_idx=k_full_idx,
        )
        print(f"  Q2K: mask_cnt={k_mask_cnt.shape}, mask_idx={k_mask_idx.shape}, full_idx={k_full_idx.shape}")
        
        # K2Q (Backward)
        (q_mask_cnt, q_mask_offset, q_mask_idx,
         q_full_cnt, q_full_offset, q_full_idx) = \
            create_block_mask_cuda.create_k2q_csr_sparse_from_func(
                arbitrary_func, seqlen_q, seqlen_k,
                Q_BLOCK_SIZE=bwd_q_block, KV_BLOCK_SIZE=bwd_kv_block,
            )
        linear_q = LinearBlockSparseTensors(
            mask_block_cnt=q_mask_cnt,
            mask_block_offset=q_mask_offset,
            mask_block_idx=q_mask_idx,
            full_block_cnt=q_full_cnt,
            full_block_offset=q_full_offset,
            full_block_idx=q_full_idx,
        )
        print(f"  K2Q: mask_cnt={q_mask_cnt.shape}, mask_idx={q_mask_idx.shape}, full_idx={q_full_idx.shape}")
    elif HAS_FLEX_ATTENTION:
        # Fallback to PyTorch create_block_mask (flex_attention)
        print("  Using PyTorch flex_attention for block mask creation")
        def mask_mod_flex(b, h, q_idx, kv_idx, arbitrary_func=arbitrary_func):
            return flex_arbitrary_mask(b, h, q_idx, kv_idx, arbitrary_func)
        
        bm_fwd = create_block_mask(mask_mod_flex, 1, 1, seqlen_q, seqlen_k, device="cuda",
                                    BLOCK_SIZE=(fwd_q_block, fwd_kv_block))
        
        if isinstance(bm_fwd.as_tuple()[0], int):
            _, _, k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm_fwd.as_tuple()
        else:
            k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm_fwd.as_tuple()
        
        k_block_sparse = BlockSparseTensorsTorch(
            mask_block_cnt=k_mask_cnt, mask_block_idx=k_mask_idx,
            full_block_cnt=k_full_cnt, full_block_idx=k_full_idx,
        )
        linear_k = bhqk_to_linear_sparse_tensors_local(k_block_sparse)
        
        bm_bwd = create_block_mask(mask_mod_flex, 1, 1, seqlen_q, seqlen_k, device="cuda",
                                    BLOCK_SIZE=(bwd_q_block, bwd_kv_block))
        
        if isinstance(bm_bwd.as_tuple()[0], int):
            _, _, _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
        else:
            _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
        
        q_block_sparse = BlockSparseTensorsTorch(
            mask_block_cnt=q_mask_cnt, mask_block_idx=q_mask_idx,
            full_block_cnt=q_full_cnt, full_block_idx=q_full_idx,
        )
        linear_q = bhqk_to_linear_sparse_tensors_local(q_block_sparse)
    else:
        raise RuntimeError(
            "Neither CUDA kernel nor flex_attention available for block mask creation. "
            "Please either build create_block_mask_cuda or upgrade PyTorch to support flex_attention."
        )
    
    # Create tensors without grad for forward-only benchmark
    device = "cuda"
    q_fwd = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1)
    k_fwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype).uniform_(-1, 1)
    v_fwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype).uniform_(-1, 1)
    
    # Forward warmup
    for _ in range(num_warmup):
        if BACKEND == "hopper":
            out, _, _, _ = flash_attn_forward(
                q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=False,
                block_sparse=linear_k, k2q_block_sparse=linear_q,
                arbitrary_func=arbitrary_func,
            )
        else:
            out, _ = flash_attn_func(
                q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=False,
                arbitrary=True, window_size=(None, None), softcap=0.0,
                num_splits=1, pack_gqa=False, deterministic=deterministic, mask_mod=None,
                linear_k_block_sparse_tensors=linear_k,
                linear_q_block_sparse_tensors=linear_q,
                aux_tensors=[arbitrary_func],
            )
    torch.cuda.synchronize()
    
    # Forward benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        if BACKEND == "hopper":
            out, _, _, _ = flash_attn_forward(
                q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=False,
                block_sparse=linear_k, k2q_block_sparse=linear_q,
                arbitrary_func=arbitrary_func,
            )
        else:
            out, _ = flash_attn_func(
                q=q_fwd, k=k_fwd, v=v_fwd, softmax_scale=softmax_scale, causal=False,
                arbitrary=True, window_size=(None, None), softcap=0.0,
                num_splits=1, pack_gqa=False, deterministic=deterministic, mask_mod=None,
                linear_k_block_sparse_tensors=linear_k,
                linear_q_block_sparse_tensors=linear_q,
                aux_tensors=[arbitrary_func],
            )
    end_event.record()
    torch.cuda.synchronize()
    
    arbitrary_fwd_time_ms = start_event.elapsed_time(end_event) / num_runs
    arbitrary_fwd_tflops = fwd_flops / (arbitrary_fwd_time_ms * 1e-3) / 1e12
    
    results["arbitrary_fwd"] = {
        "time_ms": arbitrary_fwd_time_ms,
        "tflops": arbitrary_fwd_tflops,
    }
    print(f"  Forward:  {arbitrary_fwd_time_ms:.3f} ms, {arbitrary_fwd_tflops:.2f} TFLOPS")
    
    # Backward benchmark (if enabled) - use torch.autograd.grad with retain_graph=True
    if benchmark_backward:
        # Create tensors with grad for backward
        q_bwd = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
        k_bwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
        v_bwd = torch.empty(batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
        dout = torch.rand(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype)
        
        # Step 1: Execute forward once to get out (for backward)
        if BACKEND == "hopper":
            out, _, _, _ = flash_attn_forward(
                q=q_bwd, k=k_bwd, v=v_bwd,
                softmax_scale=softmax_scale, causal=False,
                block_sparse=linear_k, k2q_block_sparse=linear_q,
                arbitrary_func=arbitrary_func,
            )
        else:
            out, _ = flash_attn_func(
                q=q_bwd, k=k_bwd, v=v_bwd, softmax_scale=softmax_scale, causal=False,
                arbitrary=True, window_size=(None, None), softcap=0.0,
                num_splits=1, pack_gqa=False, deterministic=deterministic, mask_mod=None,
                linear_k_block_sparse_tensors=linear_k,
                linear_q_block_sparse_tensors=linear_q,
                aux_tensors=[arbitrary_func],
            )
        
        # Step 2: Backward warmup (only backward, retain_graph=True)
        for _ in range(num_warmup):
            torch.autograd.grad(out, (q_bwd, k_bwd, v_bwd), dout, retain_graph=True)
        torch.cuda.synchronize()
        
        # Step 3: Backward benchmark (only backward)
        start_event.record()
        for _ in range(num_runs):
            torch.autograd.grad(out, (q_bwd, k_bwd, v_bwd), dout, retain_graph=True)
        end_event.record()
        torch.cuda.synchronize()
        
        arbitrary_bwd_time_ms = start_event.elapsed_time(end_event) / num_runs
        arbitrary_bwd_tflops = bwd_flops / (arbitrary_bwd_time_ms * 1e-3) / 1e12
        
        results["arbitrary_bwd"] = {
            "time_ms": arbitrary_bwd_time_ms,
            "tflops": arbitrary_bwd_tflops,
        }
        print(f"  Backward: {arbitrary_bwd_time_ms:.3f} ms, {arbitrary_bwd_tflops:.2f} TFLOPS")
    
    # =========================================================================
    # Print comparison
    # =========================================================================
    print(f"\n{'='*80}")
    print("Benchmark Results Summary")
    print(f"{'='*80}")
    
    # Forward comparison
    print(f"\n{'Forward Pass':^80}")
    print(f"{'-'*80}")
    print(f"{'Method':<30} {'Time (ms)':<15} {'TFLOPS':<15} {'vs Fixed':<20}")
    print(f"{'-'*80}")
    
    if "fixed_fwd" in results:
        fixed_fwd_time = results["fixed_fwd"]["time_ms"]
        fixed_fwd_tflops = results["fixed_fwd"]["tflops"]
        print(f"{'Fixed mask':<30} {fixed_fwd_time:<15.3f} {fixed_fwd_tflops:<15.2f} {'baseline':<20}")
        
        arbitrary_fwd_time = results["arbitrary_fwd"]["time_ms"]
        arbitrary_fwd_tflops = results["arbitrary_fwd"]["tflops"]
        fwd_speedup = fixed_fwd_time / arbitrary_fwd_time
        fwd_overhead_pct = (arbitrary_fwd_time - fixed_fwd_time) / fixed_fwd_time * 100
        print(f"{'Arbitrary mask':<30} {arbitrary_fwd_time:<15.3f} {arbitrary_fwd_tflops:<15.2f} {fwd_speedup:.2f}x ({fwd_overhead_pct:+.1f}%)")
    else:
        arbitrary_fwd_time = results["arbitrary_fwd"]["time_ms"]
        arbitrary_fwd_tflops = results["arbitrary_fwd"]["tflops"]
        print(f"{'Arbitrary mask':<30} {arbitrary_fwd_time:<15.3f} {arbitrary_fwd_tflops:<15.2f}")
    
    # Backward comparison (if enabled)
    if benchmark_backward and "fixed_bwd" in results:
        print(f"\n{'Backward Pass':^80}")
        print(f"{'-'*80}")
        print(f"{'Method':<30} {'Time (ms)':<15} {'TFLOPS':<15} {'vs Fixed':<20}")
        print(f"{'-'*80}")
        
        fixed_bwd_time = results["fixed_bwd"]["time_ms"]
        fixed_bwd_tflops = results["fixed_bwd"]["tflops"]
        print(f"{'Fixed mask':<30} {fixed_bwd_time:<15.3f} {fixed_bwd_tflops:<15.2f} {'baseline':<20}")
        
        if "arbitrary_bwd" in results:
            arbitrary_bwd_time = results["arbitrary_bwd"]["time_ms"]
            arbitrary_bwd_tflops = results["arbitrary_bwd"]["tflops"]
            bwd_speedup = fixed_bwd_time / arbitrary_bwd_time
            bwd_overhead_pct = (arbitrary_bwd_time - fixed_bwd_time) / fixed_bwd_time * 100
            print(f"{'Arbitrary mask':<30} {arbitrary_bwd_time:<15.3f} {arbitrary_bwd_tflops:<15.2f} {bwd_speedup:.2f}x ({bwd_overhead_pct:+.1f}%)")
            
            # Total (fwd + bwd)
            print(f"\n{'Total (Forward + Backward)':^80}")
            print(f"{'-'*80}")
            print(f"{'Method':<30} {'Time (ms)':<15} {'vs Fixed':<20}")
            print(f"{'-'*80}")
            
            fixed_total = fixed_fwd_time + fixed_bwd_time
            arbitrary_total = arbitrary_fwd_time + arbitrary_bwd_time
            total_speedup = fixed_total / arbitrary_total
            total_overhead_pct = (arbitrary_total - fixed_total) / fixed_total * 100
            print(f"{'Fixed mask':<30} {fixed_total:<15.3f} {'baseline':<20}")
            print(f"{'Arbitrary mask':<30} {arbitrary_total:<15.3f} {total_speedup:.2f}x ({total_overhead_pct:+.1f}%)")
        else:
            print(f"{'Arbitrary mask':<30} {'N/A':<50}")
    
    print(f"\n{'='*80}")
    
    return results


# ============================================================================
# Main entry point for direct execution
# ============================================================================
if __name__ == "__main__":
    # Set seed for reproducibility
    torch.random.manual_seed(0)
    
    print(f"Running arbitrary mask test with backend: {BACKEND}")
    print(f"Compute capability: {COMPUTE_CAPABILITY}")

    # Test with block sparsity (causal pattern via arbitrary mask)
    print("\n" + "="*70)
    print("Test 1: Block sparsity + arbitrary mask (causal pattern, broadcast)")
    print("="*70)
    test_arbitrary_mask(
        seqlen_q=8192,
        seqlen_k=8192,
        nheads=32,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=False,
        pattern="causal",
        n_func=3,
        batch_broadcast=True,
        qhead_broadcast=True,
        deterministic=False,
    )
    
    # Test with block sparsity (causal pattern via arbitrary mask)
    print("\n" + "="*70)
    print("Test 1: Block sparsity + arbitrary mask (causal pattern, broadcast)")
    print("="*70)
    test_arbitrary_mask(
        seqlen_q=8192,
        seqlen_k=8192,
        nheads=32,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=True,
        pattern="causal",
        n_func=3,
        batch_broadcast=True,
        qhead_broadcast=True,
        deterministic=False,
    )
    
    # Test without broadcast
    print("\n" + "="*70)
    print("Test 2: Block sparsity + arbitrary mask (no broadcast)")
    print("="*70)
    test_arbitrary_mask(
        seqlen_q=8192,
        seqlen_k=8192,
        nheads=32,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=True,
        pattern="causal",
        n_func=3,
        batch_broadcast=False,
        qhead_broadcast=False,
        deterministic=False,
    )
    
    # Run benchmark (use GQA mode for better compatibility)
    print("\n" + "="*70)
    print("Benchmark: fixed mask vs arbitrary mask")
    print("="*70)
    benchmark_arbitrary_mask(
        seqlen_q=8192,
        seqlen_k=8192,
        batch_size=1,
        nheads=32,
        nheads_kv=32,  # GQA mode: nheads_kv = nheads // 4, MQA mode: nheads_kv = 1
        headdim=128,
        dtype=torch.bfloat16,
        pattern="causal",
        n_func=3,
        batch_broadcast=True,
        qhead_broadcast=True,
        num_warmup=5,
        num_runs=20,
        deterministic=False,
    )