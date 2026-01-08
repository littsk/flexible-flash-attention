"""
Test script for create_block_mask CUDA kernels.

Tests two CUDA kernels:
  - create_q2k_block_sparse_from_func: Q2K (Forward)  - fix q_block, loop kv_blocks
  - create_k2q_block_sparse_from_func: K2Q (Backward) - fix kv_block, loop q_blocks

Compares the output of our CUDA kernels with PyTorch reference implementations.

Usage:
    # Build with Makefile (from utils directory)
    make block_mask
    
    # Or build directly
    cd create_block_mask && pip install -e . --no-build-isolation
    
    # Run tests
    python test_create_block_mask.py
"""
import torch
import pytest
from torch.nn.attention.flex_attention import create_block_mask

# Import our CUDA kernel
try:
    import create_block_mask_cuda
except ImportError:
    create_block_mask_cuda = None
    print("Warning: create_block_mask_cuda not found. Please build it first with: pip install -e .")

def flex_arbitrary_mask(b, h, q_idx, kv_idx, arbitrary_func):
    """
    Reference flex attention mask function.
    
    arbitrary_func: [B, H, n_func, seqlen_q + 256]
    Valid intervals are [0, F0), [F1, F2), [F3, F4), ...
    """
    zero = h * 0
    value_valid = kv_idx < arbitrary_func[b, zero, zero, q_idx]
    n_func = arbitrary_func.shape[2]
    for i in range(n_func // 2):
        in_range = (kv_idx >= arbitrary_func[b, zero, zero + (2*i+1), q_idx]) & (kv_idx < arbitrary_func[b, zero, zero + (2*i+2), q_idx])
        value_valid = value_valid | in_range
    return value_valid


def create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Create reference block mask using PyTorch's create_block_mask.
    
    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension
    
    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx
    """
    frozen_af = func_tensor
    
    def mask_mod(b, h, q_idx, kv_idx, af=frozen_af):
        return flex_arbitrary_mask(b, h, q_idx, kv_idx, af)
    
    bm = create_block_mask(
        mask_mod,
        1,  # B
        1,  # H
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
    )
    
    # Extract tensors from BlockMask
    # Format: (Q_LEN, KV_LEN, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, 
    #          q_num_blocks, q_indices, full_q_num_blocks, full_q_indices, Q_BLOCK_SIZE, KV_BLOCK_SIZE, mask_mod)
    bm_tuple = bm.as_tuple()
    mask_block_cnt = bm_tuple[2]   # kv_num_blocks
    mask_block_idx = bm_tuple[3]   # kv_indices
    full_block_cnt = bm_tuple[4]   # full_kv_num_blocks
    full_block_idx = bm_tuple[5]   # full_kv_indices
    
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def create_reference_k2q_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Create reference K2Q (backward) block mask using PyTorch's create_block_mask.
    
    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension
    
    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx
        Note: For K2Q, these are q_num_blocks, q_indices, full_q_num_blocks, full_q_indices
    """
    frozen_af = func_tensor
    
    def mask_mod(b, h, q_idx, kv_idx, af=frozen_af):
        return flex_arbitrary_mask(b, h, q_idx, kv_idx, af)
    
    bm = create_block_mask(
        mask_mod,
        1,  # B
        1,  # H
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
    )
    
    # Extract K2Q tensors from BlockMask
    # Format: (Q_LEN, KV_LEN, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, 
    #          q_num_blocks, q_indices, full_q_num_blocks, full_q_indices, Q_BLOCK_SIZE, KV_BLOCK_SIZE, mask_mod)
    bm_tuple = bm.as_tuple()
    mask_block_cnt = bm_tuple[6]   # q_num_blocks
    mask_block_idx = bm_tuple[7]   # q_indices
    full_block_cnt = bm_tuple[8]   # full_q_num_blocks
    full_block_idx = bm_tuple[9]   # full_q_indices
    
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def create_q2k_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=False):
    """
    Create Q2K (forward) block mask using our CUDA kernel.
    
    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension
        check_q_boundary: if True (FlexAttention mode), partial q_blocks cannot have FULL kv_blocks
    
    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx
        Output shapes: [B, H, num_q_blocks, ...] for idx tensors
    """
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = \
        create_block_mask_cuda.create_q2k_block_sparse_from_func(
            func_tensor,
            seqlen_q,
            seqlen_k,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            check_q_boundary,
            debug=True  # Initialize to -1 for testing
        )
    
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def create_k2q_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Create K2Q (backward) block mask using our CUDA kernel.
    
    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension
    
    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx
        Output shapes: [B, H, num_kv_blocks, ...] for idx tensors
    """
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = \
        create_block_mask_cuda.create_k2q_block_sparse_from_func(
            func_tensor,
            seqlen_q,
            seqlen_k,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            debug=True  # Initialize to -1 for testing
        )
    
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def compare_block_masks(ref_cnt, ref_idx, kernel_cnt, kernel_idx, name=""):
    """
    Compare block mask tensors.
    
    The idx tensors may have different order but should contain the same values.
    """
    # Compare counts
    cnt_match = torch.equal(ref_cnt, kernel_cnt)
    if not cnt_match:
        print(f"{name} count mismatch!")
        print(f"  Reference: {ref_cnt}")
        print(f"  Kernel:    {kernel_cnt}")
        return False
    
    # Compare indices (considering they might be in different order)
    B, H, num_blocks = ref_cnt.shape
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                cnt = ref_cnt[b, h, block_idx].item()
                if cnt > 0:
                    ref_indices = set(ref_idx[b, h, block_idx, :cnt].tolist())
                    kernel_indices = set(kernel_idx[b, h, block_idx, :cnt].tolist())
                    if ref_indices != kernel_indices:
                        print(f"{name} index mismatch at (b={b}, h={h}, block={block_idx})!")
                        print(f"  Reference: {sorted(ref_indices)}")
                        print(f"  Kernel:    {sorted(kernel_indices)}")
                        return False
    
    return True

def generate_func_tensor(seqlen_q, seqlen_k, n_func=3, device="cuda"):
    """
    Generate func_tensor with multiple intervals.
    
    Returns:
        func_tensor: [1, 1, n_func, seqlen_q + 256]
    """
    B, H = 1, 1
    func_tensor = torch.zeros(B, H, n_func, seqlen_q + 256, dtype=torch.int32, device=device)

    pattern = "random"
    if pattern == "random":
        # random pattern
        coef = 1.0 / n_func
        for i in range(n_func):
            low = int(i * coef * seqlen_k)
            high = int((i + 1) * coef * seqlen_k)
            if high <= low:
                high = low + 1
            func_tensor[:, :, i, :seqlen_q] = torch.randint(low, high, size=(B, H, seqlen_q), dtype=torch.int32, device=device)
    elif pattern == "causal":
        # causal mask pattern
        for q_idx in range(seqlen_q + 256):
            if q_idx < seqlen_q:
                func_tensor[0, 0, 0, q_idx] = min(q_idx + 1, seqlen_k)
    
    return func_tensor.contiguous()


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (4096, 4096),
    (4111, 4111),
    (8111, 8111),
    (16384, 16384),
    (16666, 16666),
    (32768, 32768),
    (33333, 33333),
])
@pytest.mark.parametrize("n_func", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (128, 128),
    (256, 128),
])
def test_q2k_random_mask(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test Q2K (forward) random arbitrary mask patterns."""
    print(f"\nTesting Q2K random mask: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"n_func={n_func}, Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)
    
    # Reference
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    # Kernel (Q2K)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx = \
        create_q2k_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True)
    
    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")
    
    assert mask_ok, "Mask block mismatch"
    assert full_ok, "Full block mismatch"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (4096, 4096),
    (4111, 4111),
    (8111, 8111),
    (16384, 16384),
    (16666, 16666),
    (32768, 32768),
    (33333, 33333),
])
@pytest.mark.parametrize("n_func", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (128, 128),
    (256, 128),
])
def test_k2q_random_mask(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test K2Q (backward) random arbitrary mask patterns."""
    print(f"\nTesting K2Q random mask: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"n_func={n_func}, Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)
    
    # Reference (K2Q)
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_k2q_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    # Kernel (K2Q)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx = \
        create_k2q_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")
    
    assert mask_ok, "Mask block mismatch"
    assert full_ok, "Full block mismatch"
    print("  PASSED!")


def run_q2k_test(seqlen_q=1024, seqlen_k=1024, n_func=1, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128, check_q_boundary=True):
    """
    Run a single Q2K (forward) test for debugging.
    """
    print(f"Running Q2K test:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  n_func={n_func}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    print(f"  check_q_boundary={check_q_boundary}")
    
    # Generate random func_tensor
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    print(f"func_tensor: {func_tensor}")
    
    print(f"\nFunc tensor shape: {func_tensor.shape}")
    
    if create_block_mask_cuda is None:
        print("\nCUDA kernel not built. Skipping kernel test.")
        print("Please build with: pip install -e .")
        return False
    
    # Kernel
    print("\nComputing Q2K kernel...")
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx = \
        create_q2k_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary)
    
    if check_q_boundary:
        # Compare with PyTorch reference (FlexAttention mode)
        print("\nComputing PyTorch reference (check_q_boundary=True)...")
        ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
            create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        
        # Compare
        print("\nComparing results...")
        mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
        full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")
        
        if mask_ok and full_ok:
            print("\n✓ Q2K test passed!")
        else:
            print("\n✗ Q2K test failed!")
    else:
        mask_ok = True
        full_ok = True
        
    # Print some statistics
    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    total_mask_blocks = kernel_mask_cnt.sum().item()
    total_full_blocks = kernel_full_cnt.sum().item()
    total_blocks = num_q_blocks * num_kv_blocks
    
    print(f"\nQ2K Statistics:")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Mask blocks: {total_mask_blocks} ({100*total_mask_blocks/total_blocks:.1f}%)")
    print(f"  Full blocks: {total_full_blocks} ({100*total_full_blocks/total_blocks:.1f}%)")
    print(f"  Empty blocks: {total_blocks - total_mask_blocks - total_full_blocks}")
    
    return mask_ok and full_ok


def run_k2q_test(seqlen_q=1024, seqlen_k=1024, n_func=1, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128):
    """
    Run a single K2Q (backward) test for debugging.
    """
    print(f"Running K2Q test:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  n_func={n_func}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    
    # Generate random func_tensor
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)
    
    print(f"\nFunc tensor shape: {func_tensor.shape}")
    
    if create_block_mask_cuda is None:
        print("\nCUDA kernel not built. Skipping kernel test.")
        print("Please build with: pip install -e .")
        return False
    
    # Kernel
    print("\nComputing K2Q kernel...")
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx = \
        create_k2q_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    print(f"K2Q mask_block_cnt shape: {kernel_mask_cnt.shape}")
    print(f"K2Q full_block_cnt shape: {kernel_full_cnt.shape}")
    
    # Print some statistics
    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    total_mask_blocks = kernel_mask_cnt.sum().item()
    total_full_blocks = kernel_full_cnt.sum().item()
    total_blocks = num_q_blocks * num_kv_blocks
    
    print(f"\nK2Q Statistics:")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Mask blocks: {total_mask_blocks} ({100*total_mask_blocks/total_blocks:.1f}%)")
    print(f"  Full blocks: {total_full_blocks} ({100*total_full_blocks/total_blocks:.1f}%)")
    print(f"  Empty blocks: {total_blocks - total_mask_blocks - total_full_blocks}")
    
    print("\n✓ K2Q kernel executed successfully!")
    return True


if __name__ == "__main__":
    # Run Q2K tests
    print("=" * 60)
    print("Test 1: Q2K (Forward) with check_q_boundary=True")
    print("=" * 60)
    run_q2k_test(seqlen_q=20480, seqlen_k=20480, n_func=3, Q_BLOCK_SIZE=256, KV_BLOCK_SIZE=128, check_q_boundary=True)
    
    print("\n" + "=" * 60)
    print("Test 2: Q2K Non-divisible seqlen")
    print("=" * 60)
    run_q2k_test(seqlen_q=21111, seqlen_k=21111, n_func=3, Q_BLOCK_SIZE=256, KV_BLOCK_SIZE=128, check_q_boundary=True)
    
    # Run K2Q tests
    print("\n" + "=" * 60)
    print("Test 3: K2Q (Backward)")
    print("=" * 60)
    run_k2q_test(seqlen_q=20480, seqlen_k=20480, n_func=3, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128)
    
    print("\n" + "=" * 60)
    print("Test 4: K2Q Non-divisible seqlen")
    print("=" * 60)
    run_k2q_test(seqlen_q=21111, seqlen_k=21111, n_func=3, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128)
