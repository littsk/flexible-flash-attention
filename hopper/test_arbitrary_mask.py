# Arbitrary mask test script for Hopper backend
# This file re-exports tests from the main test file with Hopper backend enabled
#
# Usage:
#   pytest test_arbitrary_mask.py                    # Run tests with Hopper backend
#   python test_arbitrary_mask.py                    # Run default test directly
#
# The main test implementation is in tests/cute/test_arbitrary_mask.py
# This file simply sets the backend to "hopper" and imports the tests.

import os
import sys

# Set backend to hopper before importing tests
os.environ["FLASH_ATTN_BACKEND"] = "hopper"

# Add the tests/cute directory to path
tests_cute_dir = os.path.join(os.path.dirname(__file__), "../tests/cute")
tests_cute_dir = os.path.abspath(tests_cute_dir)
if tests_cute_dir not in sys.path:
    sys.path.insert(0, tests_cute_dir)

# Add project root to path for flash_attn imports
project_root = os.path.join(os.path.dirname(__file__), "..")
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import everything from the main test file
from test_arbitrary_mask import (
    # Configuration
    BACKEND,
    DISABLE_BACKWARD,
    COMPUTE_CAPABILITY,
    COMPILED_HDIMS,
    HAS_CREATE_BLOCK_MASK_CUDA,
    NATIVE_MASK_PATTERNS,
    
    # Helper functions
    flex_arbitrary_mask,
    apply_arbitrary_mask_to_qk,
    create_tensors,
    compute_reference_arbitrary,
    compare_linear_sparse_tensors,
    bhqk_to_linear_sparse_tensors_local,
    
    # Test functions
    _run_mask_test,
    test_arbitrary_mask,
    
    # Benchmark function
    benchmark_arbitrary_mask,
)

# Import arbitrary_func_tensor from mask_definitions
from flash_attn.cute.mask_definitions import arbitrary_func_tensor

# Verify backend is set correctly
assert BACKEND == "hopper", f"Expected backend 'hopper', got '{BACKEND}'"

if __name__ == "__main__":
    # Set seed for reproducibility
    import torch
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
    )