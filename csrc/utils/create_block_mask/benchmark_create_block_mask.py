"""
Benchmark script for create_block_mask CUDA kernels.

Tests performance of:
  - create_q2k_csr_sparse_from_func: Q2K (Forward) CSR format  - default 256x128 tile
  - create_k2q_csr_sparse_from_func: K2Q (Backward) CSR format - default 128x128 tile

Supported attention patterns:
  - random: Random intervals (default)
  - causal: Causal attention (lower triangular, token-level)
  - diagonal: Block-level diagonal pattern (each q_block only attends to corresponding kv_block)
  - all: Run benchmarks for all patterns

Usage:
    # Random pattern (default)
    python benchmark_create_block_mask.py --seqlen 220000
    python benchmark_create_block_mask.py --seqlen 220000 --n_func 7 --iterations 100

    # Causal pattern (token-level lower triangular)
    python benchmark_create_block_mask.py --seqlen 220000 --pattern causal

    # Diagonal pattern (block-level diagonal)
    python benchmark_create_block_mask.py --seqlen 220000 --pattern diagonal --block_size 4096

    # All patterns
    python benchmark_create_block_mask.py --seqlen 220000 --pattern all
"""

import torch
import time
import argparse
from dataclasses import dataclass

# Import our CUDA kernel
try:
    import create_block_mask_cuda
except ImportError:
    create_block_mask_cuda = None
    print("Error: create_block_mask_cuda not found. Please build it first with: pip install -e .")
    exit(1)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    seqlen_q: int
    seqlen_k: int
    n_func: int
    q_block_size: int
    kv_block_size: int
    check_q_boundary: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    kernel_type: str  # "q2k_csr" or "k2q_csr"
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_gblocks_per_sec: float  # billion blocks per second
    total_mask_blocks: int
    total_full_blocks: int


def generate_func_tensor(
    seqlen_q: int,
    seqlen_k: int,
    n_func: int = 3,
    pattern: str = "random",
    block_size: int = 4096,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generate func_tensor for different attention patterns.

    Args:
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        n_func: Number of functions (only for random pattern)
        pattern: Attention pattern, one of:
            - "random": Random intervals
            - "causal": Causal/lower-triangular (token-level)
            - "diagonal": Block-level diagonal (each q_block attends to corresponding kv_block)
        block_size: Block size for diagonal pattern
        device: Device to create tensor on

    Returns:
        func_tensor: [1, 1, n_func, seqlen_q + 256]
    """
    B, H = 1, 1
    q_indices = torch.arange(seqlen_q, dtype=torch.int32, device=device)

    if pattern == "random":
        # Random pattern: multiple random intervals
        func_tensor = torch.zeros(B, H, n_func, seqlen_q + 256, dtype=torch.int32, device=device)
        coef = 1.0 / n_func
        for i in range(n_func):
            low = int(i * coef * seqlen_k)
            high = int((i + 1) * coef * seqlen_k)
            if high <= low:
                high = low + 1
            func_tensor[:, :, i, :seqlen_q] = torch.randint(low, high, size=(B, H, seqlen_q), dtype=torch.int32, device=device)
        return func_tensor.contiguous()

    elif pattern == "causal":
        # Causal pattern: each q_token attends to [0, q_idx + 1)
        # This creates lower-triangular attention (token-level)
        func_tensor = torch.zeros(B, H, 1, seqlen_q + 256, dtype=torch.int32, device=device)
        func_tensor[:, :, 0, :seqlen_q] = q_indices + 1
        return func_tensor.contiguous()

    elif pattern == "diagonal":
        # Diagonal pattern: block-level diagonal
        # Each q_block only attends to the corresponding kv_block on the diagonal
        # For q_idx, valid kv range is [(q_idx // block_size) * block_size, ((q_idx // block_size) + 1) * block_size)
        # Using func encoding: [0, F0) where F0 = 0 means first interval is empty
        #                      [F1, F2) where F1 = block_start, F2 = block_end
        func_tensor = torch.zeros(B, H, 3, seqlen_q + 256, dtype=torch.int32, device=device)

        # F0 = 0 (first interval [0, F0) is empty)
        func_tensor[:, :, 0, :seqlen_q] = 0

        # F1 = (q_idx // block_size) * block_size (block start)
        block_start = (q_indices // block_size) * block_size
        func_tensor[:, :, 1, :seqlen_q] = block_start

        # F2 = min(((q_idx // block_size) + 1) * block_size, seqlen_k) (block end)
        block_end = torch.minimum(block_start + block_size,
                                   torch.full_like(q_indices, seqlen_k))
        func_tensor[:, :, 2, :seqlen_q] = block_end

        return func_tensor.contiguous()

    else:
        raise ValueError(f"Unknown pattern: {pattern}. Supported: random, causal, diagonal")


def benchmark_q2k_csr(
    config: BenchmarkConfig,
    warmup: int = 10,
    iterations: int = 100,
    func_tensor: torch.Tensor = None
) -> BenchmarkResult:
    """Benchmark create_q2k_csr_sparse_from_func.

    Note: Optimization is controlled at compile time via macros:
      - DISABLE_REG_CACHE: Disable register caching optimization
      - DISABLE_KV_RANGE_OPT: Disable kv_block range optimization
    """

    if func_tensor is None:
        func_tensor = generate_func_tensor(config.seqlen_q, config.seqlen_k, config.n_func)

    # Warmup
    for _ in range(warmup):
        _ = create_block_mask_cuda.create_q2k_csr_sparse_from_func(
            func_tensor,
            config.seqlen_q,
            config.seqlen_k,
            config.q_block_size,
            config.kv_block_size,
            config.check_q_boundary
        )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    total_mask = 0
    total_full = 0

    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = create_block_mask_cuda.create_q2k_csr_sparse_from_func(
            func_tensor,
            config.seqlen_q,
            config.seqlen_k,
            config.q_block_size,
            config.kv_block_size,
            config.check_q_boundary
        )

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

        # Get block counts from last iteration
        total_mask = len(result[2])  # mask_block_idx
        total_full = len(result[5])  # full_block_idx

    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()

    # Calculate throughput (total blocks processed per second)
    num_q_blocks = (config.seqlen_q + config.q_block_size - 1) // config.q_block_size
    num_kv_blocks = (config.seqlen_k + config.kv_block_size - 1) // config.kv_block_size
    total_blocks = num_q_blocks * num_kv_blocks
    throughput = (total_blocks / (mean_time / 1000)) / 1e9  # billion blocks per second

    return BenchmarkResult(
        config=config,
        kernel_type="q2k_csr",
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        throughput_gblocks_per_sec=throughput,
        total_mask_blocks=total_mask,
        total_full_blocks=total_full
    )


def benchmark_k2q_csr(
    config: BenchmarkConfig,
    warmup: int = 10,
    iterations: int = 100,
    func_tensor: torch.Tensor = None
) -> BenchmarkResult:
    """Benchmark create_k2q_csr_sparse_from_func.

    Note: Optimization is controlled at compile time via macros:
      - DISABLE_REG_CACHE: Disable register caching optimization
      - DISABLE_KV_RANGE_OPT: Disable kv_block range optimization (also disables K2Q precompute)
    """

    if func_tensor is None:
        func_tensor = generate_func_tensor(config.seqlen_q, config.seqlen_k, config.n_func)

    # Warmup
    for _ in range(warmup):
        _ = create_block_mask_cuda.create_k2q_csr_sparse_from_func(
            func_tensor,
            config.seqlen_q,
            config.seqlen_k,
            config.q_block_size,
            config.kv_block_size
        )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    total_mask = 0
    total_full = 0

    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = create_block_mask_cuda.create_k2q_csr_sparse_from_func(
            func_tensor,
            config.seqlen_q,
            config.seqlen_k,
            config.q_block_size,
            config.kv_block_size
        )

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

        # Get block counts from last iteration
        total_mask = len(result[2])  # mask_block_idx
        total_full = len(result[5])  # full_block_idx

    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()

    # Calculate throughput (total blocks processed per second)
    num_q_blocks = (config.seqlen_q + config.q_block_size - 1) // config.q_block_size
    num_kv_blocks = (config.seqlen_k + config.kv_block_size - 1) // config.kv_block_size
    total_blocks = num_q_blocks * num_kv_blocks
    throughput = (total_blocks / (mean_time / 1000)) / 1e9  # billion blocks per second

    return BenchmarkResult(
        config=config,
        kernel_type="k2q_csr",
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        throughput_gblocks_per_sec=throughput,
        total_mask_blocks=total_mask,
        total_full_blocks=total_full
    )


def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    cfg = result.config
    num_q_blocks = (cfg.seqlen_q + cfg.q_block_size - 1) // cfg.q_block_size
    num_kv_blocks = (cfg.seqlen_k + cfg.kv_block_size - 1) // cfg.kv_block_size
    total_blocks = num_q_blocks * num_kv_blocks

    print(f"  {result.kernel_type.upper()}: "
          f"mean={result.mean_time_ms:.3f}ms +/- {result.std_time_ms:.3f}ms "
          f"[min={result.min_time_ms:.3f}ms, max={result.max_time_ms:.3f}ms] "
          f"| {result.throughput_gblocks_per_sec:.2f} Gblocks/s")
    print(f"    Blocks: {total_blocks:,} total, {result.total_mask_blocks:,} mask, {result.total_full_blocks:,} full")


def run_benchmark_with_pattern(
    seqlen: int,
    pattern: str,
    block_size: int = 4096,
    n_func: int = 7,
    warmup: int = 10,
    iterations: int = 100
):
    """
    Run benchmark for a specific attention pattern.

    Args:
        seqlen: Sequence length
        pattern: One of "random", "causal", "diagonal"
        block_size: Block size for diagonal pattern
        n_func: Number of functions (only for random pattern)
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
    """
    print("=" * 70)
    print(f"BENCHMARK: seqlen={seqlen}, pattern={pattern}")
    if pattern == "diagonal":
        print(f"           block_size={block_size}")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")

    # Generate func_tensor based on pattern
    func_tensor = generate_func_tensor(seqlen, seqlen, n_func=n_func, pattern=pattern, block_size=block_size)
    actual_n_func = func_tensor.shape[2]

    # Calculate block info
    num_q_blocks_256 = (seqlen + 256 - 1) // 256
    num_kv_blocks_128 = (seqlen + 128 - 1) // 128
    num_q_blocks_128 = (seqlen + 128 - 1) // 128

    print(f"Q2K Blocks (256x128): {num_q_blocks_256} x {num_kv_blocks_128} = {num_q_blocks_256 * num_kv_blocks_128:,}")
    print(f"K2Q Blocks (128x128): {num_q_blocks_128} x {num_kv_blocks_128} = {num_q_blocks_128 * num_kv_blocks_128:,}")
    print()

    # Q2K with 256x128 (forward pass default)
    config_q2k = BenchmarkConfig(
        seqlen_q=seqlen, seqlen_k=seqlen, n_func=actual_n_func,
        q_block_size=256, kv_block_size=128, check_q_boundary=True
    )
    result_q2k = benchmark_q2k_csr(config_q2k, warmup, iterations, func_tensor)
    print(f"Q2K CSR (256x128):")
    print_result(result_q2k)

    # K2Q with 128x128 (backward pass default)
    config_k2q = BenchmarkConfig(
        seqlen_q=seqlen, seqlen_k=seqlen, n_func=actual_n_func,
        q_block_size=128, kv_block_size=128, check_q_boundary=True
    )
    result_k2q = benchmark_k2q_csr(config_k2q, warmup, iterations, func_tensor)
    print(f"\nK2Q CSR (128x128):")
    print_result(result_k2q)


def run_benchmark(seqlen: int = 220000, n_func: int = 7, warmup: int = 10, iterations: int = 100):
    """Run benchmark for Q2K and K2Q CSR kernels with random pattern."""
    run_benchmark_with_pattern(seqlen, "random", n_func=n_func, warmup=warmup, iterations=iterations)


def run_all_pattern_benchmarks(seqlen: int = 220000, block_size: int = 4096, warmup: int = 10, iterations: int = 100):
    """Run benchmarks for all attention patterns."""
    patterns = ["random", "causal", "diagonal"]

    for pattern in patterns:
        run_benchmark_with_pattern(
            seqlen=seqlen,
            pattern=pattern,
            block_size=block_size,
            n_func=7,
            warmup=warmup,
            iterations=iterations
        )
        print("\n")


def run_optimization_comparison(
    seqlen: int = 220000,
    pattern: str = "diagonal",
    block_size: int = 4096,
    n_func: int = 7,
    warmup: int = 10,
    iterations: int = 50
):
    """Compare performance with and without kv_block range optimization."""

    # Generate func_tensor
    func_tensor = generate_func_tensor(seqlen, seqlen, n_func, pattern, block_size)
    actual_n_func = func_tensor.shape[2]

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 80)
    print(f"OPTIMIZATION COMPARISON: seqlen={seqlen:,}, pattern={pattern}")
    if pattern == "diagonal":
        print(f"                        block_size={block_size}")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")

    # Calculate block info
    num_q_blocks_256 = (seqlen + 256 - 1) // 256
    num_kv_blocks_128 = (seqlen + 128 - 1) // 128
    num_q_blocks_128 = (seqlen + 128 - 1) // 128

    print(f"Q2K Blocks (256x128): {num_q_blocks_256} x {num_kv_blocks_128} = {num_q_blocks_256 * num_kv_blocks_128:,}")
    print(f"K2Q Blocks (128x128): {num_q_blocks_128} x {num_kv_blocks_128} = {num_q_blocks_128 * num_kv_blocks_128:,}")
    print()

    print("=" * 80)
    print("NOTE: Optimization comparison is now controlled at compile time via macros.")
    print("=" * 80)
    print()
    print("To compare optimized vs unoptimized performance, rebuild with different macros:")
    print()
    print("  # Build with all optimizations (default)")
    print("  python setup.py build_ext --inplace")
    print()
    print("  # Build without register caching")
    print("  DISABLE_REG_CACHE=1 python setup.py build_ext --inplace")
    print()
    print("  # Build without kv range optimization")
    print("  DISABLE_KV_RANGE_OPT=1 python setup.py build_ext --inplace")
    print()
    print("  # Build without any optimizations")
    print("  DISABLE_REG_CACHE=1 DISABLE_KV_RANGE_OPT=1 python setup.py build_ext --inplace")
    print()
    print("=" * 80)
    print()

    # Run current build benchmark
    config_q2k = BenchmarkConfig(
        seqlen_q=seqlen, seqlen_k=seqlen, n_func=actual_n_func,
        q_block_size=256, kv_block_size=128, check_q_boundary=True
    )
    config_k2q = BenchmarkConfig(
        seqlen_q=seqlen, seqlen_k=seqlen, n_func=actual_n_func,
        q_block_size=128, kv_block_size=128, check_q_boundary=True
    )

    print("Current build performance:")
    print("-" * 40)
    print("Q2K CSR (256x128):")
    result_q2k = benchmark_q2k_csr(config_q2k, warmup, iterations, func_tensor)
    print_result(result_q2k)

    print()
    print("K2Q CSR (128x128):")
    result_k2q = benchmark_k2q_csr(config_k2q, warmup, iterations, func_tensor)
    print_result(result_k2q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark create_block_mask CUDA kernels")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--seqlen", type=int, default=220000, help="Sequence length")
    parser.add_argument("--n_func", type=int, default=7, help="Number of functions (for random pattern)")
    parser.add_argument("--pattern", type=str, default="random",
                        choices=["random", "causal", "diagonal", "all"],
                        help="Attention pattern to benchmark")
    parser.add_argument("--block_size", type=int, default=4096, help="Block size for diagonal pattern")
    parser.add_argument("--compare", action="store_true",
                        help="Compare performance with and without optimization")

    args = parser.parse_args()

    if args.compare:
        # Run optimization comparison
        run_optimization_comparison(
            seqlen=args.seqlen,
            pattern=args.pattern,
            block_size=args.block_size,
            n_func=args.n_func,
            warmup=args.warmup,
            iterations=args.iterations
        )
    elif args.pattern == "all":
        run_all_pattern_benchmarks(
            seqlen=args.seqlen,
            block_size=args.block_size,
            warmup=args.warmup,
            iterations=args.iterations
        )
    else:
        run_benchmark_with_pattern(
            seqlen=args.seqlen,
            pattern=args.pattern,
            block_size=args.block_size,
            n_func=args.n_func,
            warmup=args.warmup,
            iterations=args.iterations
        )
