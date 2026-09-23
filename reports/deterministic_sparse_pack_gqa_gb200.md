# Deterministic sparse PackGQA backward: GB200 validation

Date: 2026-09-23. Branch: `feat/deterministic-sparse-pack-gqa`.
Base: `megaattention/ring-ready-fa4-beta31` at
`5a1dc7e3648c5ef55c8040b75c0a667467e4d829`.

## Implementation

One CTA owns a KV head and KV tile. A head-major CuTe view combines the group's
Q heads into a single continuous loop; Q/dO are not transposed or copied.
The sparse iterator repeats the original Q list for each group head. dQ uses
the original per-head sparse write tickets, while dK/dV accumulate in FP32 in
the CTA and are stored once as BF16. Existing preprocess and dQ postprocess
remain. Public output layouts and default unpacked behavior are preserved.
An additional fix makes deterministic sparse ticket lookup compile when the
optional full-block write-order tensor is absent.

## Hardware and method

- Single NVIDIA GB200, SM100, 152 SMs; four GPUs available on the machine.
- PyTorch 2.9.1, CUDA 13.1, CuTeDSL 4.8.0, Quack 0.6.5, TVM-FFI 0.1.14.post0.
- B=1, Hq=64, Hkv=8, D=128, BF16, deterministic, 1CTA for both paths.
- Sparse block size: Q=256, KV=128. `dense` uses all full blocks; `mixed`
  combines full and triangular partial blocks and empty KV columns, with
  approximately 57% active coarse blocks. Masks are shared across heads.
- Every benchmark case passed independent FP32 PyTorch autograd checks and
  eager bitwise repeatability before timing. Maximum relative RMS gradient
  error over all cases and both paths: **0.2373%**.
- Ten backward calls per CUDA graph, nine total-time samples, alternating
  unpacked/packed measurement order. JIT compilation, mask construction, and
  reference evaluation are excluded. Total GPU time includes zero kernels,
  preprocess, main backward, and gradient postprocessing.
- Main-kernel timing uses a separate graph with external CUDA event nodes around
  the compiled main kernel, 5 replays x 10 events. These are independent
  measurements; instrumentation and clock/cache differences mean they are not
  additive components of the total-time measurements.
- No clock locking or performance pass/fail threshold. Other GPUs ran correctness
  checks. Near-unity speedups should be interpreted as no demonstrated gain.

## Measurements

Times are medians in microseconds. Speedup is unpacked / packed.

| Q | KV | Mask | Main unpacked | Main packed | Main speedup | Total unpacked | Total packed | Total speedup |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | 8192 | dense | 742.5 | 430.4 | 1.725x | 780.6 | 464.6 | 1.680x |
| 512 | 8192 | mixed | 656.5 | 412.0 | 1.593x | 695.0 | 421.8 | 1.647x |
| 512 | 32768 | dense | 2881.5 | 1549.7 | 1.859x | 2991.6 | 1503.0 | 1.990x |
| 512 | 32768 | mixed | 2527.6 | 1416.5 | 1.784x | 2635.6 | 1424.3 | 1.850x |
| 512 | 65536 | dense | 5732.7 | 2986.0 | 1.920x | 5952.6 | 2991.5 | 1.990x |
| 512 | 65536 | mixed | 5007.7 | 2733.1 | 1.832x | 5206.4 | 2743.1 | 1.898x |
| 1024 | 8192 | dense | 1071.1 | 885.1 | 1.210x | 1120.2 | 851.6 | 1.315x |
| 1024 | 8192 | mixed | 1076.3 | 1070.8 | 1.005x | 1125.5 | 1077.7 | 1.044x |
| 1024 | 32768 | dense | 4163.9 | 3021.0 | 1.378x | 4284.0 | 2928.0 | 1.463x |
| 1024 | 32768 | mixed | 4190.5 | 4161.7 | 1.007x | 4309.6 | 4164.0 | 1.035x |
| 1024 | 65536 | dense | 8284.1 | 5804.8 | 1.427x | 8496.7 | 5829.0 | 1.458x |
| 1024 | 65536 | mixed | 8313.2 | 8223.1 | 1.011x | 8517.9 | 8211.4 | 1.037x |

Q=512 shows a clear benefit: 1.59–1.92x for the main kernel and 1.65–1.99x
for complete backward. Q=1024 dense-in-sparse improves 1.21–1.43x at the main
kernel, while the mixed mask is effectively unchanged (1.005–1.011x).
Longer Q loops alone do not guarantee a gain for every mask/workload.

## Independent Nsight Systems check

A node-level CUDA graph trace for Q=512, KV=8192, mixed mask, five calls per
path confirms:

- Main-kernel CTA grid: **4096 -> 512**, exactly an 8x reduction.
- Both paths: 512 threads/CTA, 128 registers/thread, 232448 bytes shared memory.
- Main-kernel median: **657.952 -> 410.208 us**, **1.604x**.
- Kernel nodes per backward: **10 -> 4**. Unpacked launches two FP32 dKV zero
  kernels, three semaphore zero kernels, preprocess, main, and three gradient
  postprocess kernels. Packed launches one dQ semaphore zero kernel,
  preprocess, main, and one dQ postprocess kernel.
- No Q/dO packing-copy kernel; no packed-path dKV global accumulator or dKV
  postprocess kernel.

Raw profiles are outside git: `/tmp/fa_pack_trace_nodes.nsys-rep` and
`/tmp/fa_pack_trace_nodes.sqlite`.

## Correctness and checks

- 22 core tests passed: BF16 dQ/dK/dV vs FP32 reference, MHA/GQA/MQA,
  D64/D128, Q256/512/1024, batch 1/2, uneven KV tails, mixed/full/empty sparse
  work, repeatability, caller output buffers, public autograd, and CUDA graph
  replay with both sparse ordering directions and KV block sizes 128/256.
- Three additional boundary tests passed after fixing the absent-full-list
  compilation case: fully masked Q rows, original sequence coordinates in
  causal mask_mod, optional full-block metadata, and ready KV signals.
  Both packed and unpacked paths were checked.
- Existing deterministic block-sparse regression selection:
  **46 passed, 12 skipped, 1517 deselected**. Skips come from the existing test
  parametrization, not new PackGQA checks.
- `compute-sanitizer --tool memcheck` on the mixed sparse smoke:
  **0 errors**, with reference and repeatability checks also passing.
- Ruff lint: all seven changed/new Python files passed. Ruff format: six passed;
  `interface.py` has pre-existing formatting failures at the base commit.
  Modified ranges were formatted without reformatting unrelated baseline code.
- `git diff --check` passed.

CuTeDSL 4.8 emits compatibility/deprecation warnings for existing AuxData and
upstream APIs; they did not prevent compilation or GPU validation.

## Reproduce

```sh
# From this checkout, using an environment with the versions above:
FA_DISABLE_2CTA=1 PYTHONPATH=. pytest -q tests/cute/test_pack_gqa_bwd.py
FA_DISABLE_2CTA=1 PYTHONPATH=. pytest -q tests/cute/test_mask_mod.py \
  -k test_block_sparse_bwd_deterministic
FA_DISABLE_2CTA=1 PYTHONPATH=. python benchmarks/benchmark_sparse_pack_gqa_bwd.py \
  --q 512 1024 --kv 8192 32768 65536 --output /tmp/pack-gqa-results.json
```

The environment used here is `/tmp/fa-pack-gqa-venv`. Individual timing samples,
min/max, correctness errors, versions, and timing-source diff identity are in
`/tmp/fa_pack_benchmark_full.json`. Its diff SHA256 is `bfdd5987d3da65a0cd68f30b792590bb19cfbfb8a2419f525558d21e1ccacef6`.
The matrix was measured before the final optional-full-list compilation fix;
its cases all have full-list tensors. The subsequent Nsight check and three
boundary tests use the fixed source.

## Scope and remaining work

This is an opt-in 1CTA fixed-length path. Q length must be divisible by the
sparse Q block size and 128; the metadata head dimension must be 1. Native
causal/local pruning, score modifications, learnable sinks, varlen, 2CTA,
paired work maps, and external ring dKV accumulators/completion counters are
rejected. SM110 is allowed but was not hardware-tested. No multi-node ring
integration or complete upstream test suite was run.
