# FlashAttention-4 (CuTeDSL)

FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.

## Installation

```sh
pip install flash-attn-4
```

If you're on CUDA 13, install with the `cu13` extra for best performance:

```sh
pip install "flash-attn-4[cu13]"
```

## Usage

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
```

## Development

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
pytest tests/cute/
```

## Deterministic block-sparse PackGQA backward

The SM100/SM110 backward supports an opt-in head-major PackGQA path. One CTA
owns a KV head and KV tile and processes every query head in its group without
draining the Q pipeline. Q/dO use zero-copy views; dK/dV accumulate in FP32
inside the CTA and are stored directly as BF16. The public tensor layouts
remain unchanged. No global dKV accumulator or dKV postprocess is needed.

Set `FA_DISABLE_2CTA=1` before importing the library. Call `_flash_attn_bwd`
with `pack_gqa=True`, `deterministic=True`, `mask_mod`, and backward
`block_sparse_tensors` containing `dq_write_order`, `dq_write_order_full`
(when full blocks exist), and `spt`. Through `flash_attn_func`, explicitly set
`pack_gqa=True`, `deterministic=True`, and supply both forward and backward
sparse metadata. `pack_gqa=None` preserves the original unpacked backward.

The initial path requires BF16, fixed-length Q/KV, head dimensions <=128,
broadcast sparse head dimension (size 1), and Q length divisible by both the
sparse Q block size and 128. Express causal/local masks through `mask_mod`.
Native causal/local pruning, score modifications, learnable sinks, generic 2CTA,
and external accumulators/Q-head finalizer counters are explicitly unsupported.
The optional packed BF16 ready counter is described below. K/V readiness signals retain their existing semantics.
The Q sequence extent is static in the packed TMA layout, so shape/stride
changes select a separate compiled variant.

For paired 2CTA, set `FA_DISABLE_2CTA=0` and call `_flash_attn_bwd` with both
`paired_sparse_bwd=True` and `pack_gqa=True`. The pair planner must contract
its non-broadcast CSR/ticket heads to Hkv and keep one work-map row per KV-head/pair, retaining
the original Hq-indexed visibility bits for `mask_mod`. MegaAttention provides
`prepare_paired_bwd_mask(..., pack_gqa=True)` for this preparation. This path
requires D128 and sparse blocks `(256,128)`. Its two CTAs jointly traverse all
G query heads and directly store BF16 dK/dV; no external FP32 buffer is needed.
Odd-K mates are metadata-only and must never issue output stores.

Deterministic means bitwise repeatability within a fixed kernel configuration;
packed and unpacked dK/dV can differ numerically due to summation order.
See the [design](../../design/deterministic_sparse_pack_gqa.md) and
[paired 2CTA validation and CP-dispatch results](../../reports/paired_sparse_pack_gqa_gb200.md).

```sh
FA_DISABLE_2CTA=1 PYTHONPATH=. pytest -q tests/cute/test_pack_gqa_bwd.py
# Default: paired 2CTA, comparing unpacked and packed backward.
PYTHONPATH=. python benchmarks/benchmark_sparse_pack_gqa_bwd.py \
  --q 512 1024 --kv 8192 32768 65536 --output /tmp/pack-gqa-results.json

# Explicitly retain the 1CTA comparison.
PYTHONPATH=. python benchmarks/benchmark_sparse_pack_gqa_bwd.py \
  --cta-group-size 1 --output /tmp/pack-gqa-1cta-results.json
```

The benchmark defaults to `--cta-group-size 2`; the CLI selects
`FA_DISABLE_2CTA` before importing FA4. Both variants use the selected CTA mode.
Paired mode requires Q divisible by 256 and KV divisible by 128. Its standalone
fixture derives mirrored pair CSR/tickets from a 256-token KV block mask, uses
ascending KV-pair/head work order, and contracts the packed work map to Hkv.
The original mask callback handles pair-union holes, so no MegaAttention checkout
is required. This synthetic benchmark does not reproduce CP dispatch scheduling.
JSON records `cta_mode`, `cta_group_size`, original `block_density`, and paired
`executed_block_density` (including the padded physical mate for odd-K).

The benchmark fixes B=1, Hq=64, Hkv=8, D=128. It checks FP32-reference
gradients and bitwise repeatability before timing. Total backward time uses
CUDA events around an uninstrumented CUDA graph. Main-kernel time uses a
separate graph with external CUDA event nodes immediately around the compiled
backward kernel; preprocess/reset and postprocess still run for each invocation.
Both timing graphs reuse the same output buffers. Their medians are sampled
separately and should not be subtracted to estimate preprocessing overhead.
It reports individual samples, median/min/max, source identity, and versions.

The [Q512/Q1024/Q2048 comparison](../../reports/paired_pack_gqa_q_sweep_gb200.md)
reports paired pack/no-pack main-kernel and full-backward timings.


Packed sparse backward optionally accepts `dkv_done_counter`: a caller-reset
int32 `[B*Hkv*N]` or (B=1) `[Hkv,capacity]` buffer, N=ceil(Sk/128). An active
physical tile increments its slot once after complete BF16 dK/dV stores;
inactive and padded tiles leave it zero. Consumers wait with acquire semantics
using original tile activity as expected (0/1). This signal is local readiness,
not remote delivery; packed dK is already scaled. See
[the producer-ready contract](../../design/deterministic_sparse_pack_gqa.md#packed-bf16-producer-ready-signal).
