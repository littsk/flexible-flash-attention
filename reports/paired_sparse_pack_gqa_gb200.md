# Paired 2CTA PackGQA on GB200

Extends `188a61c` with deterministic paired sparse PackGQA backward. The
kernel processes all query heads sharing a KV head inside one 2CTA cluster
and writes BF16 dK/dV directly. It does not use external FP32 accumulators,
dKV postprocessing, or communication completion counters.

The caller supplies contracted paired CSR/tickets/work maps as described in
[the design](../design/deterministic_sparse_pack_gqa.md). Original visibility
bits retain Hq indexing. MegaAttention's integration prepares this metadata
outside the timed region; its planner and dispatch benchmark live in that
separate repository. The optional paired integration tests require that
checkout's `prepare_paired_bwd_mask(..., pack_gqa=True)` implementation.

## Measurement

GB200 (SM100), PyTorch 2.9.1/CUDA 13.1, CuTeDSL 4.8.0. B=1, Hq=64, Hkv=8,
D=128, BF16, deterministic. Logical CP128/256, 16384 Q tokens per rank,
seed42, sample_1m document packs, shared Q/KV MinHeap dispatch, chunk_ratio=1.
The same dispatch partitions and tensors are used for both variants. Paired
metadata follows MegaAttention's SU32/12-channel work ordering.

Each logical rank is replayed on one GPU without communication or overlap.
Time includes all backward GPU kernels, including initialization and dQ
pre/postprocessing, but excludes forward, planning, JIT and CPU launch gaps.
CUDA graphs contain three backward calls, with three warmup replays and seven
alternating measurement rounds. Each table entry is the median of the per-rank
medians for ranks 0, CP/2 and CP-1. Units: **ms / rank / backward**.

| CP | Mask | Chunk | Paired unpacked | Paired packed | Speedup |
|---|---|---:|---:|---:|---:|
|128|Full document|512|130.818|78.541|1.666x|
|128|Full document|1024|114.837|78.709|1.459x|
|128|Causal document|512|74.607|47.888|1.558x|
|128|Causal document|1024|68.003|44.331|1.534x|
|256|Full document|512|164.305|83.786|1.961x|
|256|Full document|1024|131.193|84.857|1.546x|
|256|Causal document|512|89.589|45.902|1.952x|
|256|Causal document|1024|71.383|46.514|1.535x|

All 24 same-rank comparisons improve by 1.449x–2.051x. Maximum sample CV is
2.59%; all samples are retained. No GPU clock locking was used. For CP256 full
chunk512 rank0, the original sparse tile work is 85.419 TFLOP per backward;
full-backward throughput is 519.4/1019.5 TFLOP/s unpacked/packed. Work is counted
as five GEMMs, with each FMA counted as two operations.

Nsight confirms cluster=(2,1,1) for both variants and 721792/90224 CTAs, exactly
an eightfold reduction. Packed backward launches four kernels: int semaphore
initialization, preprocessing, paired main kernel, and dQ postprocessing.
There is no FP32 dKV initialization or dKV conversion kernel.

## Validation

The tested kernel sources were copied byte-for-byte into this checkout:

- 29 native PackGQA tests passed, including four paired cases covering a
  single KV tile, odd-K padding, mixed/empty masks, caller-provided BF16
  outputs, independent FP32 gradients and bitwise CUDA graph replay.
- CUDA memcheck: four paired boundary cases, zero errors.
- MegaAttention host/planner/FLOPs tests: 25 passed; six CP4 dispatch cases
  passed independent FP32/BF16 reference and repeatability checks.
- All 24 performance shapes passed four-way gradient agreement and eager/graph
  repeatability; maximum relative RMS error was 0.0001375. Full-size independent
  PyTorch references were not materialized.

The odd-K fix prevents the metadata-only mate from issuing BF16 output stores
while preserving cluster synchronization. Cold/repeated numerical probes,
a neutral perturbation control and generated SASS inspection verified the fix.
