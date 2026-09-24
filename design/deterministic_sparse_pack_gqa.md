# Deterministic block-sparse PackGQA backward

Base: `megaattention/ring-ready-fa4-beta31`, commit
`5a1dc7e3648c5ef55c8040b75c0a667467e4d829`.

## Objective

For Hq=64 and Hkv=8, give each backward CTA one KV head and KV tile,
and process all eight associated query heads in one continuous Q loop.
Keep K/V resident and dK/dV accumulators in FP32 throughout that loop.
Store final dK and dV directly as BF16, without cross-head global atomics,
FP32 dKV scratch, dKV semaphores, or a dKV conversion kernel.
Preserve deterministic dQ accumulation using the existing sparse write-order
metadata. Determinism means bitwise repeatability for a fixed configuration;
the packed and unpacked paths may differ because their summation orders differ.

## Layout and execution

Use head-major logical packing: p = r * Sq + q, hq = hkv * G + r.
Q and dO use nested CuTe layouts over their existing storage: no global tensor
transpose or data copy. LSE, D=sum(O*dO), dQ accumulators, and dQ semaphores
already store adjacent query heads in head-major order; reinterpret those
buffers for the kernel while retaining existing preprocess/postprocess paths.

For L physical Q tiles in a KV tile's sparse list, execute G*L iterations.
Decode the group head with i // L, and the original sparse iteration with i % L.
The physical packed Q tile is original_q_tile + r * (Sq / tile_m).
Producer, MMA, compute, and dQ reduction must use identical iteration counts
and tile mappings. No pipeline drain or dKV accumulator reset at head boundaries.
The CTA grid has Hkv heads instead of Hq heads.

Sparse metadata stays in original token coordinates. Its head dimension is
broadcast (size 1), including counts, indices, and deterministic write orders.
Keep sparse block size and Q subtile factor unchanged; head expansion is an
independent factor. Partial blocks call mask_mod with original (batch,hq,q,k)
coordinates. Full blocks retain the existing boundary checks.

## Determinism and output ownership

Each (batch,hkv,physical KV tile) has one owner. It performs a fixed ordered
FP32 sum over all participating Q rows/heads and converts once to BF16.
Empty sparse KV tiles explicitly write zero dK/dV.

Each dQ tile still has independent state per query head. For iteration i,
look up the existing dq_write_order using i % L, and address the semaphore
and accumulator using the corresponding packed Q tile. Existing release and
completion ordering must remain intact. No ticket is consumed for skipped
sparse blocks. This avoids introducing a new determinism protocol.

## Initial supported contract

- SM100/SM110, BF16 Q/K/V/dO, deterministic fixed-length block-sparse backward.
- Shared-head sparse metadata and mask; Hq divisible by Hkv.
- Head dimensions up to 128 as supported by the existing kernel.
- Sq is a multiple of the sparse Q block size and tile_m, so a virtual head
  boundary cannot fall inside a Q tile or a coarse sparse Q block.
- Start with the existing 1CTA path; evaluate 2CTA separately only after
  correctness is established. The unpacked benchmark uses the same CTA mode.
- Causal/local semantics can be expressed in mask_mod. Native causal/local
  pruning, score modifications, varlen, paired sparse work maps, and external
  ring dKV accumulators/completion protocols are outside this initial path.
  Explicit unsupported combinations must fail clearly, never silently change
  a ready-counter or output ownership contract.
- Opt in via the existing backward pack_gqa argument. Preserve the default
  unpacked behavior and all existing tensor shapes/dtypes at API boundaries.

## Validation

Use an independent PyTorch FP32 attention/autograd reference with explicit
shared-head masks, including mixed partial/full blocks, empty KV columns,
fully masked query rows, short and long Q, GQA and MQA, and G=1.
Compare dQ/dK/dV with numerical tolerances appropriate to BF16 intermediates;
test repeat invocations and CUDA graph replay for bitwise equality.
Check caller-provided outputs and the public autograd path.
Run compute-sanitizer on a small sparse case.

## Benchmark

Primary matrix: B=1, Hq=64, Hkv=8, D=128, BF16, Q in {512,1024},
KV in {8192,32768,65536}; include Q=256/2048 as optional extensions.
Measure dense-in-sparse and shared sparse patterns (including partial blocks).
Report actual active block density and per-KV Q-loop lengths, not just nominal
sequence length. Compare pack_gqa=False/True on identical inputs and metadata.

Warm up JIT compilation and verify correctness before timing. Report both the
main backward kernel CUDA time and total backward GPU time (preprocess,
mainloop, dQ postprocess, and any baseline dKV postprocess/zeroing). Use CUDA
graph replay to reduce launch noise and alternate comparison order across
samples. Record hardware, software versions, base commit, working diff hash,
shape, mask, CTA mode, repetition count, median, spread, and speedup. Main-kernel timing uses external CUDA event nodes
in a separate graph; total timing uses an uninstrumented graph. No speedup
claim is made until measurements exist. Keep raw profiles outside git.

## Status

Implemented and validated on GB200 (SM100). The head-major TMA view specializes
sequence extents and input/output strides in the compilation cache. See the
[validation and performance report](../reports/deterministic_sparse_pack_gqa_gb200.md)
for the measured 12-case matrix, correctness coverage, and remaining limits.

## Paired 2CTA extension

The paired path retains the production mirrored union CSR, original visibility
bitset and pair-ordered dQ tickets. Contract each KV-head group's work map to
one cluster: keep rows whose Q head is divisible by G and divide that head by G.
Do not take every G-th row: local-phase scheduling need not place group heads
next to one another. CSR and ticket heads contract from Hq to Hkv only after
checking equality within each group. Original visibility bits remain Hq-indexed
because mask callbacks receive the original Q head.

Both CTAs traverse the identical G-times-longer head-major Q loop. Each owns
its original 128-token KV half; their existing cross-CTA dS exchange and split
dQ reduction remain unchanged. Paired dQ tickets advance once per cluster,
with independent state for each original Q head. The contracted work ordering
must preserve every original head's KV-pair order.

The initial paired stage outputs direct BF16 dK/dV, without an external FP32
accumulator or dKV postprocess. The producer-ready extension below adds the
optional local completion signal; transport integration remains outside FA4.

Validation covers shared-head partial/full masks, disjoint pair neighbors,
odd-K padding, inactive halves, deterministic replay, and the same CP128/256
production-dispatch benchmark with paired pack disabled/enabled.


## Standalone benchmark CTA selection

`benchmark_sparse_pack_gqa_bwd.py` defaults to paired 2CTA; `--cta-group-size 1`
retains the original comparison. Set the upstream environment flag before FA4
imports rather than mutating the cached private setting after import.

The shared synthetic fixture builds a coarse (Q256,KV256) mask with the original
element predicate. Its transposed CSR and pair-level dQ tickets are mirrored to
the two physical KV128 rows. CSR/tickets remain head-broadcast; work maps and
original-active flags use Hq or Hkv for unpacked/packed execution. Pair-major,
ascending-head order preserves dQ and dKV deterministic order. The predicate
masks holes introduced by the pair union, so this fixture needs neither the
MegaAttention planner nor its original-visibility callback wrapper. Report both
original and executed tile density; pair padding must not inflate original density.


## Packed BF16 producer-ready signal

The packed 1CTA and paired 2CTA paths optionally accept `dkv_done_counter`.
It is caller-owned contiguous CUDA int32: either `[B*Hkv*N]`, or `[Hkv,capacity]`
for B=1 with capacity >= N, where N=ceil(Sk/128). The second layout preserves
its physical head stride. Reset it to zero before every invocation/replay.

One active physical KV tile publishes one system-release increment after its
complete BF16 dV and scaled dK outputs are visible. All issuing compute
warpgroups drain their bulk store groups with a full wait (not `.read`), then
synchronize before the single publisher. No last-Q-head test is used: packed
head indices already name KV heads. A paired CTA publishes only its own valid
physical half; original-active metadata suppresses inactive halves. Empty CSR
rows and the padded odd-K mate publish nothing. The consumer's expected array
is the original per-tile activity (0/1), not the union CSR activity or G.
An expected-zero consumer must synthesize zero or skip the contribution;
it must not read the output before compute completion merely because no signal
is expected. Counters describe local partial readiness, not remote delivery.

Consumers acquire the counter and apply their required proxy fence before
reading the caller's BF16 buffers. Communication can use a head-major strided
BSHD output view directly. External FP32 accumulators, multicast completion,
Q-head finalizer queues and `skip_dkv_postprocess` remain unsupported in packed
mode. Omitting the counter preserves the compute-only specialization.

Validate by running an independent, bounded polling consumer on another CUDA
stream. Snapshot dK and dV immediately after acquire and compare bitwise against
the final BF16 output, plus an independent FP32 reference. Cover both CTA modes,
all KV heads, inactive/empty tiles, odd pairs, nontrivial counter head stride,
head-major output views, resets, and CUDA Graph replay. A pre-published signal
with poisoned output must be detected by the same consumer harness.
