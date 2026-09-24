# Paired 2CTA PackGQA: Q512 / Q1024 / Q2048 on GB200

Both variants use paired 2CTA. B=1, Hq=64, Hkv=8, D=128, BF16, deterministic.
KV lengths are 8192, 32768 and 65536. `dense` is fully visible attention represented
as block-sparse metadata; `mixed` combines full, partially masked and absent tiles.
These are standalone synthetic cases, not CP-dispatch replays. Q is the actual
per-invocation query length.

Q512, Q1024 and Q2048 ran independently on GB200 GPUs 0, 1 and 2. Each pack/no-pack
comparison ran on the same GPU with identical input tensors and masks. No communication
or overlap was launched; GPU clocks were not locked.

## Timing and correctness

Nine alternating samples of ten calls per CUDA graph; report medians. The full
backward graph includes initialization and all pre/main/postprocessing, excluding
forward, metadata construction, reference evaluation and compilation. Main-kernel
durations use a separate graph with external CUDA event nodes (50 samples). Both
graphs reuse the same caller-provided output buffers. These are independently sampled
medians; do not subtract them to estimate preprocessing/postprocessing overhead.

All 18 cases passed independent FP32 gradient checks (atol=0.015, rtol=0.04),
finite BF16 output checks and bitwise eager repeatability before timing.
Maximum sample coefficient of variation across both timing methods: 2.22%.

## Results

All times are **milliseconds per single backward invocation on one GPU**.

| Q | KV | Mask | Main no pack | Main pack | Main speedup | Full no pack | Full pack | Full speedup |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
|512|8192|dense|0.7475|0.3731|2.004x|0.7888|0.3817|2.066x|
|512|8192|mixed|0.9217|0.5431|1.697x|0.9633|0.5534|1.741x|
|512|32768|dense|2.9055|1.2634|2.300x|3.0237|1.2871|2.349x|
|512|32768|mixed|3.5785|1.8709|1.913x|3.6901|1.8815|1.961x|
|512|65536|dense|5.7928|2.4712|2.344x|6.0126|2.4835|2.421x|
|512|65536|mixed|7.1200|3.6041|1.975x|7.3221|3.6144|2.026x|
|1024|8192|dense|1.0527|0.7141|1.474x|1.1041|0.7332|1.506x|
|1024|8192|mixed|1.3787|1.0547|1.307x|1.4327|1.0769|1.330x|
|1024|32768|dense|4.1128|2.5018|1.644x|4.2477|2.5254|1.682x|
|1024|32768|mixed|5.3565|3.6407|1.471x|5.4796|3.6554|1.499x|
|1024|65536|dense|8.2497|5.0753|1.625x|8.5361|4.9208|1.735x|
|1024|65536|mixed|10.6685|7.0138|1.521x|10.8990|7.0263|1.551x|
|2048|8192|dense|1.8545|1.4376|1.290x|1.9288|1.4656|1.316x|
|2048|8192|mixed|2.3200|2.0599|1.126x|2.3973|2.1021|1.140x|
|2048|32768|dense|7.3042|5.1103|1.429x|7.5558|5.1009|1.481x|
|2048|32768|mixed|9.0754|7.1845|1.263x|9.2216|7.2264|1.276x|
|2048|65536|dense|14.5971|10.1594|1.437x|14.9894|9.9502|1.506x|
|2048|65536|mixed|18.0910|13.8555|1.306x|18.3237|13.9024|1.318x|

## Interpretation

- Q=512: main speedup 1.697x–2.344x; full backward 1.741x–2.421x.
- Q=1024: main speedup 1.307x–1.644x; full backward 1.330x–1.735x.
- Q=2048: main speedup 1.126x–1.437x; full backward 1.140x–1.506x.

The gain decreases as Q grows. Dense-case Q loops grow from 4 to 32 iterations
at Q512, 8 to 64 at Q1024, and 16 to 128 at Q2048. This is consistent with
shorter original loops benefiting more from amortized pipeline/cluster overhead,
but does not isolate that effect from KV reuse and eliminating cross-head dKV
accumulation and synchronization.

## Reproduction

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python benchmarks/benchmark_sparse_pack_gqa_bwd.py \
  --cta-group-size 2 --q 512 1024 2048 --kv 8192 32768 65536 \
  --patterns dense mixed --samples 9 --calls 10 --output /tmp/paired-q-sweep.json
```

Base commit: `73befea349c2e8292730fe910eca7cf28b543609`; working diff SHA256:
`dc6563956eaeab76bd25ca9bdb79a5343db7123328272cf6bf8fbb8a233a01f0`.
Software: torch 2.9.1, CUDA 13.1, packages `{'nvidia-cutlass-dsl': '4.8.0', 'quack-kernels': '0.6.5', 'apache-tvm-ffi': '0.1.14.post0'}`.
Raw JSON, logs, source snapshot and patch are under
`agent_space/paired_q_sweep_20260924/shared_outputs/` (not committed).
The parent directory retains the initial measurement before output-buffer reuse
was aligned between the two timing graphs; only the corrected run is tabulated here.
