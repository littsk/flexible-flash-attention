"""Compare deterministic native PackGQA and unpacked sparse backward on GB200.

FA_DISABLE_2CTA=1 PYTHONPATH=. python benchmarks/benchmark_sparse_pack_gqa_bwd.py \
    --output /tmp/pack-gqa-benchmark.json
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "cute"))
from pack_gqa_utils import check_gradients, make_case, reference_gradients

from flash_attn.cute import utils
from flash_attn.cute.interface import _flash_attn_bwd


def capture(case, pack: bool, calls: int) -> torch.cuda.CUDAGraph:
    outputs = tuple(torch.empty_like(t) for t in (case.q, case.k, case.v))
    kwargs = {"dq": outputs[0], "dk": outputs[1], "dv": outputs[2]}
    for _ in range(3):
        case.backward(pack, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            case.backward(pack, **kwargs)
    # Keep captured output storage alive for all graph replays.
    graph.outputs = outputs
    return graph


def time_graph(graph: torch.cuda.CUDAGraph, calls: int) -> float:
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / calls


def main_kernel_times(case, pack: bool, calls: int) -> list[float]:
    """External CUDA event nodes bracket only the compiled main kernel.

    The normal graph used for total timing contains no instrumentation. Keep
    preprocess/reset and postprocess in this separate graph so replay resets
    all deterministic semaphores before each measured invocation.
    """
    events = []

    def instrument(kernel):
        def measured(*args):
            start, end = (
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)
            )
            events.append((start, end))
            start.record()
            result = kernel(*args)
            end.record()
            return result

        return measured

    cache = _flash_attn_bwd.compile_cache.cache
    graph = torch.cuda.CUDAGraph()
    with (
        patch.dict(cache, {key: instrument(fn) for key, fn in cache.items()}),
        torch.cuda.graph(graph),
    ):
        for _ in range(calls):
            case.backward(pack)
    if len(events) != calls:
        raise RuntimeError(
            f"Expected {calls} main backward launches, got {len(events)}"
        )
    samples = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        samples.extend(start.elapsed_time(end) * 1000 for start, end in events)
    return samples


def summary(samples: list[float]) -> dict:
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--kv", nargs="+", type=int, default=[8192, 32768, 65536])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["dense", "sparse", "mixed"],
        default=["dense", "mixed"],
    )
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3 or args.calls < 1:
        parser.error("at least 3 samples and 1 call are required")
    if torch.cuda.get_device_capability()[0] not in (10, 11):
        raise RuntimeError("SM100/SM110 is required")
    utils._fa_disable_2cta_enabled = True
    repo = Path(__file__).resolve().parents[1]
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    props = torch.cuda.get_device_properties(0)
    report = {
        "gpu": props.name,
        "capability": torch.cuda.get_device_capability(),
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("nvidia-cutlass-dsl", "quack-kernels", "apache-tvm-ffi")
        },
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "hq": 64,
        "hkv": 8,
        "batch": 1,
        "dim": 128,
        "dtype": "bfloat16",
        "deterministic": True,
        "cta_mode": "1CTA",
        "calls_per_graph": args.calls,
        "samples": args.samples,
        "main_kernel_method": "external CUDA event nodes bracketing the compiled main kernel in a separate graph",
        "total_method": "CUDA events around graph replay; all backward kernels included",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sq in args.q:
        for sk in args.kv:
            for pattern in args.patterns:
                print(f"Q={sq} KV={sk} {pattern}: validating", flush=True)
                case = make_case(sq, sk, pattern=pattern)
                ref = reference_gradients(case)
                errors = {}
                for pack in (False, True):
                    result = case.backward(pack)
                    errors[str(pack)] = check_gradients(result, ref)
                    repeated = case.backward(pack)
                    if not all(torch.equal(x, y) for x, y in zip(result, repeated)):
                        raise AssertionError(f"Non-deterministic backward: pack={pack}")
                del ref, result, repeated
                graphs = {
                    pack: capture(case, pack, args.calls) for pack in (False, True)
                }
                for graph in graphs.values():
                    for _ in range(5):
                        graph.replay()
                torch.cuda.synchronize()
                totals = {False: [], True: []}
                kernels = {False: [], True: []}
                for i in range(args.samples):
                    order = (False, True) if i % 2 == 0 else (True, False)
                    for pack in order:
                        totals[pack].append(time_graph(graphs[pack], args.calls))
                for pack in (False, True):
                    kernels[pack] = main_kernel_times(case, pack, args.calls)
                counts = case.bwd_sparse.mask_block_cnt + case.bwd_sparse.full_block_cnt
                row = {
                    "q": sq,
                    "kv": sk,
                    "pattern": pattern,
                    "correctness_max_abs": errors,
                    "repeat_bitwise": True,
                    "block_density": counts.sum().item()
                    / (counts.numel() * (sq // 256)),
                    "unpacked_q_loop_min": int(counts.min()) * 2,
                    "unpacked_q_loop_max": int(counts.max()) * 2,
                    "unpacked_total": summary(totals[False]),
                    "packed_total": summary(totals[True]),
                    "unpacked_main": summary(kernels[False]),
                    "packed_main": summary(kernels[True]),
                }
                row["total_speedup"] = (
                    row["unpacked_total"]["median_us"]
                    / row["packed_total"]["median_us"]
                )
                row["main_speedup"] = (
                    row["unpacked_main"]["median_us"] / row["packed_main"]["median_us"]
                )
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"  main {row['main_speedup']:.3f}x, total {row['total_speedup']:.3f}x; "
                    f"total {row['unpacked_total']['median_us']:.1f} -> {row['packed_total']['median_us']:.1f} us",
                    flush=True,
                )
                del graphs, case
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
