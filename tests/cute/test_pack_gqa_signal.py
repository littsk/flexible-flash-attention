"""Packed BF16 producer readiness, observed concurrently without a producer event."""

import pytest
import torch
import triton
import triton.language as tl
from pack_gqa_utils import make_case, reference_gradients

from flash_attn.cute import utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="packed producer requires Blackwell",
)


@triton.jit
def consume_tiles(
    DK,
    DV,
    DONE,
    ACTIVE,
    SNAP_K,
    SNAP_V,
    STATUS,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    CAP: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
):
    offsets = tl.arange(0, 2048)
    for unit in range(tl.program_id(0), H * N, 4):
        head, tile = unit // N, unit % N
        if tl.load(ACTIVE + unit) != 0:
            begin = tl.inline_asm_elementwise(
                "mov.u64 $0, %clock64;", "=l", [], tl.uint64, False, 1
            )
            ready = tl.full((), 0, tl.int32)
            elapsed = tl.full((), 0, tl.uint64)
            while (ready == 0) & (elapsed < 3_000_000_000):
                ready = tl.inline_asm_elementwise(
                    "ld.acquire.sys.global.b32 $0, [$1];",
                    "=r,l",
                    [DONE + head * CAP + tile],
                    tl.int32,
                    False,
                    1,
                )
                now = tl.inline_asm_elementwise(
                    "mov.u64 $0, %clock64;", "=l", [], tl.uint64, False, 1
                )
                elapsed = now - begin
            tl.store(STATUS + unit, ready)
            if ready != 0:
                for chunk in range(tl.cdiv(128 * D, 2048)):
                    index = chunk * 2048 + offsets
                    row, col = tile * 128 + index // D, index % D
                    valid = (index < 128 * D) & (row < S)
                    source = row * ST + head * SH + col
                    target = (row * H + head) * D + col
                    tl.store(SNAP_K + target, tl.load(DK + source, valid, 0), valid)
                    tl.store(SNAP_V + target, tl.load(DV + source, valid, 0), valid)


@pytest.mark.parametrize(
    "cta,sk,pattern,head_major,padding",
    [
        (1, 1153, "mixed", False, 0),
        (2, 1152, "mixed", True, 7),
        (2, 128, "mixed", False, 3),
        (2, 384, "empty", True, 5),
    ],
)
def test_packed_bf16_ready(monkeypatch, cta, sk, pattern, head_major, padding):
    monkeypatch.setattr(utils, "_fa_disable_2cta_enabled", cta == 1)
    case = make_case(sq=512, sk=sk, cta_group_size=cta, pattern=pattern)
    heads, dim = case.k.shape[2:]
    n = triton.cdiv(sk, 128)
    capacity = n + padding
    if head_major:
        storage_k = torch.empty(
            (1, heads, sk + 256, dim), device="cuda", dtype=torch.bfloat16
        )
        storage_v = torch.empty_like(storage_k)
        dk = storage_k[:, :, :sk].transpose(1, 2)
        dv = storage_v[:, :, :sk].transpose(1, 2)
    else:
        dk, dv = torch.empty_like(case.k), torch.empty_like(case.v)
    outputs = (torch.empty_like(case.q), dk, dv)
    done = torch.zeros((heads, capacity), device="cuda", dtype=torch.int32)
    metadata = case.packed_bwd_sparse if cta == 2 else case.bwd_sparse
    if cta == 2:
        active = metadata.bwd_original_active[0, :, :n].contiguous()
    else:
        active = ((metadata.mask_block_cnt + metadata.full_block_cnt)[0, :, :n] > 0).to(
            torch.int32
        )
        active = active.expand(heads, n).contiguous()
    snapshot = (torch.empty_like(case.k), torch.empty_like(case.v))
    status = torch.zeros_like(active)
    kwargs = {
        "dq": outputs[0],
        "dk": dk,
        "dv": dv,
        "dkv_done_counter": done if padding else done.flatten(),
    }

    def backward():
        return case.backward(True, **kwargs)

    def consume():
        consume_tiles[(4,)](
            dk,
            dv,
            done,
            active,
            *snapshot,
            status,
            sk,
            heads,
            dim,
            n,
            capacity,
            dk.stride(1),
            dk.stride(2),
        )

    backward()
    high = reference_gradients(case)
    low = reference_gradients(case, dtype=torch.bfloat16)
    for actual, hi, lo in zip(outputs, high, low):
        intrinsic = (lo.float() - hi).abs().max().item()
        assert (actual.float() - hi).abs().max().item() <= 2 * intrinsic + 0.003
        torch.testing.assert_close(actual.float(), hi, atol=0.02, rtol=0.04)
        assert torch.isfinite(actual).all()
    expected = tuple(t.clone() for t in outputs)
    torch.testing.assert_close(done[:, :n], active, rtol=0, atol=0)
    assert not done[:, n:].any()
    # Compile consumer, and verify that publishing before stores exposes poison.
    done.fill_(1)
    dk.fill_(float("nan"))
    dv.fill_(float("nan"))
    consume()
    valid = (
        active.repeat_interleave(128, dim=1)[:, :sk]
        .T[None, :, :, None]
        .expand_as(dk)
        .bool()
    )
    if valid.any():
        assert torch.isnan(snapshot[0][valid]).all()
        assert torch.isnan(snapshot[1][valid]).all()
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())

    def run():
        done.zero_()
        status.zero_()
        dk.fill_(float("nan"))
        dv.fill_(float("nan"))
        consumer.wait_stream(producer)
        with torch.cuda.stream(consumer):
            consume()
        backward()
        producer.wait_stream(consumer)

    for _ in range(10):
        with torch.cuda.stream(producer):
            run()
        producer.synchronize()
        assert torch.equal(status, active), (
            "consumer timed out or saw duplicate publication"
        )
        assert torch.equal(done[:, :n], active) and not done[:, n:].any()
        assert all(torch.equal(a, b) for a, b in zip(outputs, expected))
        for observed, final in zip(snapshot, outputs[1:]):
            assert torch.equal(observed[valid], final[valid]), (
                "ready preceded complete BF16 stores"
            )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=producer):
        run()
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(status, active)
        assert torch.equal(done[:, :n], active) and not done[:, n:].any()
        assert all(torch.equal(a, b) for a, b in zip(outputs, expected))
        for observed, final in zip(snapshot, outputs[1:]):
            assert torch.equal(observed[valid], final[valid])


def test_packed_signal_flat_batch_and_validation(monkeypatch):
    monkeypatch.setattr(utils, "_fa_disable_2cta_enabled", True)
    case = make_case(sq=256, sk=256, hq=8, hkv=2, dim=64, batch=2)
    done = torch.zeros(8, device="cuda", dtype=torch.int32)
    result = case.backward(True, dkv_done_counter=done)
    assert torch.equal(
        done.view(2, 2, 2),
        (case.bwd_sparse.mask_block_cnt + case.bwd_sparse.full_block_cnt > 0)
        .expand(2, 2, 2)
        .to(torch.int32),
    )
    high = reference_gradients(case)
    for actual, expected in zip(result, high):
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.04)
    for invalid in (done.float(), done[:1], done.view(2, 4)):
        with pytest.raises((ValueError, AssertionError)):
            case.backward(True, dkv_done_counter=invalid)
