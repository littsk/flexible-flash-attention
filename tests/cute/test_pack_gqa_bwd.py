"""GB200 correctness and repeatability for native head-major PackGQA backward.

Run with FA_DISABLE_2CTA=1; no tensor transpose is used by the implementation.
"""

import pytest
import torch
from pack_gqa_utils import check_gradients, make_case, reference_gradients

from flash_attn.cute import utils
from flash_attn.cute.interface import flash_attn_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="PackGQA backward requires SM100/SM110",
)


@pytest.fixture(autouse=True)
def one_cta(monkeypatch):
    monkeypatch.setattr(utils, "_fa_disable_2cta_enabled", True)


@pytest.mark.parametrize(
    "sq,sk,hq,hkv,dim,batch",
    [
        (256, 1024, 8, 1, 64, 1),
        (512, 2048, 64, 8, 128, 1),
        (1024, 4096, 64, 8, 128, 1),
        (512, 1025, 8, 2, 128, 2),
        (256, 1024, 8, 8, 128, 1),
    ],
)
@pytest.mark.parametrize("pattern", ["mixed", "sparse", "empty"])
def test_pack_gqa_gradients(sq, sk, hq, hkv, dim, batch, pattern):
    case = make_case(sq, sk, hq, hkv, dim, pattern, batch)
    ref = reference_gradients(case)
    for pack in (False, True):
        got = case.backward(pack)
        check_gradients(got, ref)
        for _ in range(3):
            repeated = case.backward(pack)
            assert all(torch.equal(x, y) for x, y in zip(got, repeated))


@pytest.mark.parametrize("spt", [False, True])
@pytest.mark.parametrize("kv_block", [128, 256])
def test_pack_gqa_graph_and_outputs(spt, kv_block):
    case = make_case(sq=512, sk=1153, hq=16, hkv=2, spt=spt, kv_block=kv_block)
    outputs = (
        torch.empty_like(case.q),
        torch.empty_like(case.k),
        torch.empty_like(case.v),
    )
    kwargs = {"dq": outputs[0], "dk": outputs[1], "dv": outputs[2]}
    case.backward(True, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = case.backward(True, **kwargs)
    assert all(x.data_ptr() == y.data_ptr() for x, y in zip(outputs, returned))
    graph.replay()
    expected = tuple(t.clone() for t in outputs)
    for _ in range(5):
        graph.replay()
        assert all(torch.equal(x, y) for x, y in zip(expected, outputs))
    check_gradients(outputs, reference_gradients(case))


def test_pack_gqa_public_autograd():
    case = make_case(sq=512, sk=1024, hq=16, hkv=2)
    q, k, v = (t.detach().requires_grad_() for t in (case.q, case.k, case.v))
    out, _ = flash_attn_func(
        q,
        k,
        v,
        pack_gqa=True,
        deterministic=True,
        mask_mod=case.cute_mask,
        block_sparse_tensors=case.fwd_sparse,
        block_sparse_tensors_bwd=case.bwd_sparse,
    )
    got = torch.autograd.grad(out, (q, k, v), case.dout)
    check_gradients(got, reference_gradients(case))


def test_pack_gqa_rejects_per_head_metadata():
    case = make_case(sq=256, sk=512, hq=8, hkv=1)
    case.bwd_sparse = case.bwd_sparse._replace(
        mask_block_cnt=case.bwd_sparse.mask_block_cnt.expand(1, 8, -1),
    )
    with pytest.raises(ValueError, match="shared-head metadata"):
        case.backward(True)


def test_pack_gqa_rejects_external_accumulators():
    case = make_case(sq=256, sk=512, hq=8, hkv=1)
    with pytest.raises(ValueError, match="external ring"):
        case.backward(True, dk_accum_external=torch.empty(1, device="cuda"))


@pytest.mark.parametrize("pattern", ["rows", "causal", "dense"])
def test_pack_gqa_mask_coordinates_and_optional_full_list(pattern):
    case = make_case(sq=512, sk=1537, hq=16, hkv=2, pattern=pattern)
    if pattern == "rows":
        # No fully visible coarse block: exercise the optional full-list path.
        assert not case.bwd_sparse.full_block_cnt.any()
        case.bwd_sparse = case.bwd_sparse._replace(
            full_block_cnt=None,
            full_block_idx=None,
            dq_write_order_full=None,
        )
    case.bwd_sparse = case.bwd_sparse._replace(
        kv_block_signal=torch.ones(
            2, (1537 + 127) // 128, dtype=torch.int32, device="cuda"
        ),
    )
    reference = reference_gradients(case)
    for pack in (False, True):
        got = case.backward(pack)
        check_gradients(got, reference)
        repeated = case.backward(pack)
        assert all(torch.equal(x, y) for x, y in zip(got, repeated))
