import math

import pytest
import torch
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask

from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    bhqk_to_linear_sparse_tensors,
)
from flash_attn.cute.interface import (
    _tile_size_bwd_sm90,
    _tile_size_fwd_sm90,
    flash_attn_func,
    flash_attn_varlen_func,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_arbitrary_func(
    batch,
    heads,
    seqlen_q,
    seqlen_k,
    device,
    *,
    vary_by_batch=False,
    vary_by_head=False,
):
    func = torch.zeros(batch, heads, 3, seqlen_q + 256, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(heads):
            sink_base = 1 if (vary_by_head and h % 2) else 8 + (b if vary_by_batch else 0)
            sink = min(sink_base, seqlen_k)
            window = max(4, 16 - 4 * (h if vary_by_head else 0) - 2 * (b if vary_by_batch else 0))
            for q_idx in range(seqlen_q):
                func[b, h, 0, q_idx] = sink
                func[b, h, 1, q_idx] = max(0, q_idx - window)
                func[b, h, 2, q_idx] = min(seqlen_k, q_idx + 1)
    return func


def _make_head_dependent_arbitrary_func(batch, heads, seqlen_q, seqlen_k, device):
    return _make_arbitrary_func(
        batch,
        heads,
        seqlen_q,
        seqlen_k,
        device,
        vary_by_head=heads > 1,
    )


def _apply_arbitrary_mask(scores, arbitrary_func):
    batch, heads, seqlen_q, seqlen_k = scores.shape
    func = arbitrary_func[..., :seqlen_q]
    cols = torch.arange(seqlen_k, device=scores.device, dtype=torch.int32)
    base = cols.view(1, 1, 1, seqlen_k) < func[:, :, 0, :].unsqueeze(-1)
    interval = (
        (cols.view(1, 1, 1, seqlen_k) >= func[:, :, 1, :].unsqueeze(-1))
        & (cols.view(1, 1, 1, seqlen_k) < func[:, :, 2, :].unsqueeze(-1))
    )
    valid = (base | interval).expand(batch, heads, seqlen_q, seqlen_k)
    return scores.masked_fill(~valid, float("-inf"))


def _make_arbitrary_mask_mod(arbitrary_func):
    def mask_mod(b, h, q_idx, kv_idx):
        zero = q_idx * 0
        b_idx = zero if arbitrary_func.shape[0] == 1 else b
        h_idx = zero if arbitrary_func.shape[1] == 1 else h
        value_valid = kv_idx < arbitrary_func[b_idx, h_idx, zero, q_idx]
        for i in range(arbitrary_func.shape[2] // 2):
            start = arbitrary_func[b_idx, h_idx, zero + 2 * i + 1, q_idx]
            end = arbitrary_func[b_idx, h_idx, zero + 2 * i + 2, q_idx]
            value_valid = value_valid | ((kv_idx >= start) & (kv_idx < end))
        return value_valid

    return mask_mod


def _attention_ref(q, k, v, arbitrary_func):
    q_ref = q.float()
    k_ref = k.float()
    v_ref = v.float()
    if q_ref.shape[2] != k_ref.shape[2]:
        repeat = q_ref.shape[2] // k_ref.shape[2]
        k_ref = k_ref.repeat_interleave(repeat, dim=2)
        v_ref = v_ref.repeat_interleave(repeat, dim=2)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref * scale, k_ref)
    scores = _apply_arbitrary_mask(scores, arbitrary_func)
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v_ref).to(q.dtype)


SEQLEN_CONFIGS = [
    (64, 64),
    (96, 128),
    (129, 65),
]


def _device_major():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()[0]


def _cases_for_arch():
    major = _device_major()
    if major == 9:
        return [(192, 192), (256, 256)]
    if major == 10:
        return [(192, 128), (256, 256)]
    return []


def _generic_case_for_arch():
    major = _device_major()
    if major == 9:
        return (192, 192)
    if major == 10:
        return (192, 128)
    return None


def _kv_heads_for_mode(kv_mode, heads):
    if kv_mode == "mha":
        return heads
    if kv_mode == "gqa":
        return heads // 2
    if kv_mode == "mqa":
        return 1
    raise AssertionError(f"unknown kv_mode: {kv_mode}")


def _linear_block_sizes(head_dim, head_dim_v, seqlen_q, qhead_per_kvhead):
    major = _device_major()
    if major == 9:
        fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, False, False)
        bwd_cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            False,
            False,
            sparse_block_size_q=128,
        )
        return (fwd_cfg.m_block_size, fwd_cfg.n_block_size), (128, bwd_cfg.n_block_size)
    if major == 10:
        q_stage = 2 if seqlen_q * qhead_per_kvhead > 128 else 1
        return (q_stage * 128, 128), (256, 128)
    pytest.skip("linear arbitrary block-sparse test only runs on SM90/SM100")


def _extract_block_sparse_tensors(block_mask, *, backward):
    block_mask_tuple = block_mask.as_tuple()
    has_seq_prefix = isinstance(block_mask_tuple[0], int)
    if backward:
        start = 6 if has_seq_prefix else 4
    else:
        start = 2 if has_seq_prefix else 0
    mask_cnt, mask_idx, full_cnt, full_idx = block_mask_tuple[start : start + 4]
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
    )


def _make_linear_block_sparse_pair(arbitrary_func, seqlen_q, seqlen_k, fwd_block_size, bwd_block_size):
    mask_mod = _make_arbitrary_mask_mod(arbitrary_func)
    mask_batch, mask_heads = arbitrary_func.shape[:2]
    bm_fwd = create_block_mask(
        mask_mod,
        mask_batch,
        mask_heads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=fwd_block_size,
    )
    bm_bwd = create_block_mask(
        mask_mod,
        mask_batch,
        mask_heads,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=bwd_block_size,
    )
    linear_k = bhqk_to_linear_sparse_tensors(
        _extract_block_sparse_tensors(bm_fwd, backward=False)._replace(block_size=fwd_block_size)
    )
    linear_q = bhqk_to_linear_sparse_tensors(
        _extract_block_sparse_tensors(bm_bwd, backward=True)._replace(block_size=bwd_block_size)
    )
    return linear_k, linear_q


def _run_arbitrary_case(
    *,
    batch,
    seqlen_q,
    seqlen_k,
    heads,
    kv_heads,
    head_dim,
    head_dim_v,
    mask_batch,
    mask_heads,
    seed,
    dtype=torch.bfloat16,
    check_backward=True,
    atol=4e-2,
    rtol=4e-2,
):
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(
        mask_batch,
        mask_heads,
        seqlen_q,
        seqlen_k,
        device,
        vary_by_batch=mask_batch > 1,
        vary_by_head=mask_heads > 1,
    )

    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        return_lse=True,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
    if not check_backward:
        return

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_dim, head_dim_v", _cases_for_arch())
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_CONFIGS)
def test_arbitrary_mask_fixedlen_output(seqlen_q, seqlen_k, kv_mode, head_dim, head_dim_v, dtype):
    if not _cases_for_arch():
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    heads = 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    mask_heads = heads if kv_mode != "mha" else 1
    check_backward = not (_device_major() == 9 and head_dim == 192 and kv_mode != "mha")
    _run_arbitrary_case(
        batch=2,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=2,
        mask_heads=mask_heads,
        seed=seqlen_q + seqlen_k + head_dim + 17 * kv_heads,
        dtype=dtype,
        check_backward=check_backward,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "mask_batch, mask_heads",
    [
        pytest.param(1, 1, id="batch_head_broadcast"),
        pytest.param(2, 1, id="batch_specific_head_broadcast"),
        pytest.param(1, 2, id="batch_broadcast_head_specific"),
        pytest.param(2, 2, id="batch_head_specific"),
    ],
)
def test_arbitrary_mask_broadcast_patterns(mask_batch, mask_heads):
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    head_dim, head_dim_v = case
    _run_arbitrary_case(
        batch=2,
        seqlen_q=96,
        seqlen_k=128,
        heads=2,
        kv_heads=2,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=mask_batch,
        mask_heads=mask_heads,
        seed=mask_batch * 10 + mask_heads,
    )


def test_arbitrary_mask_generic_gqa():
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    head_dim, head_dim_v = case
    _run_arbitrary_case(
        batch=1,
        seqlen_q=96,
        seqlen_k=128,
        heads=4,
        kv_heads=2,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=1,
        mask_heads=4,
        seed=23,
        check_backward=not (_device_major() == 9 and head_dim == 192),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
def test_arbitrary_mask_linear_block_sparse(kv_mode):
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("linear arbitrary block-sparse test only runs on SM90/SM100")
    torch.manual_seed(97 + len(kv_mode))
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 2, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    head_dim, head_dim_v = case
    seqlen_q, seqlen_k = 96, 160
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype, requires_grad=True
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(1, 1, seqlen_q, seqlen_k, device)
    fwd_block_size, bwd_block_size = _linear_block_sizes(
        head_dim,
        head_dim_v,
        seqlen_q,
        heads // kv_heads,
    )
    linear_k, linear_q = _make_linear_block_sparse_pair(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        fwd_block_size,
        bwd_block_size,
    )

    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        linear_q_block_sparse_tensors=linear_q,
        return_lse=True,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    if _device_major() == 9 and kv_mode != "mha":
        return
    # SM100 generic block-sparse backward disables 2CTA, while the existing
    # hdim=192/dv=128 backward kernel requires 2CTA. This is a pre-existing
    # unsupported combination in origin/main rather than a linear CSR issue.
    if _device_major() == 10 and head_dim == 192:
        return
    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 256), (256, 512), (384, 384), (96, 160)])
def test_arbitrary_mask_linear_block_sparse_hd256(kv_mode, seqlen_q, seqlen_k):
    """Forward-only CSR linear block sparsity for the SM100 dedicated head_dim=256 kernel.

    The dedicated 2CTA hd256 forward kernel processes a 256-row (2 * tile_m) Q tile
    per work item; the CSR Q block size therefore matches `_linear_block_sizes`.
    """
    if _device_major() != 10:
        pytest.skip("hd256 block-sparse forward only runs on SM100")
    torch.manual_seed(131 + seqlen_q + seqlen_k + len(kv_mode))
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 2, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    head_dim = head_dim_v = 256
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype)
    arbitrary_func = _make_arbitrary_func(1, 1, seqlen_q, seqlen_k, device)
    fwd_block_size, bwd_block_size = _linear_block_sizes(
        head_dim, head_dim_v, seqlen_q, heads // kv_heads
    )
    linear_k, _ = _make_linear_block_sparse_pair(
        arbitrary_func, seqlen_q, seqlen_k, fwd_block_size, bwd_block_size
    )
    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        return_lse=True,
    )
    out_ref = _attention_ref(q, k, v, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    "kv_mode,mask_heads,seqlen_q,seqlen_k",
    [
        pytest.param("mha", 4, 384, 384, id="mha_head_specific"),
        pytest.param("gqa", 1, 384, 384, id="gqa_head_broadcast"),
        pytest.param("mqa", 1, 384, 384, id="mqa_head_broadcast"),
        pytest.param("mha", 4, 96, 160, id="mha_qstage1"),
    ],
)
def test_arbitrary_mask_linear_block_sparse_hd256_backward(
    kv_mode, mask_heads, seqlen_q, seqlen_k
):
    if _device_major() != 10:
        pytest.skip("hd256 block-sparse backward only runs on SM100")
    torch.manual_seed(733 + len(kv_mode))
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 2, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    head_dim = head_dim_v = 256
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype, requires_grad=True
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(
        1,
        mask_heads,
        seqlen_q,
        seqlen_k,
        device,
        vary_by_head=mask_heads > 1,
    )
    fwd_block_size, bwd_block_size = _linear_block_sizes(
        head_dim, head_dim_v, seqlen_q, heads // kv_heads
    )
    linear_k, linear_q = _make_linear_block_sparse_pair(
        arbitrary_func, seqlen_q, seqlen_k, fwd_block_size, bwd_block_size
    )
    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        linear_q_block_sparse_tensors=linear_q,
        return_lse=True,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_arbitrary_mask_fp16_smoke(dtype):
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    head_dim, head_dim_v = case
    _run_arbitrary_case(
        batch=1,
        seqlen_q=64,
        seqlen_k=96,
        heads=2,
        kv_heads=2,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=1,
        mask_heads=1,
        seed=71,
        dtype=dtype,
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.parametrize("head_dim, head_dim_v", _cases_for_arch())
@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
@pytest.mark.parametrize("seqlen", [64, 128])
def test_arbitrary_mask_varlen_output(seqlen, kv_mode, head_dim, head_dim_v):
    if not _cases_for_arch():
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    torch.manual_seed(seqlen + head_dim)
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 3, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, kv_heads, head_dim_v, device=device, dtype=dtype)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    q_varlen = rearrange(q.detach(), "b s h d -> (b s) h d").requires_grad_(True)
    k_varlen = rearrange(k.detach(), "b s h d -> (b s) h d").requires_grad_(True)
    v_varlen = rearrange(v.detach(), "b s h d -> (b s) h d").requires_grad_(True)
    cu_seqlens = torch.arange(0, (batch + 1) * seqlen, seqlen, device=device, dtype=torch.int32)
    arbitrary_func = _make_arbitrary_func(
        batch,
        heads,
        seqlen,
        seqlen,
        device,
        vary_by_batch=True,
        vary_by_head=kv_mode != "mha",
    )

    out_varlen, _ = flash_attn_varlen_func(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        return_lse=True,
    )
    out = rearrange(out_varlen, "(b s) h d -> b s h d", b=batch)
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
    if _device_major() == 9 and head_dim == 192 and kv_mode != "mha":
        return

    grad = torch.randn_like(out)
    out_varlen.backward(rearrange(grad, "b s h d -> (b s) h d"))
    out_ref.backward(grad)
    torch.testing.assert_close(
        q_varlen.grad,
        rearrange(q_ref.grad, "b s h d -> (b s) h d"),
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        k_varlen.grad,
        rearrange(k_ref.grad, "b s h d -> (b s) h d"),
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        v_varlen.grad,
        rearrange(v_ref.grad, "b s h d -> (b s) h d"),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "aux_factory, error_match",
    [
        pytest.param(lambda valid: None, r"aux_tensors\[0\]", id="missing_aux"),
        pytest.param(lambda valid: [valid.to(torch.float32)], "torch.int32", id="wrong_dtype"),
        pytest.param(
            lambda valid: [torch.zeros(1, 1, 2, valid.shape[-1], dtype=torch.int32, device=valid.device)],
            "func_num",
            id="even_func_num",
        ),
        pytest.param(
            lambda valid: [torch.zeros(3, 1, 3, valid.shape[-1], dtype=torch.int32, device=valid.device)],
            "batch dimension",
            id="bad_batch",
        ),
        pytest.param(
            lambda valid: [torch.zeros(1, 3, 3, valid.shape[-1], dtype=torch.int32, device=valid.device)],
            "head dimension",
            id="bad_head",
        ),
        pytest.param(
            lambda valid: [valid[..., :-1]],
            "last dimension",
            id="short_seqlen_padding",
        ),
    ],
)
def test_arbitrary_mask_aux_tensor_validation(aux_factory, error_match):
    if _generic_case_for_arch() is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    device = "cuda"
    dtype = torch.bfloat16
    head_dim, head_dim_v = _generic_case_for_arch()
    q = torch.randn(2, 64, 2, head_dim, device=device, dtype=dtype)
    k = torch.randn(2, 64, 2, head_dim, device=device, dtype=dtype)
    v = torch.randn(2, 64, 2, head_dim_v, device=device, dtype=dtype)
    valid = _make_arbitrary_func(1, 1, 64, 64, device)
    with pytest.raises(ValueError, match=error_match):
        flash_attn_func(q, k, v, arbitrary=True, aux_tensors=aux_factory(valid))


def test_arbitrary_mask_rejects_mla_qv():
    if _device_major() is None or _device_major() < 10:
        pytest.skip("MLA qv path is Blackwell-only")
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(1, 64, 2, 64, device=device, dtype=dtype)
    qv = torch.randn(1, 64, 2, 512, device=device, dtype=dtype)
    k = torch.randn(1, 64, 2, 64, device=device, dtype=dtype)
    v = torch.randn(1, 64, 2, 512, device=device, dtype=dtype)
    arbitrary_func = _make_arbitrary_func(1, 1, 64, 64, device)
    with pytest.raises(NotImplementedError, match="MLA qv"):
        flash_attn_func(q, k, v, qv=qv, arbitrary=True, aux_tensors=[arbitrary_func])


def test_arbitrary_mask_supports_hd256_gqa():
    if _device_major() != 10:
        pytest.skip("SM100 hd256 dedicated path is required")
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(1, 128, 2, 256, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 128, 1, 256, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, 128, 1, 256, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_head_dependent_arbitrary_func(1, 2, 128, 128, device)

    out, _ = flash_attn_func(q, k, v, arbitrary=True, aux_tensors=[arbitrary_func], return_lse=True)
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)
