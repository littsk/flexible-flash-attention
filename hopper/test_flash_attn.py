import math
import random

import pytest
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, _flash_attn_forward, flex_flash_attn_func
from tests.test_util import generate_random_padding_mask, generate_qkv, construct_local_mask, attention_ref

ABS_TOL = 5e-3
REL_TOL = 1e-1

def print_diffs(out, out_ref):
    out_1d = out.flatten()
    out_ref_1d = out_ref.flatten()
    for idx, (e_o, e_o_ref) in enumerate(zip(out_1d, out_ref_1d)):
        diff = e_o - e_o_ref
        abs_diff = abs(diff)
        abs_ref = abs(e_o_ref + 1e-5)
        relative_diff = abs_diff / abs_ref
        if abs_diff > ABS_TOL or relative_diff > REL_TOL:
            print(f"==== diff ==== {idx}, test: {e_o}, ref: {e_o_ref}")


# @pytest.mark.skip(reason="skipped")
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("gqa_parallel", [False, True])
@pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize("descale", [1.0])
@pytest.mark.parametrize("descale", [1.0, 2.0, 3.0])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
def test_flash_attn_output_fp8(
    seqlen_q, seqlen_k, d, causal, local, deterministic, mha_type, dtype, descale, gqa_parallel
):
    device = "cuda"
    dtype_init = torch.bfloat16
    print(dtype)
    print('causal',causal)
    print('local',local)
    print('gqa_parallel',gqa_parallel)
    # set seed
    torch.random.manual_seed(42)
    # batch_size = 40
    # nheads = 16
    batch_size = 4
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    # nheads_kv = 1
    # batch_size = 9
    # nheads = 6
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_init, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_init, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_init, requires_grad=True)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    softmax_scale = q.shape[-1] ** (-0.5)
    descale_q = torch.tensor([descale], dtype=torch.float32, device='cuda')
    descale_k = torch.tensor([descale], dtype=torch.float32, device='cuda')
    descale_v = torch.tensor([descale], dtype=torch.float32, device='cuda')

    out, lse = flash_attn_func(q, k, v, causal=causal, window_size=window_size, deterministic=deterministic, gqa_parallel=gqa_parallel,
                                descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)

    q = q.to(dtype_init)
    k = k.to(dtype_init)
    v = v.to(dtype_init)

    descale_q = descale_q.to(dtype_init)
    descale_k = descale_k.to(dtype_init)
    descale_v = descale_v.to(dtype_init)
    q = q * descale_q
    k = k * descale_k
    v = v * descale_v

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    # qk = torch.einsum('bshd,bthd->bhst', q, k).float()
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # exp_sum = s_tmp.sum(-1)
    # qk = torch.einsum('bthd,bshd->bhts', q.float() / math.sqrt(d), k.float())
    # lse_ref = torch.logsumexp(qk, dim=-1)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
    # P = torch.softmax(qk, -1)
    # dP = P * (dS - do_o.unsqueeze(1))
    # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
    # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
    # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())
    # breakpoint()
    
    # assert (out - out_ref).abs().max().item() <= 4 * (out_pt - out_ref).abs().max().item() + 1e-2
    atol = 4 * (out_pt - out_ref).abs().max().item() + 1e-2
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=atol, check_dtype=False)


# @pytest.mark.skip(reason="skipped")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("gqa_parallel", [False, True])
# @pytest.mark.parametrize("gqa_parallel", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize("d", [64, 96, 128])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("descale", [1.0])
# @pytest.mark.parametrize("descale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
    seqlen_q, seqlen_k, d, causal, local, deterministic, mha_type, dtype, descale, gqa_parallel
):
    device = "cuda"
    if(dtype == torch.float8_e4m3fn):
        dtype_init = torch.bfloat16
    else:
        dtype_init = dtype
    print(dtype)
    print('causal',causal)
    print('local',local)
    print('gqa_parallel',gqa_parallel)
    # set seed
    torch.random.manual_seed(42)
    # batch_size = 40
    # nheads = 16
    batch_size = 4
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    # nheads_kv = 1
    # batch_size = 9
    # nheads = 6
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_init, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_init, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_init, requires_grad=True)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    softmax_scale = q.shape[-1] ** (-0.5)
    descale_q = torch.tensor([descale], dtype=torch.float32, device='cuda')
    descale_k = torch.tensor([descale], dtype=torch.float32, device='cuda')
    descale_v = torch.tensor([descale], dtype=torch.float32, device='cuda')

    if(dtype != torch.float8_e4m3fn):
        out, lse = flash_attn_func(q, k, v, causal=causal, window_size=window_size, deterministic=deterministic, gqa_parallel=gqa_parallel)
    else:
        out, lse = flash_attn_func(q, k, v, causal=causal, window_size=window_size, deterministic=deterministic, gqa_parallel=gqa_parallel,
                                   descale_q=descale_q, descale_k=descale_k, descale_v=descale_v)

    q = q.to(dtype_init)
    k = k.to(dtype_init)
    v = v.to(dtype_init)

    if(dtype == torch.float8_e4m3fn):
        descale_q = descale_q.to(dtype_init)
        descale_k = descale_k.to(dtype_init)
        descale_v = descale_v.to(dtype_init)
        q = q * descale_q
        k = k * descale_k
        v = v * descale_v

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    # qk = torch.einsum('bshd,bthd->bhst', q, k).float()
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # exp_sum = s_tmp.sum(-1)
    # qk = torch.einsum('bthd,bshd->bhts', q.float() / math.sqrt(d), k.float())
    # lse_ref = torch.logsumexp(qk, dim=-1)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    if d <= 128 and dtype != torch.float8_e4m3fn:
        g = torch.randn_like(out)
        do_o = (g.float() * out.float()).sum(-1)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
    # P = torch.softmax(qk, -1)
    # dP = P * (dS - do_o.unsqueeze(1))
    # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
    # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
    # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())
    # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    # breakpoint()
    if(dtype != torch.float8_e4m3fn):
        assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 3e-5
    else:
        # just test correctness of fp8 kernel w/o further quantization techniques
        assert (out - out_ref).abs().max().item() <= 4 * (out_pt - out_ref).abs().max().item() + 2e-2

    if d <= 128 and dtype != torch.float8_e4m3fn:
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + 3e-5
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + 3e-5
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + 3e-5


# @pytest.mark.skip(reason="skipped")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("add_unused_qkv", [False, True])
# @pytest.mark.parametrize("add_unused_qkv", [True])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [256])
# @pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("d", [64, 128])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (1, 3),
        (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (113, 203),
        (128, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (512, 256),
        (640, 128),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, causal, local, deterministic, add_unused_qkv, mha_type, dtype
):
    print(f"seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, d: {d}, causal: {causal}, local: {local}, deterministic: {deterministic}, add_unused_qkv: {add_unused_qkv}, mha_type: {mha_type}, dtype: {dtype}")
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 1
    # nheads = 1
    # nheads_kv = 1
    batch_size = 9
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)

    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random", zero_lengths=False)
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random", zero_lengths=True)
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')

    def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
        if add_unused:
            another_mask = generate_random_padding_mask(max_seq_len, bs, device)
            attn_mask = torch.logical_and(padding_mask, another_mask)
            unused_mask = torch.logical_xor(torch.logical_or(padding_mask, another_mask), attn_mask)
        else:
            attn_mask = padding_mask
            unused_mask = None
        return attn_mask, unused_mask

    query_padding_mask, query_unused_mask = _gen_unused_masks(query_padding_mask, add_unused_qkv, seqlen_q, batch_size, q.device)
    key_padding_mask, key_unused_mask = _gen_unused_masks(key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device)

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False, query_unused_mask=query_unused_mask, key_unused_mask=key_unused_mask)
    # print("cu_seqlens_q: ", cu_seqlens_q)
    # print("cu_seqlens_k: ", cu_seqlens_k)
    # print("q_unpad, shape: ", q_unpad.shape)
    # print("k_unpad, shape: ", k_unpad.shape)
    # print("v_unpad, shape: ", v_unpad.shape)
    out_unpad, sm_lse = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        deterministic=deterministic,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        window_size=window_size,
    )
    out = output_pad_fn(out_unpad)
    if query_unused_mask is not None:
        q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")
        out.masked_fill_(q_zero_masking, 0.0)
    dropout_mask = None

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    if d <= 128:
        (
            dq_unpad,
            dk_unpad,
            dv_unpad,
        ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        if key_unused_mask is not None:
            k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
            dk.masked_fill_(k_zero_masking, 0.0)
            dv.masked_fill_(k_zero_masking, 0.0)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        zero_masking = rearrange(torch.logical_not(torch.any(key_padding_mask, 1)), "b -> b 1 1 1")
        dk_ref.masked_fill_(zero_masking, 0.0)
        dv_ref.masked_fill_(zero_masking, 0.0)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        dk_pt.masked_fill_(zero_masking, 0.0)
        dv_pt.masked_fill_(zero_masking, 0.0)
        dq = dq_pad_fn(dq_unpad)
        if query_unused_mask is not None:
            dq.masked_fill_(q_zero_masking, 0.0)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if d <= 128:
        assert (dq - dq_ref).abs().max().item() < 1e-4 or (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() < 1e-4 or (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() < 1e-4 or (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()


def get_mask_from_ranges(q_ranges, k_ranges, is_causal_mapping, q_len, k_len):
    bsz = q_ranges.shape[0]
    mask = torch.zeros((q_len, k_len), device='cuda', dtype=torch.bool)
    for i in range(bsz):
        assert is_causal_mapping[i] == False
        mask[q_ranges[i, 0]:q_ranges[i, 1], k_ranges[i, 0]:k_ranges[i, 1]] = True
    return mask


def torch_attn_ref(q, k, v, mask, layout="thd", high_precision=True):
    if layout == "thd":
        q = rearrange(q, "t h d -> 1 h t d")
        k = rearrange(k, "t h d -> 1 h t d")
        v = rearrange(v, "t h d -> 1 h t d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        out = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64), attn_mask=mask)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if layout == "thd":
        out = rearrange(out, "1 h t d -> t h d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        out = out.to(q.dtype)
    return out


def generate_qk_ranges(seqlen_q, seqlen_k, bsz, device='cuda'):
    """生成q和k的ranges
    
    Args:
        seqlen: 序列长度
        bsz: batch size
        device: 设备,默认为'cuda'
        
    Returns:
        q_range: q的ranges张量,形状为(bsz, 2)
        k_range: k的ranges张量,形状为(bsz, 2)
    """
    
    random.seed(42)
    
    if bsz == 1:
        # bsz为1时直接使用完整序列
        q_ranges = [[0, seqlen_q]]
        max_seqlen_q = seqlen_q
        
        # 随机生成k_range
        start = random.randint(0, seqlen_k-1)
        end = random.randint(start+1, seqlen_k)
        k_ranges = [[start, end]]
        max_seqlen_k = end - start
        
    else:
        # 随机获取bsz-1个整数作为q的分割点
        points = sorted(random.sample(range(seqlen_q), bsz-1))
        
        max_seqlen_q = 0
        max_seqlen_k = 0

        # 构建q_range
        q_ranges = [[0, points[0]]]
        for i in range(bsz-2):
            q_ranges.append([points[i], points[i+1]])
        q_ranges.append([points[-1], seqlen_q])
        for q_range in q_ranges:
            max_seqlen_q = max(max_seqlen_q, q_range[1] - q_range[0])
        
        # 随机生成k_ranges
        k_ranges = []
        for i in range(bsz):
            start = random.randint(0, seqlen_k-1)
            end = random.randint(start+1, seqlen_k)
            k_ranges.append([start, end])
            max_seqlen_k = max(max_seqlen_k, end - start)
            
    q_ranges = torch.tensor(q_ranges, device=device, dtype=torch.int32)
    k_ranges = torch.tensor(k_ranges, device=device, dtype=torch.int32)

    is_causal_mapping = torch.tensor([False] * bsz, device=device, dtype=torch.bool)
    
    return q_ranges, k_ranges, is_causal_mapping, max_seqlen_q, max_seqlen_k

# @pytest.mark.skip(reason="skipped")
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen_q", [8, 256, 551, 1234, 1999]) # hang when seqlen is smaller than 7
@pytest.mark.parametrize("seqlen_k", [8, 256, 551, 1234]) # hang when seqlen is smaller than 7
@pytest.mark.parametrize("bsz", [1, 2])                       
def test_flex_flash_attn_output(
    seqlen_q, 
    seqlen_k, 
    bsz,
    d,
    mha_type, 
    dtype
):
    device = 'cuda'
    torch.random.manual_seed(42)

    q_ranges, k_ranges, is_causal_mapping, max_seqlen_q, max_seqlen_k = generate_qk_ranges(seqlen_q * bsz, seqlen_k * bsz, bsz, device)

    # print(f"q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    q = torch.randn(bsz * seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz * seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(bsz * seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(bsz * seqlen_q, nheads, d, device=device, dtype=dtype)

    out, _ = flex_flash_attn_func(q, k, v, q_ranges, k_ranges, is_causal_mapping, max_seqlen_q, max_seqlen_k, softmax_scale=None, deterministic=False)
    out.backward(g)
    dq, dk, dv = q.grad, k.grad, v.grad
    q.grad, k.grad, v.grad = None, None, None

    out_ref = torch_attn_ref(q, k, v, mask=get_mask_from_ranges(q_ranges, k_ranges, is_causal_mapping, seqlen_q * bsz, seqlen_k * bsz), layout="thd", high_precision=True)
    out_ref.backward(g)
    dq_ref, dk_ref, dv_ref = q.grad, k.grad, v.grad
    q.grad, k.grad, v.grad = None, None, None
    
    out_ref_low_precision = torch_attn_ref(q, k, v, mask=get_mask_from_ranges(q_ranges, k_ranges, is_causal_mapping, seqlen_q * bsz, seqlen_k * bsz), layout="thd", high_precision=False)
    out_ref_low_precision.backward(g)
    dq_ref_low_precision, dk_ref_low_precision, dv_ref_low_precision = q.grad, k.grad, v.grad
    q.grad, k.grad, v.grad = None, None, None

    assert (out - out_ref_low_precision).abs().max().item() <= 2 * (out_ref_low_precision - out_ref).abs().max().item()
    assert (dq - dq_ref_low_precision).abs().max().item() <= 2 * (dq_ref_low_precision - dq_ref).abs().max().item()

    if d <= 128:
        assert (dk - dk_ref_low_precision).abs().max().item() < 1e-4 or (dk - dk_ref_low_precision).abs().max().item() <= 3 * (dk_ref_low_precision - dk_ref).abs().max().item()
        assert (dv - dv_ref_low_precision).abs().max().item() < 1e-4 or (dv - dv_ref_low_precision).abs().max().item() <= 3 * (dv_ref_low_precision - dv_ref).abs().max().item()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    elsit = []
    print("\n", flush=True)
    print(f"=========================START=========================", flush=True)
    try:
        torch.testing.assert_close(out, out_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    except Exception as e:
        print(f"---------------------------Start Out check---------------------------", flush=True)
        print(f"Failed out check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
        print(e, flush=True)
        print(f"---------------------------End Out check---------------------------", flush=True)
        elsit.append(e)
    try:
        torch.testing.assert_close(dq, dq_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    except Exception as e:
        print(f"---------------------------Start dq check---------------------------", flush=True)
        print(f"Failed dq check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
        print(e, flush=True)
        print(f"---------------------------End dq check---------------------------", flush=True)
        elsit.append(e)
    try:
        torch.testing.assert_close(dk, dk_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    except Exception as e:
        print(f"---------------------------Start dk check---------------------------", flush=True)
        print(f"Failed dk check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
        print(e, flush=True)
        print(f"---------------------------End dk check---------------------------", flush=True)
        elsit.append(e)
    try:
        torch.testing.assert_close(dv, dv_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    except Exception as e:
        print(f"---------------------------Start dv check---------------------------", flush=True)
        print(f"Failed dv check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
        print(e, flush=True)
        print(f"---------------------------End dv check---------------------------", flush=True)
        elsit.append(e)
    print(f"=========================END=========================", flush=True)

    # for e in elsit:
    #     raise e
