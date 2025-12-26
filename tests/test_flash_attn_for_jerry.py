# mask mod test script
# REFACTORED to use _flash_attn_fwd as the kernel entrypoint
#
# Test Organization:
# - test_static_masks: Fast tests for masks that don't need per-seqlen compilation
#   (identity, document, block_diagonal, etc.) with comprehensive seqlen coverage
# - test_parameterized_masks: Slower tests for masks that require recompilation per
#   seqlen pair (causal, block_causal, sliding_window) with reduced seqlen coverage
#
# Usage:
#   pytest test_mask_mod.py::test_static_masks         # Run only fast tests
#   pytest test_mask_mod.py::test_parameterized_masks  # Run only slow tests
#   pytest test_mask_mod.py                            # Run all tests

import math
from typing import Optional
from einops import rearrange, repeat

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch, bhqk_to_linear_sparse_tensors
from flash_attn.cute.mask_definitions import (
    get_mask_pair,
    STATIC_MASKS,
    random_arbitrary_func_tensor,
)
COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]


# def get_mask(arbitrary_func, seqlen_q, seqlen_k):
#     func_num = arbitrary_func.shape[2]
#     mask = torch.ones(1, 1, seqlen_q, seqlen_k).cuda()
#     for i in range(seqlen_q):
#         for j in range(seqlen_k):
#             value_valid = j < arbitrary_func[0, 0, 0, i]
#             for k in range(func_num // 2):
#                 if j >= arbitrary_func[0, 0, 2 * k + 1, i] and j < arbitrary_func[0, 0, 2 * k + 2, i]:
#                     value_valid = True
#             mask[:, :, i, j] = False if not value_valid else True
#     # print("mask is {}".format(mask))
#     return mask

import torch

import torch

def apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k):
    """
    qk_attn: [B, H, seqlen_q, seqlen_k]
    arbitrary_func: [1, 1, func_num, seqlen_q + 256] (int-like indices)
    Only the first `seqlen_q` positions of the last dim are used.
    """
    device = qk_attn.device
    B, H, sq, sk = qk_attn.shape
    assert sq == seqlen_q and sk == seqlen_k

    # Use only the first seqlen_q entries along last dim
    af = arbitrary_func[..., :seqlen_q]   # [1,1,func_num,seqlen_q]
    func_num = af.shape[2]

    # j grid: [1,1,1,seqlen_k]
    j = torch.arange(seqlen_k, device=device, dtype=af.dtype).view(1, 1, 1, seqlen_k)

    # base cutoff: af[0,0,0,i] -> [1,1,seqlen_q,1]
    base_cut = af[0, 0, 0, :].view(1, 1, seqlen_q, 1)
    base_valid = j < base_cut   # [1,1,seqlen_q,seqlen_k]

    # intervals: starts = af[0,0,1::2,:], ends = af[0,0,2::2,:]
    num_intervals = func_num // 2
    if num_intervals > 0:
        starts = af[0, 0, 1:func_num:2, :]  # [num_intervals, seqlen_q]
        ends   = af[0, 0, 2:func_num:2, :]  # [num_intervals, seqlen_q]

        # reshape for broadcasting: [1, num_intervals, seqlen_q, 1]
        starts = starts.view(1, num_intervals, seqlen_q, 1)
        ends   = ends.view(1, num_intervals, seqlen_q, 1)

        # j expanded: [1,1,seqlen_q,seqlen_k] -> then [1, num_intervals, seqlen_q, seqlen_k]
        j_exp = j.expand(1, num_intervals, seqlen_q, seqlen_k)

        in_interval = (j_exp >= starts) & (j_exp < ends)  # [1,num_intervals,seqlen_q,seqlen_k]
        interval_valid = in_interval.any(dim=1, keepdim=True)  # [1,1,seqlen_q,seqlen_k]
    else:
        interval_valid = torch.zeros(1, 1, seqlen_q, seqlen_k, dtype=torch.bool, device=device)

    # combined mask [1,1,seqlen_q,seqlen_k]
    valid = base_valid | interval_valid

    # expand to [B,H,seqlen_q,seqlen_k]
    valid = valid.expand(B, H, seqlen_q, seqlen_k)

    # set invalid positions to -inf (works in-place style by assigning back)
    qk_attn = torch.where(valid, qk_attn, torch.full_like(qk_attn, -float("inf")))

    return qk_attn





def create_tensors(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
):
    device = "cuda"
    # lengths = torch.randint(1, seqlen_q + 1, (batch_size,))
    # cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    # cu_seqlens_q = cu_seqlens_q.contiguous().to(dtype=torch.int32, device=device)
    # total_q = cu_seqlens_q[-1]
    # total_k = total_q
    q = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
    k = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    ).uniform_(-1, 1).requires_grad_(True)
    v = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    ).uniform_(-1, 1).requires_grad_(True)

    # q = torch.ones(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)
    # k = torch.ones(
    #     batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    # ).requires_grad_(True)
    # v = torch.ones(
    #     batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    # ).requires_grad_(True)

    out = torch.empty(
        batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, nheads, seqlen_q, device=device, dtype=torch.float32)

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "out": out.contiguous(),
        "lse": lse.contiguous(),
        "cu_seqlens_q": None,
        "cu_seqlens_k": None,
    }

def compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False):
    """Compute reference of arbitrary mask"""
    q = tensors["q"] if not up_cast else tensors["q"].float()
    k = tensors["k"] if not up_cast else tensors["k"].float()
    v = tensors["v"] if not up_cast else tensors["v"].float()
    batch_size = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    nheads = q.shape[2]
    nheads_kv = k.shape[2]
    headdim = q.shape[3]
    scale = 1.0 / math.sqrt(headdim)


    if nheads_kv == nheads:
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q * scale,
            k,
        )
    else:
        k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q * scale,
            k,
        )
    
    has_nan = torch.isnan(qk_attn).any()

    # func_num = arbitrary_func.shape[2]
    # for i in range(seqlen_q):
    #     for j in range(seqlen_k):
    #         value_valid = j < arbitrary_func[0, 0, 0, i]
    #         for k in range(func_num // 2):
    #             if j >= arbitrary_func[0, 0, 2 * k + 1, i] and j < arbitrary_func[0, 0, 2 * k + 2, i]:
    #                 value_valid = True
    #         qk_attn[:, :, i, j] = -float("inf") if not value_valid else qk_attn[:, :, i, j]

    qk_attn = apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k)


    all_inf_mask = torch.all(qk_attn == float('-inf'), dim=-1, keepdim=True)  # [B, H, seqlen_q, 1]
    softmax_attn = F.softmax(qk_attn, dim=-1)
    softmax_attn = torch.where(all_inf_mask, torch.zeros_like(softmax_attn), softmax_attn)

    out = torch.einsum(
        "bhnm,bmhd->bnhd",
        softmax_attn,
        v,
    )


    is_all_zero = torch.count_nonzero(arbitrary_func) == 0
    if is_all_zero:
        out[:] = 0.0

    return out


def _run_mask_test(
    seqlen_q,
    seqlen_k,
    nheads,
    kv_mode,
    headdim,
    dtype,
    tile_m,
    tile_n,
    use_block_sparsity,
    load_tensor=False,
):
    # torch.manual_seed(42)

    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
    elif kv_mode == "gqa":
        nheads_kv = nheads // 2
    elif kv_mode == "mqa":
        nheads_kv = 1
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    batch_size = 1
    headdim_v = headdim

    # aux_tensors_arg = None
    # mask_mod_cute, mask_mod_flex = get_mask_pair("causal", seqlen_q, seqlen_k)
    mask_mod_cute, mask_mod_flex = get_mask_pair("arbitrary")

    if not load_tensor:
        arbitrary_func = random_arbitrary_func_tensor(1, 1, 3, seqlen_q, seqlen_k, device="cuda")


    if load_tensor:
        func_path = "jerry_func_true.pt"
        arbitrary_func = torch.load(func_path, map_location="cpu").cuda()

    original_flex_mask = mask_mod_flex

    causal = False

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    if load_tensor:
        q_path = "jerry_q_true.pt"
        k_path = "jerry_k_true.pt"
        v_path = "jerry_v_true.pt"

        tensors["q"] = torch.load(q_path, map_location="cpu").cuda().requires_grad_()  # 建议先都加载到 CPU
        tensors["k"] = torch.load(k_path, map_location="cpu").cuda().requires_grad_()  # 建议先都加载到 CPU
        tensors["v"] = torch.load(v_path, map_location="cpu").cuda().requires_grad_()   # 建议先都加载到 CPU
        seqlen_q = tensors["q"].shape[1]
        seqlen_k = tensors["k"].shape[1]

    aux_tensors_arg = [arbitrary_func]

    frozen_af = arbitrary_func  # freeze
    def mask_mod_flex_new(b, h, q_idx, kv_idx, arbitrary_func_func=frozen_af):
        return original_flex_mask(b, h, q_idx, kv_idx, arbitrary_func_func)

    # Compute block sparsity for mask_mod
    if COMPUTE_CAPABILITY == 10:
        sparse_tile_m = 2 * tile_m
    else:
        sparse_tile_m = tile_m

    bm = create_block_mask(
        mask_mod_flex_new,
        1,
        1,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx = None, None, None, None
    if COMPUTE_CAPABILITY == 10:
        _, _, k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm.as_tuple()
    else:
        k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm.as_tuple()
    headdim = tensors["q"].shape[3]
    softmax_scale = 1.0 / math.sqrt(headdim)

    k_block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=k_mask_cnt,
        mask_block_idx=k_mask_idx,
        full_block_cnt=k_full_cnt,
        full_block_idx=k_full_idx,
    )
    linear_k_block_sparse_mask = bhqk_to_linear_sparse_tensors(k_block_sparse_mask)

    bm_bwd = create_block_mask(
        mask_mod_flex_new,
        1,
        1,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx = None, None, None, None
    if COMPUTE_CAPABILITY == 10:
        _, _, _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
    else:
        _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
    q_block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=q_mask_cnt,
        mask_block_idx=q_mask_idx,
        full_block_cnt=q_full_cnt,
        full_block_idx=q_full_idx,
    )
    linear_q_block_sparse_mask = bhqk_to_linear_sparse_tensors(q_block_sparse_mask)
    

    out_cute, lse_cute = flash_attn_func(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        softmax_scale=softmax_scale,
        causal=causal,
        arbitrary=True,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=False,
        deterministic=False,
        mask_mod=None,
        linear_k_block_sparse_tensors=linear_k_block_sparse_mask,
        linear_q_block_sparse_tensors=linear_q_block_sparse_mask,
        aux_tensors=aux_tensors_arg,
    )

    out_ref_fp32 = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=True)
    out_ref = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False)

    print(f"Output max diff: {(out_cute - out_ref_fp32).abs().max().item()}")
    print(f"Pytorch max diff: {(out_ref - out_ref_fp32).abs().max().item()}")

    # Check for invalid values
    print("out_cute is {}".format(out_cute))
    print("out_ref_fp16 is {}".format(out_ref))
    print("out_ref_fp32 is {}".format(out_ref_fp32))
    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    
    assert (out_cute - out_ref_fp32).abs().max().item() <= 2 * (out_ref - out_ref_fp32).abs().max().item()

    # print("out_cute is {}".format(out_cute))
    # print("out_ref_fp32 is {}".format(out_ref_fp32))

    dout = torch.rand_like(out_cute)

    dq, dk, dv = torch.autograd.grad(
        out_cute, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref_fp32, dk_ref_fp32, dv_ref_fp32) = torch.autograd.grad(
        out_ref_fp32, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
        out_ref, (tensors["q"], tensors["k"], tensors["v"]), dout
    )

    print("dq is {}".format(dq))
    print("dq_ref_fp32 is {}".format(dq_ref_fp32))

    print("dk is {}".format(dk))
    print("dk_ref_fp32 is {}".format(dk_ref_fp32))

    print("dv is {}".format(dv))
    print("dv_ref_fp32 is {}".format(dv_ref_fp32))

    print(f"dV max diff: {(dv - dv_ref_fp32).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_ref - dv_ref_fp32).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref_fp32).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_ref - dk_ref_fp32).abs().max().item()}")
    print(f"dQ max diff: {(dq - dq_ref_fp32).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_ref - dq_ref_fp32).abs().max().item()}")

    # diff = (dq - dq_ref).abs()
    # print(f"shape diff: {diff.shape}")  # b s h d
    # for bs in range(batch_size):
    #     for h in range(nheads):
    #         for s in range(seqlen_q):
    #             diff_i = diff[bs, s, h, 0] #只取当前dim的第一个值
    #             if diff_i.abs().max().item() > 1e-2:
    #                 out_i = dq[bs, s, h, 0]
    #                 out_ref_i = dq_ref[bs, s, h, 0]
    #                 print(f"=== out[bs, s, h, 0] = {out_i}, out_ref[bs, s, h, 0] = {out_ref_i}, diff[bs, s, h, 0] = {diff_i}, bs = {bs}, s = {s}, h = {h}")
    
    assert (dv - dv_ref_fp32).abs().max().item() <= 5 * (dv_ref - dv_ref_fp32).abs().max().item()
    assert (dk - dk_ref_fp32).abs().max().item() <= 5 * (dk_ref - dk_ref_fp32).abs().max().item()
    assert (dq - dq_ref_fp32).abs().max().item() <= 5 * (dq_ref - dq_ref_fp32).abs().max().item()


def test_arbitrary_mask(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, use_block_sparsity, tile_m, tile_n, load_tensor=False
):
    """Test arbitrary mask
    """
    if COMPUTE_CAPABILITY == 10 and (tile_m, tile_n) != (128, 128):
        pytest.skip("TODO: Non-128x128 tiles currently not supported on SM 10.0. due to TMEM")

    _run_mask_test(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=nheads,
        kv_mode=kv_mode,
        headdim=headdim,
        dtype=dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        use_block_sparsity=use_block_sparsity,
        load_tensor=load_tensor
    )


if __name__ == "__main__":
    test_arbitrary_mask(
        seqlen_q=12288,
        seqlen_k=12288,
        nheads=1,
        kv_mode="mha",
        headdim=64,
        dtype=torch.bfloat16,
        use_block_sparsity=True,
        tile_m=128,
        tile_n=128,
        load_tensor=True
    )