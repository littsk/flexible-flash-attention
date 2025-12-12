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
from einops import rearrange

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


def create_tensors(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
):
    device = "cuda"
    # lengths = torch.randint(1, seqlen_q + 1, (batch_size,))
    # cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    # cu_seqlens_q = cu_seqlens_q.contiguous().to(dtype=torch.int32, device=device)
    # total_q = cu_seqlens_q[-1]
    # total_k = total_q
    q = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(1, 1).requires_grad_(True)
    k = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    ).uniform_(1, 1).requires_grad_(True)
    v = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    ).uniform_(1, 1).requires_grad_(True)
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

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        q * scale,
        k,
    )

    func_num = arbitrary_func.shape[2]
    for i in range(seqlen_q):
        for j in range(func_num // 2):
            qk_attn[:, :, i, arbitrary_func[0, 0, 2 * j, i]:arbitrary_func[0, 0, 2 * j + 1, i]] = -float("inf")
        qk_attn[:, :, i, arbitrary_func[0, 0, func_num - 1, i]:] = -float("inf")

    softmax_attn = F.softmax(qk_attn, dim=-1)
    out = torch.einsum(
        "bhnm,bmhd->bnhd",
        softmax_attn,
        v,
    )

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
    arbitrary_func = random_arbitrary_func_tensor(1, 1, 1, seqlen_q, seqlen_k, device="cuda")
    original_flex_mask = mask_mod_flex

    def mask_mod_flex(b, h, q_idx, kv_idx, arbitrary_func=arbitrary_func):
        return original_flex_mask(b, h, q_idx, kv_idx, arbitrary_func)

    aux_tensors_arg = [arbitrary_func]
    causal = False

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    # Compute block sparsity for mask_mod
    if COMPUTE_CAPABILITY == 10:
        sparse_tile_m = 2 * tile_m
    else:
        sparse_tile_m = tile_m

    bm = create_block_mask(
        mask_mod_flex,
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
    softmax_scale = 1.0 / math.sqrt(headdim)

    k_block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=k_mask_cnt,
        mask_block_idx=k_mask_idx,
        full_block_cnt=k_full_cnt,
        full_block_idx=k_full_idx,
    )
    linear_k_block_sparse_mask = bhqk_to_linear_sparse_tensors(k_block_sparse_mask)

    bm_bwd = create_block_mask(
        mask_mod_flex,
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
    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert (out_cute - out_ref_fp32).abs().max().item() <= 2 * (out_ref - out_ref_fp32).abs().max().item()

    dout = torch.ones_like(out_cute)

    dq, dk, dv = torch.autograd.grad(
        out_cute, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref_fp32, dk_ref_fp32, dv_ref_fp32) = torch.autograd.grad(
        out_ref_fp32, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
        out_ref, (tensors["q"], tensors["k"], tensors["v"]), dout
    )

    torch.set_printoptions(profile="full")
    print("dv: ", dv)
    print("dv_ref: ", dv_ref)

    print(f"dV max diff: {(dv - dv_ref_fp32).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_ref - dv_ref_fp32).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref_fp32).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_ref - dk_ref_fp32).abs().max().item()}")
    print(f"dQ max diff: {(dq - dq_ref_fp32).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_ref - dq_ref_fp32).abs().max().item()}")

    assert (dv - dv_ref_fp32).abs().max().item() <= 5 * (dv_ref - dv_ref_fp32).abs().max().item()
    assert (dk - dk_ref_fp32).abs().max().item() <= 5 * (dk_ref - dk_ref_fp32).abs().max().item()
    assert (dq - dq_ref_fp32).abs().max().item() <= 5 * (dq_ref - dq_ref_fp32).abs().max().item()


def test_arbitrary_mask(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, use_block_sparsity, tile_m, tile_n
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
    )


if __name__ == "__main__":
    test_arbitrary_mask(
        seqlen_q=128,
        seqlen_k=128,
        nheads=1,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=True,
        tile_m=128,
        tile_n=128,
    )