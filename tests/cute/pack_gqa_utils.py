"""Shared correctness cases for the PackGQA backward tests and benchmark."""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import torch
from cutlass import cute
from torch.nn.attention.flex_attention import create_block_mask

from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    compute_dq_write_order_from_block_mask,
)
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


def make_masks(pattern: str, sq: int, sk: int) -> tuple[Callable, Callable]:
    def torch_mask(b, h, q, k):
        if pattern == "causal":
            return k <= q + sk - sq
        if pattern == "rows":
            return (q % 256 != 255) & (k % 128 <= q % 128)
        if pattern == "dense":
            return q >= 0
        block = (q // 256 + k // 128) % 3
        active = (k // 128) % 7 != 5
        if pattern == "sparse":
            return (block != 2) & active
        if pattern == "empty":
            return q < 0
        return ((block == 0) | ((block == 1) & (q % 128 >= k % 128))) & active

    @cute.jit
    def cute_mask(b, h, q, k, seqlen_info, aux_tensors):
        if cutlass.const_expr(pattern == "causal"):
            return k <= q + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        elif cutlass.const_expr(pattern == "rows"):
            return (q % 256 != 255) & (k % 128 <= q % 128)
        elif cutlass.const_expr(pattern == "dense"):
            return q >= 0
        elif cutlass.const_expr(pattern == "empty"):
            return q < 0
        else:
            block = (q // 256 + k // 128) % 3
            active = (k // 128) % 7 != 5
            if cutlass.const_expr(pattern == "sparse"):
                return (block != 2) & active
            else:
                return ((block == 0) | ((block == 1) & (q % 128 >= k % 128))) & active

    return torch_mask, cute_mask


@dataclass
class BackwardCase:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    dout: torch.Tensor
    fwd_sparse: BlockSparseTensorsTorch
    bwd_sparse: BlockSparseTensorsTorch
    torch_mask: Callable
    cute_mask: Callable
    packed_bwd_sparse: BlockSparseTensorsTorch | None = None

    def backward(self, pack: bool, **kwargs) -> tuple[torch.Tensor, ...]:
        return _flash_attn_bwd(
            self.q,
            self.k,
            self.v,
            self.out,
            self.dout,
            self.lse,
            deterministic=True,
            pack_gqa=pack,
            mask_mod=self.cute_mask,
            block_sparse_tensors=(
                self.packed_bwd_sparse
                if pack and self.packed_bwd_sparse is not None
                else self.bwd_sparse
            ),
            paired_sparse_bwd=self.packed_bwd_sparse is not None,
            **kwargs,
        )


def make_case(
    sq: int = 512,
    sk: int = 2048,
    hq: int = 64,
    hkv: int = 8,
    dim: int = 128,
    pattern: str = "mixed",
    batch: int = 1,
    spt: bool = False,
    kv_block: int = 128,
    cta_group_size: int = 1,
) -> BackwardCase:
    if cta_group_size not in (1, 2):
        raise ValueError("cta_group_size must be 1 or 2")
    if cta_group_size == 2 and (
        sq % 256 or sk % 128 or dim != 128 or kv_block != 128 or spt
    ):
        raise ValueError(
            "paired 2CTA requires aligned Q256/KV128, D128, KV block128 and spt=False"
        )
    torch.manual_seed(2026)
    mask, mask_cute = make_masks(pattern, sq, sk)
    bm = create_block_mask(
        mask, batch, 1, sq, sk, device="cuda", BLOCK_SIZE=(256, kv_block)
    )
    fwd = BlockSparseTensorsTorch(
        bm.kv_num_blocks,
        bm.kv_indices,
        bm.full_kv_num_blocks,
        bm.full_kv_indices,
        block_size=(256, kv_block),
    )
    order, full_order = compute_dq_write_order_from_block_mask(bm, spt=spt)
    bwd = BlockSparseTensorsTorch(
        bm.q_num_blocks,
        bm.q_indices,
        bm.full_q_num_blocks,
        bm.full_q_indices,
        block_size=(256, kv_block),
        dq_write_order=order,
        dq_write_order_full=full_order,
        spt=spt,
    )
    packed_bwd = None
    if cta_group_size == 2:
        # A 256-token KV mask gives the exact pair union/full intersection.
        # The original element predicate also masks newly introduced pair tiles.
        paired_bm = create_block_mask(
            mask, batch, 1, sq, sk, device="cuda", BLOCK_SIZE=(256, 256)
        )
        pair_order, pair_full_order = compute_dq_write_order_from_block_mask(
            paired_bm, spt=False
        )
        pair_counts = paired_bm.q_num_blocks.repeat_interleave(2, dim=2)
        physical_k = pair_counts.shape[2]
        active = (bm.q_num_blocks + bm.full_q_num_blocks > 0).to(torch.int32)
        active = torch.nn.functional.pad(active, (0, physical_k - sk // 128))
        work = torch.tensor(
            [
                (n, h, b)
                for b in range(batch)
                for n in range(0, physical_k, 2)
                for h in range(hq)
            ],
            dtype=torch.int32,
            device="cuda",
        )
        bwd = BlockSparseTensorsTorch(
            pair_counts,
            paired_bm.q_indices.repeat_interleave(2, dim=2),
            paired_bm.full_q_num_blocks.repeat_interleave(2, dim=2),
            paired_bm.full_q_indices.repeat_interleave(2, dim=2),
            block_size=(256, 128),
            dq_write_order=pair_order.repeat_interleave(2, dim=2),
            dq_write_order_full=pair_full_order.repeat_interleave(2, dim=2),
            spt=False,
            bwd_kv_order=torch.arange(physical_k, dtype=torch.int32, device="cuda"),
            bwd_work_map=work,
            bwd_original_active=active.expand(batch, hq, physical_k).contiguous(),
        )
        group = hq // hkv
        packed_work = work[work[:, 1] % group == 0].clone()
        packed_work[:, 1] //= group
        packed_bwd = bwd._replace(
            bwd_work_map=packed_work,
            bwd_original_active=active.expand(batch, hkv, physical_k).contiguous(),
        )
    q = torch.randn(batch, sq, hq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, sk, hkv, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        mask_mod=mask_cute,
        block_sparse_tensors=fwd,
        pack_gqa=False,
        return_lse=True,
    )
    return BackwardCase(
        q, k, v, out, lse, torch.randn_like(out), fwd, bwd, mask, mask_cute, packed_bwd
    )


def reference_gradients(
    case: BackwardCase,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent native PyTorch autograd, one KV group at a time."""
    _b, sq, hq, d = case.q.shape
    sk, hkv = case.k.shape[1:3]
    group = hq // hkv
    q_idx = torch.arange(sq, device=case.q.device)[:, None]
    k_idx = torch.arange(sk, device=case.q.device)[None, :]
    valid = case.torch_mask(0, 0, q_idx, k_idx).expand(sq, sk)
    row_valid = valid.any(-1)
    dq = torch.empty_like(case.q, dtype=dtype)
    dk = torch.empty_like(case.k, dtype=dtype)
    dv = torch.empty_like(case.v, dtype=dtype)
    for h in range(hkv):
        q = (
            case.q[:, :, h * group : (h + 1) * group]
            .to(dtype)
            .detach()
            .requires_grad_()
        )
        k = case.k[:, :, h].to(dtype).detach().requires_grad_()
        v = case.v[:, :, h].to(dtype).detach().requires_grad_()
        scores = torch.einsum("bqhd,bkd->bhqk", q, k) * d**-0.5
        scores = scores.masked_fill(~valid, -torch.inf)
        # Avoid undefined softmax gradients for all-masked rows.
        scores = torch.where(row_valid[None, None, :, None], scores, 0.0)
        p = scores.softmax(-1).masked_fill(~valid, 0.0)
        out = torch.einsum("bhqk,bkd->bqhd", p, v)
        grads = torch.autograd.grad(
            out, (q, k, v), case.dout[:, :, h * group : (h + 1) * group].to(dtype)
        )
        dq[:, :, h * group : (h + 1) * group], dk[:, :, h], dv[:, :, h] = grads
    return dq, dk, dv


def check_gradients(actual: tuple, reference: tuple) -> dict[str, float]:
    errors = {}
    for name, got, expected in zip(("dq", "dk", "dv"), actual, reference):
        assert got.dtype == torch.bfloat16
        assert torch.isfinite(got).all()
        # BF16 P/dS rounding causes small errors around reference zeros.
        torch.testing.assert_close(got.float(), expected.float(), atol=0.015, rtol=0.04)
        delta = got.float() - expected.float()
        relative_rms = (
            (delta.square().mean() / expected.float().square().mean().clamp_min(1e-30))
            .sqrt()
            .item()
        )
        assert relative_rms < 0.02, f"{name}: relative RMS error {relative_rms}"
        errors[name] = delta.abs().max().item()
        errors[f"{name}_relative_rms"] = relative_rms
    return errors
