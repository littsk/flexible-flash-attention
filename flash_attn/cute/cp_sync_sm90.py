"""Hopper CP ready gates shared by forward and backward load warps."""
from typing import Optional
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
import cutlass.utils.distributed as cute_dist

@cute.jit
def wait_kv_ready(signal: Optional[cute.Tensor], head: Int32, block: Int32):
    if const_expr(signal is not None):
        if cute.arch.lane_idx() == 0:
            row = signal[head, None] if const_expr(cute.rank(signal) == 2) else signal
            view = cute.make_tensor(row.iterator + block, cute.make_layout((1,), stride=(1,)))
            ready = cute_dist.ld_bypass(view)[0]
            while ready == 0:
                ready = cute_dist.ld_bypass(view)[0]
        cute.arch.sync_warp()

@cute.jit
def gated_k_load(load_fn, signal, head, src_idx, producer_state):
    wait_kv_ready(signal, head, src_idx)
    load_fn(src_idx=src_idx, producer_state=producer_state)

# A completed async bulk write crosses the async/generic proxy boundary.
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm

@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    value = llvm.inline_asm(T.i32(), [], "fence.proxy.async.global; mov.u32 $0, 0;", "=r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(value)
