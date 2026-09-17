"""Hopper CP ready gates shared by forward and backward load warps."""

from typing import Optional

import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def load_kv_ready(ptr: cute.Pointer, *, loc=None, ip=None):
    """Keep the peer-ready reload inside polling loops.

    CUTLASS DSL 4.6 can hoist ``ld_bypass`` out of a dynamic loop and leave an
    empty self-loop. Side-effecting inline PTX preserves both the reload and the
    system-acquire contract.
    """
    value = llvm.inline_asm(
        T.i32(),
        [ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.acquire.sys.global.u32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(value)


@cute.jit
def wait_kv_ready(signal: Optional[cute.Tensor], head: Int32, block: Int32):
    if const_expr(signal is not None):
        if cute.arch.lane_idx() == 0:
            row = signal[head, None] if const_expr(cute.rank(signal) == 2) else signal
            view = cute.make_tensor(row.iterator + block, cute.make_layout((1,), stride=(1,)))
            ready = load_kv_ready(view.iterator)
            while ready == 0:
                ready = load_kv_ready(view.iterator)
        cute.arch.sync_warp()


@cute.jit
def gated_k_load(load_fn, signal, head, src_idx, producer_state):
    wait_kv_ready(signal, head, src_idx)
    load_fn(src_idx=src_idx, producer_state=producer_state)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    """Make completed async bulk writes visible to generic loads."""
    value = llvm.inline_asm(
        T.i32(),
        [],
        "fence.proxy.async.global; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(value)
