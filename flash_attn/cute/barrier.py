import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def red_relaxed(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def atomic_add_acq_rel_sys(
    ptr: cute.Pointer, val: Int32, *, loc=None, ip=None
) -> Int32:
    """System-scope atomic add returning the value before the add."""
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    old = llvm.inline_asm(
        T.i32(),
        [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "atom.acq_rel.sys.global.add.u32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(old)


@dsl_user_op
def atomic_exch_release_sys(
    ptr: cute.Pointer, val: Int32, *, loc=None, ip=None
) -> None:
    """System-scope release exchange used to reset a grid-barrier counter."""
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        T.i32(),
        [ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "atom.release.sys.global.exch.b32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_acquire_sys(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """System-scope acquire load for the sense-reversing grid barrier."""
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [ptr_i64],
        "ld.acquire.sys.global.u32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(state)


@cute.jit
def sense_reversing_grid_barrier(
    barrier: cute.Tensor,
    participant_count: Int32,
    thread_idx: Int32,
) -> None:
    """Barrier a fully resident grid using int32 ``[count, generation]`` state.

    Every CTA calls this with all threads, but only thread 0 contributes one
    arrival. The final participant resets count before publishing the next
    generation; waiters observe generation with acquire semantics.
    """
    cute.arch.sync_threads()
    if thread_idx == 0:
        count_ptr = barrier.iterator
        generation_ptr = barrier.iterator + 1
        generation = ld_acquire_sys(generation_ptr)
        old_count = atomic_add_acq_rel_sys(count_ptr, Int32(1))
        if old_count == participant_count - 1:
            atomic_exch_release_sys(count_ptr, Int32(0))
            atomic_add_acq_rel_sys(generation_ptr, Int32(1))
        else:
            observed_generation = generation
            while observed_generation == generation:
                observed_generation = ld_acquire_sys(generation_ptr)
    cute.arch.sync_threads()


@dsl_user_op
def multimem_red_add_release_sys(
    mc_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    """Push-signal: atomic-add `val` to a MULTICAST address, broadcasting the increment
    to every rank's copy in-switch (NVLS). Lets a kv-block owner poll its LOCAL counter
    (cheap ld.acquire) instead of pulling a cross-rank multimem.ld_reduce each spin.
    .release.sys orders the producer's dK/dV writes before the signal is visible."""
    mc_ptr_i64 = mc_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [mc_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "multimem.red.release.sys.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_eq(lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        read_val = Int32(0)
        while read_val != val:
            read_val = ld_acquire(flag_ptr)


@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: cutlass.Constexpr[Int32]
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        red_release(flag_ptr, val)
        # red_relaxed(flag_ptr, val)
