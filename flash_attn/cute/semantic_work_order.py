"""Lower sparse-Q-block work IDs to smaller native forward tiles.

This preserves logical phase/batch/head/order. It never drops an order supplied
by the caller. Preparation is cached on the source tensor, outside graph capture.
"""
import torch


def lower_semantic_work_order(mask, *, batch_size, num_head, seqlen_q,
                              native_block_m, packed_heads):
    order = mask.fwd_work_order
    logical_block_m = mask.block_size[0]
    if order is None or logical_block_m == native_block_m:
        return mask
    if logical_block_m % native_block_m:
        raise ValueError('Native Q tile must divide the logical sparse Q block')
    logical_rows = (seqlen_q * packed_heads + logical_block_m - 1) // logical_block_m
    native_rows = (seqlen_q * packed_heads + native_block_m - 1) // native_block_m
    phases = 3 if mask.intra_mask_block_cnt is not None else 2
    groups = phases * batch_size * (num_head // packed_heads)
    # Direct FA callers may already supply an explicitly native work grid.
    if order.numel() == groups * native_rows:
        return mask
    if order.numel() != groups * logical_rows:
        raise ValueError('Forward order must cover the logical or native work grid')
    try:
        version = order._version
    except RuntimeError:
        version = None
    key = (version, logical_rows, native_rows, groups, logical_block_m, native_block_m)
    cached = getattr(order, '_semantic_native_order', None)
    if cached is not None and cached[0] == key:
        return mask._replace(fwd_work_order=cached[1])
    if order.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Warm up semantic work-order lowering before graph capture')
    ids = order.detach().cpu().tolist()
    if sorted(ids) != list(range(groups * logical_rows)):
        raise ValueError('Forward work order must be a permutation')
    factor = logical_block_m // native_block_m
    native = []
    for work in ids:
        group, row = divmod(work, logical_rows)
        native.extend(group * native_rows + sub
                      for sub in range(row * factor, min((row + 1) * factor, native_rows)))
    result = torch.tensor(native, device=order.device, dtype=order.dtype)
    order._semantic_native_order = (key, result)
    return mask._replace(fwd_work_order=result)
