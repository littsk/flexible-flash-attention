# create_block_mask

CUDA extension for converting arbitrary interval-mask metadata into compact block-sparse CSR tensors.

Build:

```bash
cd csrc/utils/create_block_mask
python setup.py build_ext --inplace
```

Forward uses Q2K metadata and backward uses K2Q metadata:

```python
import create_block_mask_cuda
from flash_attn.cute import flash_attn_func
from flash_attn.cute.block_sparsity import LinearBlockSparseTensorsTorch

q2k = create_block_mask_cuda.create_q2k_csr_sparse_from_func(
    arbitrary_func, seqlen_q, seqlen_k, q_block_size, kv_block_size
)
k2q = create_block_mask_cuda.create_k2q_csr_sparse_from_func(
    arbitrary_func, seqlen_q, seqlen_k, q_block_size_bwd, kv_block_size_bwd
)

linear_k = LinearBlockSparseTensorsTorch(*q2k, block_size=(q_block_size, kv_block_size))
linear_q = LinearBlockSparseTensorsTorch(
    *k2q, block_size=(q_block_size_bwd, kv_block_size_bwd)
)

out, lse = flash_attn_func(
    q,
    k,
    v,
    arbitrary=True,
    aux_tensors=[arbitrary_func],
    linear_k_block_sparse_tensors=linear_k,
    linear_q_block_sparse_tensors=linear_q,
    return_lse=True,
)
```

The CSR tuple order is:

```python
(
    mask_block_cnt,
    mask_block_offset,
    mask_block_idx,
    full_block_cnt,
    full_block_offset,
    full_block_idx,
)
```

Use `create_q2k_csr_sparse_auto` and `create_k2q_csr_sparse_auto` to get the selected tile sizes back with the CSR tensors.
