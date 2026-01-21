# Arbitrary Mask for Flash Attention

Arbitrary Mask enables flexible, user-defined masking patterns for the attention computation (Q @ K^T) in Flash Attention. This document describes how to define, generate, and use arbitrary masks.

## Overview

In standard attention computation, masking is typically limited to predefined patterns like causal or sliding window. Arbitrary Mask extends this capability by allowing users to specify **any valid region** for each query token using a compact interval-based representation called the **Func Tensor**.

## Func Tensor Definition

### Tensor Shape

The Func Tensor has shape `[batch_size, nheads, nFunc, seqlen_q + 256]`:

| Dimension | Description |
|-----------|-------------|
| `batch_size` | Batch dimension. Can be 1 (shared across batches) or equal to the QKV batch size |
| `nheads` | Number of attention heads. Can be 1 (shared across heads) or equal to Q's head count |
| `nFunc` | Number of interval descriptors. **Must be an odd number** (see below) |
| `seqlen_q + 256` | Query sequence length with 256-element padding to avoid boundary checks in kernels |

**Broadcasting**: If all sequences in a batch share the same mask, set `batch_size=1`. Similarly, set `nheads=1` if all heads share the same mask.

### nFunc: Interval-Based Mask Representation

For each query position `i`, the Func Tensor encodes which key positions `j` are valid using **intervals**:

Given `nFunc` values `[F0, F1, F2, ..., F_{nFunc-1}]` at position `i`:
- **Base interval**: `j ∈ [0, F0)` is valid
- **Additional intervals**: For `k = 1, 2, ...`, `j ∈ [F_{2k-1}, F_{2k})` is valid

The total valid region is the **union** of all these intervals.

**Formula**: If a sequence has at most `n` disjoint valid intervals per query token, then `nFunc = 2n - 1`.

**Note**: Set `nFunc` as small as possible to get best performance of attention kernel.

### Examples

#### Example 1: Causal Mask

For causal attention, each query at position `i` attends to keys `j ∈ [0, i+1)`.

```python
# nFunc = 1 (single interval)
# Func[b, h, 0, i] = i + 1
arbitrary_func = torch.zeros(1, 1, 1, seqlen_q + 256, dtype=torch.int32, device="cuda")
for i in range(seqlen_q):
    arbitrary_func[:, :, 0, i] = i + 1
```

Visualization (seqlen=8):
```
Query 0: [0, 1) → attends to key 0
Query 1: [0, 2) → attends to keys 0, 1
Query 2: [0, 3) → attends to keys 0, 1, 2
...
```

#### Example 2: Causal Mask with variable length

For variable length, set func for different sequences separately.

```python
arbitrary_func = torch.zeros(b, 1, 1, seqlen_q + 256, dtype=torch.int32, device="cuda")
for i in range(b):
    for j in range(actual_seqlen_q[i]):
        arbitrary_func[i, :, 0, j] = j + 1
```

#### Example 3: Sink + Local Window

A common pattern combines "sink tokens" (always attended) with a local sliding window:

<img src="sink&local.png" alt="Sink + Local Window" width="300">

```python
# nFunc = 3: base interval [0, F0) + one additional interval [F1, F2)
# - Sink tokens: positions [0, sink_width)
# - Local window: positions [i - window_left, i + 1 + window_right)

def create_sink_local_func(seqlen_q, seqlen_k, sink_width, window_left, window_right):
    arbitrary_func = torch.zeros(1, 1, 3, seqlen_q + 256, dtype=torch.int32, device="cuda")
    hole_flag = False
    for i in range(seqlen_q):
        if not hole_flag:
            # No gap yet: single interval covers both sink and local
            arbitrary_func[:, :, 0, i] = min(i + 1 + window_right, seqlen_k)
            if i - window_left >= sink_width:
                hole_flag = True
        else:
            # Gap exists: need two intervals
            arbitrary_func[:, :, 0, i] = sink_width  # Sink: [0, sink_width)
            arbitrary_func[:, :, 1, i] = min(i - window_left, seqlen_k)  # Local start
            arbitrary_func[:, :, 2, i] = min(i + 1 + window_right, seqlen_k)  # Local end
    return arbitrary_func
```

## Block Sparsity Optimization

Since the Func Tensor is known before kernel execution, we can precompute **block sparsity** information to skip computation for blocks that are entirely masked out.

### Block Sparse Tensors

The `LinearBlockSparseTensorsTorch` structure stores block-level sparsity information:

```python
class LinearBlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor    # [batch, nheads, num_blocks] - count of partial blocks per row
    mask_block_offset: torch.Tensor # [batch * nheads * num_blocks + 1] - CSR-style offsets
    mask_block_idx: torch.Tensor    # [total_partial_blocks] - indices of partial blocks
    full_block_cnt: torch.Tensor    # [batch, nheads, num_blocks] - count of full blocks per row
    full_block_offset: torch.Tensor # [batch * nheads * num_blocks + 1] - CSR-style offsets
    full_block_idx: torch.Tensor    # [total_full_blocks] - indices of full blocks
```

**Block types**:
- **Full blocks**: Entirely within valid regions (no masking needed)
- **Partial blocks**: Partially valid (require per-element masking)
- **Empty blocks**: Entirely masked (skipped completely)

### Forward vs Backward Iteration

- **Forward pass**: Each CTA processes a fixed `m_block` (query block) and iterates over `n_blocks` (key blocks)
- **Backward pass**: Each CTA processes a fixed `n_block` (key block) and iterates over `m_blocks` (query blocks)

This requires two separate block sparsity structures:
- `linear_k_block_sparse_tensors`: For forward (Q2K direction)
- `linear_q_block_sparse_tensors`: For backward (K2Q direction)

## Generating Block Sparsity from Func Tensor

We provide CUDA kernels to efficiently generate block sparsity from the Func Tensor.

### CUDA Kernel API

```python
import create_block_mask_cuda

# Forward: Q2K (iterate over K blocks for each Q block)
(k_mask_cnt, k_mask_offset, k_mask_idx,
 k_full_cnt, k_full_offset, k_full_idx) = \
    create_block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func,    # [batch, nheads, nFunc, seqlen_q + 256]
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=128,  # Must match kernel tile size
        KV_BLOCK_SIZE=128,
        check_q_boundary=False
    )

# Backward: K2Q (iterate over Q blocks for each K block)
(q_mask_cnt, q_mask_offset, q_mask_idx,
 q_full_cnt, q_full_offset, q_full_idx) = \
    create_block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=128,  # May differ from forward
        KV_BLOCK_SIZE=128
    )
```

### PyTorch Fallback

If the CUDA kernel is unavailable, you can use PyTorch's `create_block_mask` from Flex Attention:

```python
from torch.nn.attention.flex_attention import create_block_mask

# Define mask function compatible with Flex Attention
def mask_mod_flex(b, h, q_idx, kv_idx, arbitrary_func=arbitrary_func):
    zero = h * 0
    value_valid = kv_idx < arbitrary_func[b, h, zero, q_idx]
    n_func = arbitrary_func.shape[2]
    for i in range(n_func // 2):
        in_range = (kv_idx >= arbitrary_func[b, h, zero + (2*i+1), q_idx]) & \
                   (kv_idx < arbitrary_func[b, h, zero + (2*i+2), q_idx])
        value_valid = value_valid | in_range
    return value_valid

# Create block mask
bm = create_block_mask(
    mask_mod_flex,
    batch_size, nheads, seqlen_q, seqlen_k,
    device="cuda",
    BLOCK_SIZE=(tile_m, tile_n)
)
```

## Installation

```bash
bash install.sh
```

**Required components**:
1. Python packages in `requirements.txt`
2. CUDA kernels in `csrc/utils/create_block_mask/`

If `flash_attn_cute` installation fails, you can still run by setting `PYTHONPATH` explicitly.

To build only the block mask utility:

```bash
cd csrc/utils
make create_block_mask
```

## Usage Example

### Complete Workflow

```python
import torch
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.block_sparsity import LinearBlockSparseTensorsTorch
import create_block_mask_cuda

# Configuration
batch_size, seqlen_q, seqlen_k = 1, 8192, 8192
nheads, headdim = 32, 128
dtype = torch.bfloat16

# Create input tensors
q = torch.randn(batch_size, seqlen_q, nheads, headdim, device="cuda", dtype=dtype).requires_grad_(True)
k = torch.randn(batch_size, seqlen_k, nheads, headdim, device="cuda", dtype=dtype).requires_grad_(True)
v = torch.randn(batch_size, seqlen_k, nheads, headdim, device="cuda", dtype=dtype).requires_grad_(True)

# Create arbitrary func tensor (causal pattern example)
nFunc = 3  # Must be odd
arbitrary_func = torch.zeros(1, 1, nFunc, seqlen_q + 256, dtype=torch.int32, device="cuda")
for i in range(seqlen_q):
    arbitrary_func[:, :, 0, i] = i + 1  # Causal: attend to [0, i+1)

# Generate block sparsity (Forward: Q2K)
tile_m, tile_n = 128, 128  # Match kernel tile sizes
(k_mask_cnt, k_mask_offset, k_mask_idx,
 k_full_cnt, k_full_offset, k_full_idx) = \
    create_block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func, seqlen_q, seqlen_k,
        Q_BLOCK_SIZE=tile_m * 2,  # SM100: 2x tile_m; SM90: tile_m
        KV_BLOCK_SIZE=tile_n
    )

linear_k_block_sparse = LinearBlockSparseTensorsTorch(
    mask_block_cnt=k_mask_cnt,
    mask_block_offset=k_mask_offset,
    mask_block_idx=k_mask_idx,
    full_block_cnt=k_full_cnt,
    full_block_offset=k_full_offset,
    full_block_idx=k_full_idx,
)

# Generate block sparsity (Backward: K2Q)
(q_mask_cnt, q_mask_offset, q_mask_idx,
 q_full_cnt, q_full_offset, q_full_idx) = \
    create_block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func, seqlen_q, seqlen_k,
        Q_BLOCK_SIZE=128,  # SM100: 128; SM90: 64
        KV_BLOCK_SIZE=128
    )

linear_q_block_sparse = LinearBlockSparseTensorsTorch(
    mask_block_cnt=q_mask_cnt,
    mask_block_offset=q_mask_offset,
    mask_block_idx=q_mask_idx,
    full_block_cnt=q_full_cnt,
    full_block_offset=q_full_offset,
    full_block_idx=q_full_idx,
)

# Run attention with arbitrary mask
out, lse = flash_attn_func(
    q=q,
    k=k,
    v=v,
    softmax_scale=1.0 / (headdim ** 0.5),
    causal=False,           # Disable built-in causal (using arbitrary instead)
    arbitrary=True,         # Enable arbitrary mask mode
    window_size=(None, None),
    linear_k_block_sparse_tensors=linear_k_block_sparse,
    linear_q_block_sparse_tensors=linear_q_block_sparse,
    aux_tensors=[arbitrary_func],  # Pass func tensor for element-wise masking
)

# Backward pass works automatically
out.sum().backward()
```

### Hardware-Specific Block Sizes

| GPU Architecture | Forward `Q_BLOCK_SIZE` | Backward `Q_BLOCK_SIZE` | `KV_BLOCK_SIZE` |
|------------------|------------------------|-------------------------|-----------------|
| SM100 (Blackwell) | `2 * tile_m` (256) | 128 | 128 |
| SM90 (Hopper) | `tile_m` (128) | 64 | 128 |

### Supported Attention Modes

| Mode | Description | `nheads_kv` |
|------|-------------|-------------|
| MHA | Multi-Head Attention | `nheads_kv = nheads` |
| GQA | Grouped Query Attention | `nheads_kv = nheads // group_size` |
| MQA | Multi-Query Attention | `nheads_kv = 1` |

## API Reference

### `flash_attn_func`

```python
def flash_attn_func(
    q: torch.Tensor,                    # [B, seqlen_q, H, D]
    k: torch.Tensor,                    # [B, seqlen_k, H_kv, D]
    v: torch.Tensor,                    # [B, seqlen_k, H_kv, D]
    softmax_scale: Optional[float],     # Default: 1/sqrt(D)
    causal: bool = False,
    arbitrary: bool = False,            # Enable arbitrary mask
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    linear_k_block_sparse_tensors: Optional[LinearBlockSparseTensorsTorch] = None,
    linear_q_block_sparse_tensors: Optional[LinearBlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,  # [arbitrary_func]
) -> Tuple[torch.Tensor, torch.Tensor]  # (output, lse)
```

**Key parameters for arbitrary mask**:
- `arbitrary=True`: Enables arbitrary mask mode
- `aux_tensors=[arbitrary_func]`: Passes the Func Tensor
- `linear_k_block_sparse_tensors`: Block sparsity for forward pass
- `linear_q_block_sparse_tensors`: Block sparsity for backward pass

## Additional Resources

- **Test file**: `tests/cute/test_arbitrary_mask.py` - Comprehensive tests and benchmarks
- **Mask definitions**: `flash_attn/cute/mask_definitions.py` - Helper functions for creating Func Tensors
- **CUDA kernel source**: `csrc/utils/create_block_mask/` - Block sparsity generation
