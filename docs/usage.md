# Arbitrary Mask for Flash Attention

Arbitrary Mask enables flexible, user-defined masking patterns for the attention computation (Q @ K^T) in Flash Attention. This document describes how to define, generate, and use arbitrary masks.

## Overview

In standard attention , mask is typically limited to predefined patterns like causal or sliding window. Arbitrary Mask extends this capability by allowing users to specify **any valid region** for each query token using a compact interval-based representation called the **Func Tensor**.

## Quick Start

Get started quickly by running the provided test scripts.

### 1. Install Dependencies

```bash
cd flash-attention
make install
```

### 2. Run Tests

**Run with CUTE DSL backend (default, for Blackwell/Hopper):**
```bash
cd flash-attention
make tt                    # Run default test
# or
python tests/cute/test_arbitrary_mask.py
```

**Run with Cutlass C++ backend (for SM8x Ampere/Ada and SM90 Hopper in Hopper folder):** 
```bash
cd flash-attention/hopper
make install ARBITRARY=1 NUM_FUNC=3 HDIM128=1 SM8X=1 (if ampere/ada)
make tt                    # Run test
# or
FLASH_ATTN_BACKEND=hopper python tests/cute/test_arbitrary_mask.py
```




### 3. Minimal Code Example

> **Note**: This example uses DSL backend. On SM90, DSL only supports `headdim <= 128` and MHA mode.

```python
import torch
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.block_sparsity import LinearBlockSparseTensorsTorch
from flash_attn.utils.tile_size import get_fwd_tile_sizes_dsl, get_bwd_tile_sizes_dsl, get_arch
from flash_attn.cute.mask_definitions import arbitrary_func_tensor
import create_block_mask_cuda

# Config (MHA mode: nheads == nheads_kv)
batch, seqlen, nheads, headdim = 1, 4096, 32, 128
n_func = 3  # Must be odd

# Create tensors (requires_grad=True enables backward pass and returns valid lse)
q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

# Create arbitrary func (causal pattern, broadcast across batch and heads)
arbitrary_func = arbitrary_func_tensor(1, 1, n_func, seqlen, seqlen, device="cuda", pattern="causal")

# Get tile sizes for DSL backend
arch = get_arch()
fwd_q, fwd_kv = get_fwd_tile_sizes_dsl(arch=arch)  # (128, 128) SM90 / (256, 128) SM100
bwd_q, bwd_kv = get_bwd_tile_sizes_dsl(arch=arch, is_arbitrary=True)  # (64, 128) SM90 / (128, 128) SM100

# Generate block sparsity
# q2k for forward
k_cnt, k_off, k_idx, k_fcnt, k_foff, k_fidx = create_block_mask_cuda.create_q2k_csr_sparse_from_func(
    arbitrary_func, seqlen, seqlen, Q_BLOCK_SIZE=fwd_q, KV_BLOCK_SIZE=fwd_kv, check_q_boundary=True)
# k2q for backward
q_cnt, q_off, q_idx, q_fcnt, q_foff, q_fidx = create_block_mask_cuda.create_k2q_csr_sparse_from_func(
    arbitrary_func, seqlen, seqlen, Q_BLOCK_SIZE=bwd_q, KV_BLOCK_SIZE=bwd_kv)

linear_k = LinearBlockSparseTensorsTorch(k_cnt, k_off, k_idx, k_fcnt, k_foff, k_fidx)
linear_q = LinearBlockSparseTensorsTorch(q_cnt, q_off, q_idx, q_fcnt, q_foff, q_fidx)

# Run attention
out, lse = flash_attn_func(
    q, k, v, causal=False, arbitrary=True,
    linear_k_block_sparse_tensors=linear_k,
    linear_q_block_sparse_tensors=linear_q,
    aux_tensors=[arbitrary_func],
)
```

## Func Tensor Definition

### Tensor Shape

The Func Tensor has shape `[batch_size, nheads_q, nFunc, seqlen_q + 256]`:

| Dimension | Description |
|-----------|-------------|
| `batch_size` | Batch dimension. Supports broadcasting: set to 1 to share mask across all batches |
| `nheads_q` | Number of **query** attention heads. Supports broadcasting: set to 1 to share mask across all heads |
| `nFunc` | Number of interval descriptors. **Must be an odd number** (see below) |
| `seqlen_q + 256` | Query sequence length with 256-element padding to avoid boundary checks in kernels |

**Broadcasting**: 
- If all sequences in a batch share the same mask, set `batch_size=1`
- If all query heads share the same mask, set `nheads_q=1`

### nFunc: Interval-Based Mask Representation

For each query position `i`, the Func Tensor encodes which key positions `j` are valid using **intervals**:

Given `nFunc` values `[F0, F1, F2, ..., F_{nFunc-1}]` at position `i`:
- **Base interval**: `j ∈ [0, F0)` is valid
- **Additional intervals**: For `k = 1, 2, ...`, `j ∈ [F_{2k-1}, F_{2k})` is valid

The total valid region is the **union** of all these intervals.

**Formula**: If a sequence has at most `n` disjoint valid intervals per query token, then `nFunc = 2n - 1`.

**Note**: Set `nFunc` as small as possible to get best performance of attention kernel.

### Examples

> **Tip: Visualize Your Mask**
> 
> Use `visualize_arbitrary_mask` to verify your custom func tensor produces the expected mask pattern:
> ```python
> from tests.cute.test_arbitrary_mask import visualize_arbitrary_mask
> visualize_arbitrary_mask(arbitrary_func, seqlen_q, seqlen_k)
> ```
> 
> To see all examples below visualized, run:
> ```bash
> cd flash-attention
> python -c "from tests.cute.test_arbitrary_mask import demo_mask_examples; demo_mask_examples()"
> ```

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

#### Example 2: Causal Mask with Variable Length

For variable-length sequences with **separate batches**, set func for each sequence separately:

```python
# Each batch has its own causal mask based on actual sequence length
# actual_seqlen_q[i] is the actual length of sequence i
arbitrary_func = torch.zeros(batch_size, 1, 1, max_seqlen_q + 256, dtype=torch.int32, device="cuda")
for i in range(batch_size):
    for j in range(actual_seqlen_q[i]):
        arbitrary_func[i, :, 0, j] = j + 1  # Causal: attend to [0, j+1)
```

We can also treat variable-length sequences as a **single packed sequence** and implement it using a mask, like this:

<img src="varlen_as_fixedlen_mask.png" alt="Variable-length as fixed-length with mask" width="300">

Given `cu_seqlens_qk`: cumulative sequence lengths of shape `(batch_size + 1,)`, each token only attends to tokens within the same sequence with causal masking:

```python
# Pack multiple batch_size into one, each with causal attention within its boundaries
total_seqlen = cu_seqlens_qk[-1]
batch_size = len(cu_seqlens_qk) - 1

arbitrary_func = torch.zeros(1, 1, 3, total_seqlen + 256, dtype=torch.int32, device="cuda")
for i in range(batch_size):
    seq_start = cu_seqlens_qk[i]
    seq_end = cu_seqlens_qk[i + 1]
    for s in range(seq_end - seq_start):
        q_token = seq_start + s
        # Causal within sequence: attend to [seq_start, q_token + 1)
        # Using nFunc=3: [0, F0) ∪ [F1, F2) = [0, 0) ∪ [seq_start, q_token + 1)
        arbitrary_func[0, 0, 0, q_token] = 0           # F0: base interval is empty
        arbitrary_func[0, 0, 1, q_token] = seq_start   # F1: sequence start
        arbitrary_func[0, 0, 2, q_token] = q_token + 1 # F2: current position + 1 (exclusive)
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

#### Example 4: HSTU - Causal + Context + Target

Mask pattern for Generative Recommendation scenarios (e.g., HSTU model). The sequence is divided into three parts:
- **Context tokens**: Can attend to all context tokens (full attention within context)
- **Causal tokens**: Causal attention (can see context + previous causal tokens)
- **Target tokens**: Can attend to context + itself (no cross-attention between targets)

<img src="hstu_causal_context_target.png" alt="HSTU: Causal + Context + Target" width="300">

```python
# Sequence structure: [context (kv_context)] [causal (q_causal)] [target (q_target)]
# kv_context: number of context tokens in K/V
# q_context: number of context tokens in Q (typically equals kv_context)
# q_causal: number of causal tokens
# q_target: number of target tokens

seqlen_q = q_context + q_causal + q_target
arbitrary_func = torch.zeros(1, 1, 3, seqlen_q + 256, dtype=torch.int32, device="cuda")

# Context tokens: attend to all context [0, kv_context)
for c in range(q_context):
    arbitrary_func[:, :, 0, c] = kv_context  # [0, kv_context)

# Causal tokens: attend to context + previous causal [0, current_pos + 1)
for ca in range(q_causal):
    pos = q_context + ca
    arbitrary_func[:, :, 0, pos] = pos + 1  # [0, pos + 1) includes context and causal up to current

# Target tokens: attend to context + itself [0, kv_context) ∪ [pos, pos + 1)
for t in range(q_target):
    pos = q_context + q_causal + t
    # Using nFunc=3: [0, F0) ∪ [F1, F2) = [0, kv_context) ∪ [pos, pos + 1)
    arbitrary_func[:, :, 0, pos] = kv_context  # F0: attend to context
    arbitrary_func[:, :, 1, pos] = pos         # F1: self position start
    arbitrary_func[:, :, 2, pos] = pos + 1     # F2: self position end
```


## Block Sparsity

Since the Func Tensor is known before kernel execution, we can precompute **block sparsity** information to skip computation for blocks that are entirely masked out.

### Block Sparse Tensors (CSR Format)

The `LinearBlockSparseTensorsTorch` structure stores block-level sparsity information using **CSR (Compressed Sparse Row) format** for efficient memory access:

```python
class LinearBlockSparseTensorsTorch(NamedTuple):
    # Partial blocks (require per-element masking)
    mask_block_cnt: torch.Tensor    # [batch, nheads, num_q_blocks] - count of partial blocks per row
    mask_block_offset: torch.Tensor # [batch * nheads * num_q_blocks + 1] - CSR offsets (prefix sum)
    mask_block_idx: torch.Tensor    # [total_partial_blocks] - column indices of partial blocks
    
    # Full blocks (no masking needed, entirely valid)
    full_block_cnt: torch.Tensor    # [batch, nheads, num_q_blocks] - count of full blocks per row
    full_block_offset: torch.Tensor # [batch * nheads * num_q_blocks + 1] - CSR offsets (prefix sum)
    full_block_idx: torch.Tensor    # [total_full_blocks] - column indices of full blocks
```

**Block types**:
- **Full blocks**: Entirely within valid regions (no masking needed)
- **Partial blocks**: Partially valid (require per-element masking using Func Tensor)
- **Empty blocks**: Entirely masked (skipped completely)

### Forward vs Backward Iteration

- **Forward pass (Q2K)**: Each CTA processes a fixed `m_block` (query block) and iterates over `n_blocks` (key blocks)
- **Backward pass (K2Q)**: Each CTA processes a fixed `n_block` (key block) and iterates over `m_blocks` (query blocks)

This requires two separate block sparsity structures:
- `linear_k_block_sparse_tensors`: For forward (Q2K direction)
- `linear_q_block_sparse_tensors`: For backward (K2Q direction)

### Automatic Block Size Detection

**Important**: Block sparsity granularity must match the attention kernel's tile size. 

**DSL and C++ backends use DIFFERENT tile sizes:**

| Backend | Forward Tile Size | Backward Tile Size | Source |
|---------|-------------------|-------------------|--------|
| **CUTE DSL** | (128, 128) SM90 / (256, 128) SM100 | (64, 128) SM90 / (128, 128) SM100 | `interface.py` |
| **Cutlass C++** (`hopper/`) | Varies by headdim | Varies by headdim | `tile_size.h` |

Use the appropriate function based on your backend:

```python
from flash_attn.utils.tile_size import (
    get_fwd_tile_sizes_dsl, get_bwd_tile_sizes_dsl,  # For DSL backend
    get_fwd_tile_sizes, get_bwd_tile_sizes,          # For C++ backend
    get_tile_sizes_by_backend,                        # Auto-select by backend
    get_arch,
)

arch = get_arch()  # e.g., 90 for Hopper, 100 for Blackwell

# Option 1: Use backend-specific functions
# For DSL backend (cute):
fwd_q, fwd_kv = get_fwd_tile_sizes_dsl(arch=arch)             # (128, 128) SM90 / (256, 128) SM100
bwd_q, bwd_kv = get_bwd_tile_sizes_dsl(arch=arch, is_arbitrary=True)  # (64, 128) SM90 / (128, 128) SM100

# For C++ backend (hopper):
fwd_q, fwd_kv = get_fwd_tile_sizes(arch=arch, headdim=headdim, is_arbitrary=True)
bwd_q, bwd_kv = get_bwd_tile_sizes(arch=arch, headdim=headdim, is_arbitrary=True)

# Option 2: Use unified function with backend parameter
fwd_q, fwd_kv = get_tile_sizes_by_backend(
    backend="cute",  # or "hopper"
    pass_type="forward",
    arch=arch, headdim=headdim, is_arbitrary=True
)
bwd_q, bwd_kv = get_tile_sizes_by_backend(
    backend="cute",  # or "hopper"
    pass_type="backward",
    arch=arch, headdim=headdim, is_arbitrary=True
)
```

**Users do not need to manually set block sizes.** The utility functions handle all backend and architecture-specific configurations automatically.

## Architecture Support Summary

| Architecture | Backend | headdim Support | Attention Mode | Notes |
|-------------|---------|-----------------|----------------|-------|
| Blackwell (SM100) | CUTE DSL | 64, 128 | MHA, GQA, MQA | Recommended |
| Hopper (SM90) | CUTE DSL | ≤128 | **MHA only** | GQA/MQA not supported |
| SM8x/SM90 | Cutlass C++ | 64, 128, 256 | MHA, GQA, MQA | Full feature support |
| Ampere/Ada (SM8x) | C++ | 64, 128, 256 | MHA, GQA, MQA | Requires `SM8X=1` in Compile |

## Installation

### Blackwell (SM100) Architecture

Use the **CUTE DSL** version. Install dependencies from the flash-attention root directory:

```bash
cd flash-attention
make install
```

This installs:
- CUTLASS DSL library
- TVM FFI runtime
- `create_block_mask_cuda` kernel

### Hopper (SM90) Architecture

Two versions are available:

**Option 1: CUTE DSL version**
```bash
cd flash-attention
make install
```

**Note**: CUTE DSL version on SM90 has limitations:
- `headdim <= 128`
- **MHA only** (`num_head == num_head_kv`). GQA/MQA not supported.

**Option 2: CUTLASS C++ version** (supports all headdims and attention modes)
```bash
cd flash-attention/hopper
make install
```

### Ampere/Ada (SM80/SM86/SM89) Architecture

Use the **CUTLASS C++ version**:

```bash
cd flash-attention/hopper
make install SM8X=1
```

### Makefile Configuration (hopper/Makefile)

The Makefile uses `FLAG=1` to enable and `FLAG=0` to disable features. 

> **Compilation Time Tip**: Each enabled feature generates additional kernel templates. To minimize compilation time, **only enable features you actually need**. A minimal build can complete in minutes, while a full build may take hours.

#### Core Features

| Flag | Default | Description |
|------|---------|-------------|
| `BACKWARD` | **1** | Backward pass kernels. Required for training. Set `=0` for inference-only builds. |
| `ARBITRARY` | **1** | Arbitrary mask / block sparsity support. |
| `NUM_FUNC` | **3** | nFunc values to compile (odd numbers 1-33). Use comma-separated list: `NUM_FUNC=3,5,9`. More values = slower compile. |

#### Attention Variants (default: disabled)

| Flag | Default | Description |
|------|---------|-------------|
| `SPLIT` | 0 | Split-KV attention for short q seq and long kv seq in inference. |
| `PAGEDKV` | 0 | Paged KV cache for inference. |
| `APPENDKV` | 0 | Append KV mode for incremental decoding. |
| `LOCAL` | 0 | Local (sliding window) attention. |
| `SOFTCAP` | 0 | Softmax capping (used by Gemma 2, etc.). |
| `PACKGQA` | 0 | Packed GQA implementation. |
| `VARLEN` | 0 | Variable-length sequence support (via cu_seqlens). |
| `CLUSTER` | 0 | Hopper cluster feature (SM90+ TMA multicast). |

#### Data Types (default: BF16 only)

| Flag | Default | Description |
|------|---------|-------------|
| `FP16` | 0 | Enable FP16 (float16) support. |
| `FP8` | 0 | Enable FP8 (float8_e4m3fn) support. |

#### Head Dimensions

| Flag | Default | Description |
|------|---------|-------------|
| `HDIM64` | 0 | Enable head_dim=64. |
| `HDIM96` | 0 | Enable head_dim=96. |
| `HDIM128` | **1** | Enable head_dim=128 (most common). |
| `HDIM192` | 0 | Enable head_dim=192. |
| `HDIM256` | 0 | Enable head_dim=256. |
| `HDIMDIFF64` | 0 | Enable head_dim_k ≠ head_dim_v with dim=64. |
| `HDIMDIFF192` | 0 | Enable head_dim_k ≠ head_dim_v with dim=192. |

#### GPU Architecture

| Flag | Default | Description |
|------|---------|-------------|
| `SM8X` | **1** | SM80/86/89 (Ampere A100, Ada L40, RTX 4090, etc.). Set `=0` for SM90-only builds. |
| `SM90` | **1** | SM90 (Hopper H100). Set `=0` for SM8x-only builds. |

> **Note**: At least one of `SM8X` or `SM90` must be enabled. Disabling the architecture you don't have can reduce compilation time by ~50%.

#### Advanced Options

| Flag | Default | Description |
|------|---------|-------------|
| `VCOLMAJOR` | 0 | V tensor column-major layout. |

#### Build Examples

```bash
cd hopper

# View current build configuration
make show_flags

# SM8x only (A100, L40/20, etc.) - skip Hopper kernels
make install SM90=0
# SM90 only (H100) - skip Ampere/Ada kernels  
make install SM8X=0

# Training with arbitrary mask
make install BACKWARD=1 ARBITRARY=1 NUM_FUNC=3,5 SM8X=1 SM90=1

```

### Build Only Block Mask Utility

If you only need the block mask generation kernel:

```bash
cd csrc/utils
make create_block_mask
```

## Generating Block Sparsity from Func Tensor

We provide CUDA kernels to efficiently generate block sparsity from the Func Tensor.

### CUDA Kernel API

```python
import create_block_mask_cuda
from flash_attn.utils.tile_size import get_fwd_tile_sizes, get_bwd_tile_sizes, get_arch

# Get automatic tile sizes
arch = get_arch()
fwd_q_block, fwd_kv_block = get_fwd_tile_sizes(arch=arch, headdim=headdim, is_arbitrary=True)
bwd_q_block, bwd_kv_block = get_bwd_tile_sizes(arch=arch, headdim=headdim, is_arbitrary=True)

# Forward: Q2K (iterate over K blocks for each Q block)
(k_mask_cnt, k_mask_offset, k_mask_idx,
 k_full_cnt, k_full_offset, k_full_idx) = \
    create_block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func,    # [batch, nheads_q, nFunc, seqlen_q + 256]
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=fwd_q_block,
        KV_BLOCK_SIZE=fwd_kv_block,
        check_q_boundary=True
    )

# Backward: K2Q (iterate over Q blocks for each K block)
(q_mask_cnt, q_mask_offset, q_mask_idx,
 q_full_cnt, q_full_offset, q_full_idx) = \
    create_block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=bwd_q_block,
        KV_BLOCK_SIZE=bwd_kv_block
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
    batch_size, nheads_q, seqlen_q, seqlen_k,
    device="cuda",
    BLOCK_SIZE=(fwd_q_block, fwd_kv_block)
)
```

## Usage Example

### Complete Workflow (CUTE DSL Backend)

> **SM90 Limitations**: DSL backend on Hopper only supports `headdim <= 128` and MHA mode (`nheads == nheads_kv`).

```python
import torch
import math
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.block_sparsity import LinearBlockSparseTensorsTorch
from flash_attn.utils.tile_size import get_fwd_tile_sizes_dsl, get_bwd_tile_sizes_dsl, get_arch
from flash_attn.cute.mask_definitions import arbitrary_func_tensor
import create_block_mask_cuda

# =============================================================================
# Configuration (SM90 DSL requires MHA mode: nheads == nheads_kv)
# =============================================================================
batch_size, seqlen_q, seqlen_k = 1, 8192, 8192
nheads, nheads_kv, headdim = 32, 32, 128  # MHA mode (required for SM90 DSL)
dtype = torch.bfloat16
n_func = 3  # Must be odd (supports up to (n_func+1)/2 disjoint intervals)

# =============================================================================
# Create input tensors
# =============================================================================
q = torch.randn(batch_size, seqlen_q, nheads, headdim, device="cuda", dtype=dtype).requires_grad_(True)
k = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device="cuda", dtype=dtype).requires_grad_(True)
v = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device="cuda", dtype=dtype).requires_grad_(True)
softmax_scale = 1.0 / math.sqrt(headdim)

# =============================================================================
# Create arbitrary func tensor
# Shape: [batch, nheads_q, n_func, seqlen_q + 256]
# batch=1 and nheads_q=1 means broadcast across all batches and heads
# =============================================================================
# Option 1: Use helper function (supports "causal", "full", "random" patterns)
arbitrary_func = arbitrary_func_tensor(
    batch=1,           # broadcast across batches
    nheads=1,          # broadcast across heads
    n_func=n_func,
    seqlen_q=seqlen_q,
    seqlen_k=seqlen_k,
    device="cuda",
    pattern="causal"   # or "full", "random"
)

# Option 2: Manual creation (causal example)
# arbitrary_func = torch.zeros(1, 1, n_func, seqlen_q + 256, dtype=torch.int32, device="cuda")
# for i in range(seqlen_q):
#     arbitrary_func[:, :, 0, i] = i + 1  # Causal: attend to [0, i+1)

# =============================================================================
# Get tile sizes for DSL backend (different from C++ backend!)
# DSL uses fixed tile sizes: Forward varies by arch, Backward varies by arch
# =============================================================================
arch = get_arch()
fwd_q_block, fwd_kv_block = get_fwd_tile_sizes_dsl(arch=arch)  # (128, 128) SM90 / (256, 128) SM100
bwd_q_block, bwd_kv_block = get_bwd_tile_sizes_dsl(arch=arch, is_arbitrary=True)  # (64, 128) SM90 / (128, 128) SM100
print(f"Arch: {arch}, Forward: ({fwd_q_block}, {fwd_kv_block}), Backward: ({bwd_q_block}, {bwd_kv_block})")

# =============================================================================
# Generate block sparsity using CUDA kernel
# =============================================================================
# Forward: Q2K direction
(k_mask_cnt, k_mask_offset, k_mask_idx,
 k_full_cnt, k_full_offset, k_full_idx) = \
    create_block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func, seqlen_q, seqlen_k,
        Q_BLOCK_SIZE=fwd_q_block,
        KV_BLOCK_SIZE=fwd_kv_block,
        check_q_boundary=True
    )

linear_k_block_sparse = LinearBlockSparseTensorsTorch(
    mask_block_cnt=k_mask_cnt,
    mask_block_offset=k_mask_offset,
    mask_block_idx=k_mask_idx,
    full_block_cnt=k_full_cnt,
    full_block_offset=k_full_offset,
    full_block_idx=k_full_idx,
)

# Backward: K2Q direction
(q_mask_cnt, q_mask_offset, q_mask_idx,
 q_full_cnt, q_full_offset, q_full_idx) = \
    create_block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func, seqlen_q, seqlen_k,
        Q_BLOCK_SIZE=bwd_q_block,
        KV_BLOCK_SIZE=bwd_kv_block
    )

linear_q_block_sparse = LinearBlockSparseTensorsTorch(
    mask_block_cnt=q_mask_cnt,
    mask_block_offset=q_mask_offset,
    mask_block_idx=q_mask_idx,
    full_block_cnt=q_full_cnt,
    full_block_offset=q_full_offset,
    full_block_idx=q_full_idx,
)

# =============================================================================
# Run attention with arbitrary mask
# =============================================================================
out, lse = flash_attn_func(
    q=q,
    k=k,
    v=v,
    softmax_scale=softmax_scale,
    causal=False,           # Disable built-in causal (using arbitrary instead)
    arbitrary=True,         # Enable arbitrary mask mode
    window_size=(None, None),
    softcap=0.0,
    num_splits=1,
    pack_gqa=False,
    deterministic=False,
    mask_mod=None,
    linear_k_block_sparse_tensors=linear_k_block_sparse,
    linear_q_block_sparse_tensors=linear_q_block_sparse,
    aux_tensors=[arbitrary_func],  # Pass func tensor for element-wise masking
)

# =============================================================================
# Backward pass works automatically
# =============================================================================
dout = torch.randn_like(out)
out.backward(dout)
print(f"dQ shape: {q.grad.shape}, dK shape: {k.grad.shape}, dV shape: {v.grad.shape}")
```

### Cutlass C++ Backend (`hopper/`)

> **Location**: `flash-attention/hopper/` directory
> **Supported architectures**: SM8x (Ampere/Ada) and SM90 (Hopper)

> **Prerequisite**: Before running these examples, compile and install the Cutlass C++ backend:
> ```bash
> cd flash-attention/hopper
> make install ARBITRARY=1 NUM_FUNC=3 HDIM128=1 BACKWARD=1
> # Add SM8X=1 for Ampere/Ada support
> ```

#### Usage (Arbitrary Mask + Block Sparsity)

For custom masking patterns with block sparsity:

```python
import sys
sys.path.insert(0, "hopper")

import torch
import math
from flash_attn_interface import flash_attn_func, LinearBlockSparseTensors
from flash_attn.utils.tile_size import get_fwd_tile_sizes, get_bwd_tile_sizes, get_arch
from flash_attn.cute.mask_definitions import arbitrary_func_tensor
import create_block_mask_cuda

# Configuration
batch_size, seqlen_q, seqlen_k = 1, 8192, 8192
nheads, nheads_kv, headdim = 32, 32, 128
dtype = torch.bfloat16
n_func = 3  # Must be odd

# Create tensors
q = torch.randn(batch_size, seqlen_q, nheads, headdim, device="cuda", dtype=dtype, requires_grad=True)
k = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device="cuda", dtype=dtype, requires_grad=True)
v = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device="cuda", dtype=dtype, requires_grad=True)
softmax_scale = 1.0 / math.sqrt(headdim)

# Create arbitrary func tensor (causal pattern)
arbitrary_func = arbitrary_func_tensor(1, 1, n_func, seqlen_q, seqlen_k, device="cuda", pattern="causal")

# Get tile sizes for block sparsity
arch = get_arch()
fwd_q_block, fwd_kv_block = get_fwd_tile_sizes(arch=arch, headdim=headdim, is_arbitrary=True)
bwd_q_block, bwd_kv_block = get_bwd_tile_sizes(arch=arch, headdim=headdim, is_arbitrary=True)

# Generate Q2K block sparsity (for forward pass)
(k_mask_cnt, k_mask_offset, k_mask_idx, k_full_cnt, k_full_offset, k_full_idx) = \
    create_block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func, seqlen_q, seqlen_k,
        Q_BLOCK_SIZE=fwd_q_block, KV_BLOCK_SIZE=fwd_kv_block, check_q_boundary=True
    )
q2k_sparse = LinearBlockSparseTensors(k_mask_cnt, k_mask_offset, k_mask_idx, k_full_cnt, k_full_offset, k_full_idx)

# Generate K2Q block sparsity (for backward pass)
(q_mask_cnt, q_mask_offset, q_mask_idx, q_full_cnt, q_full_offset, q_full_idx) = \
    create_block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func, seqlen_q, seqlen_k,
        Q_BLOCK_SIZE=bwd_q_block, KV_BLOCK_SIZE=bwd_kv_block
    )
k2q_sparse = LinearBlockSparseTensors(q_mask_cnt, q_mask_offset, q_mask_idx, q_full_cnt, q_full_offset, q_full_idx)

# Run attention with block sparsity (forward + backward)
out = flash_attn_func(
    q, k, v,
    softmax_scale=softmax_scale,
    causal=False,  # Using arbitrary_func instead
    arbitrary_func=arbitrary_func,
    q2k_block_sparse=q2k_sparse,
    k2q_block_sparse=k2q_sparse,
)
out.sum().backward()
```

### Supported Attention Modes

| Mode | Description | `nheads_kv` | DSL SM90 | DSL SM100 | C++ SM8x/90 |
|------|-------------|-------------|----------|-----------|-----|
| MHA | Multi-Head Attention | `nheads_kv = nheads` | ✓ | ✓ | ✓ |
| GQA | Grouped Query Attention | `nheads_kv = nheads // group_size` | ✗ | ✓ | ✓ |
| MQA | Multi-Query Attention | `nheads_kv = 1` | ✗ | ✓ | ✓ |

**Note**: DSL backend on SM90 only supports MHA mode. Use Cutlass C++ backend (`hopper/`) for GQA/MQA on SM8x/SM90.

## API Reference

### `flash_attn_func` (CUTE DSL)

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
- `aux_tensors=[arbitrary_func]`: Passes the Func Tensor for element-wise masking
- `linear_k_block_sparse_tensors`: Block sparsity for forward pass (Q2K)
- `linear_q_block_sparse_tensors`: Block sparsity for backward pass (K2Q)

**Return values**:
- `output`: Attention output tensor `[B, seqlen_q, H, D]`
- `lse`: Log-sum-exp values `[B, H, seqlen_q]` (required for backward pass). Note: `lse` is only computed when any input tensor has `requires_grad=True`; otherwise it may be `None`.

### Tile Size Utility Functions

```python
from flash_attn.utils.tile_size import (
    get_fwd_tile_sizes_dsl, get_bwd_tile_sizes_dsl,  # DSL backend
    get_fwd_tile_sizes, get_bwd_tile_sizes,          # C++ backend
    get_tile_sizes_by_backend,                        # Unified API
    get_arch,
)

# Get GPU architecture
arch = get_arch()  # Returns: 80, 86, 89, 90, or 100

# ===== DSL Backend (CUTE DSL) =====
# Fixed tile sizes, independent of headdim
fwd_q, fwd_kv = get_fwd_tile_sizes_dsl(arch=arch)  # (128, 128) SM90 / (256, 128) SM100
bwd_q, bwd_kv = get_bwd_tile_sizes_dsl(arch=arch, is_arbitrary=True)  # (64, 128) SM90 / (128, 128) SM100

# ===== Cutlass C++ Backend (hopper/, SM8x/SM90) =====
# Tile sizes vary by architecture and headdim
fwd_q_block, fwd_kv_block = get_fwd_tile_sizes(
    arch=arch, headdim=128, is_arbitrary=True
)
bwd_q_block, bwd_kv_block = get_bwd_tile_sizes(
    arch=arch, headdim=128, is_arbitrary=True
)

# ===== Unified API =====
# Auto-select based on backend parameter
fwd_q, fwd_kv = get_tile_sizes_by_backend(
    backend="cute",  # "cute" for DSL, "hopper" for Cutlass C++
    pass_type="forward",
    arch=arch, headdim=128, is_arbitrary=True
)
```

## Additional Resources

- **Test file**: `tests/cute/test_arbitrary_mask.py` - Comprehensive tests and benchmarks
- **CUDA kernel source**: `csrc/utils/create_block_mask/` - Block sparsity generation
- **Mask definitions**: `flash_attn/cute/mask_definitions.py` - Helper functions for creating Func Tensors
- **Tile size utility**: `flash_attn/utils/tile_size.py` - Automatic tile size detection
- **Cutlass C++ tile sizes**: `hopper/tile_size.h` - Single source of truth for tile sizes
