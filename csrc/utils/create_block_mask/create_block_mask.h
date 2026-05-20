#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/**
 * Convert function encoding tensor to block sparse tensors (Q2K: forward pass).
 * 
 * Forward pass: fix q_block, loop kv_blocks.
 * 
 * Function encoding format (same as magi_to_hstu output):
 *   interval 0: [0, F0)       - valid range (no mask needed)
 *   interval 1: [F1, F2)      - valid range (no mask needed)
 *   interval 2: [F3, F4)      - valid range (no mask needed)
 *   ...
 * 
 * These intervals represent positions where attention is VALID (not masked).
 * 
 * Output tensors (BlockSparseTensorsTorch format):
 * - mask_block_cnt: [B, H, num_q_blocks], number of partial blocks per q_block
 * - full_block_cnt: [B, H, num_q_blocks], number of full blocks per q_block
 * - block_idx: [B, H, num_q_blocks, num_kv_blocks], combined indices of kv_blocks
 *              full blocks are stored left-to-right (indices 0, 1, 2, ...)
 *              mask blocks are stored right-to-left (indices num_kv_blocks-1, num_kv_blocks-2, ...)
 * 
 * Where:
 * - mask_block: blocks that are partially valid (need to apply mask)
 * - full_block: blocks that are fully valid (no mask needed, all attention computed)
 * 
 * @param d_func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 *                       func_q_len must be >= Q_LEN + 256 (to avoid bounds checking)
 *                       n_func must be odd (e.g., 1, 3, 5, 7, ...)
 * @param stride_b: stride for batch dimension
 * @param stride_h: stride for head dimension
 * @param stride_f: stride for n_func dimension
 * @param stride_q: stride for query dimension
 * @param B: batch size
 * @param H: number of heads
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param n_func: number of function values per query (must be odd)
 * @param Q_BLOCK_SIZE: block size for query dimension
 * @param KV_BLOCK_SIZE: block size for key/value dimension
 * @param check_q_boundary: if true (FlexAttention mode), partial q_blocks cannot have FULL kv_blocks;
 *                          if false, partial q_blocks can have FULL kv_blocks based on active q_tokens
 * @param d_mask_block_cnt: [B, H, num_q_blocks], output
 * @param d_full_block_cnt: [B, H, num_q_blocks], output
 * @param d_block_idx: [B, H, num_q_blocks, num_kv_blocks], output (combined full + mask indices)
 * @param stream: CUDA stream
 */
void launch_create_q2k_block_sparse_from_func(
    const int* d_func_tensor,
    int64_t stride_b, int64_t stride_h, int64_t stride_f, int64_t stride_q,
    int B,
    int H,
    int Q_LEN,
    int KV_LEN,
    int n_func,
    int Q_BLOCK_SIZE,
    int KV_BLOCK_SIZE,
    bool check_q_boundary,
    int* d_mask_block_cnt,
    int* d_full_block_cnt,
    int* d_block_idx,
    cudaStream_t stream = nullptr
);

/**
 * Precompute kv range for each q_block (K2Q optimization).
 * 
 * This kernel precomputes the min/max kv positions that each q_block can access.
 * The results are used by K2Q kernel to skip q_blocks that don't overlap with
 * the current kv_block, significantly reducing computation for sparse patterns.
 * 
 * Output tensors:
 * - d_q_block_kv_min: [B, H, num_q_blocks], minimum kv position for each q_block
 * - d_q_block_kv_max: [B, H, num_q_blocks], maximum kv position for each q_block (exclusive)
 * 
 * If a q_block has no valid intervals, min=INT_MAX and max=0.
 * 
 * @param d_func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param stride_b: stride for batch dimension
 * @param stride_h: stride for head dimension
 * @param stride_f: stride for n_func dimension
 * @param stride_q: stride for query dimension
 * @param B: batch size
 * @param H: number of heads
 * @param Q_LEN: query sequence length
 * @param n_func: number of function values per query (must be odd)
 * @param Q_BLOCK_SIZE: block size for query dimension
 * @param d_q_block_kv_min: [B, H, num_q_blocks], output
 * @param d_q_block_kv_max: [B, H, num_q_blocks], output
 * @param stream: CUDA stream
 */
void launch_compute_q_block_kv_range(
    const int* d_func_tensor,
    int64_t stride_b, int64_t stride_h, int64_t stride_f, int64_t stride_q,
    int B,
    int H,
    int Q_LEN,
    int n_func,
    int Q_BLOCK_SIZE,
    int* d_q_block_kv_min,
    int* d_q_block_kv_max,
    cudaStream_t stream = nullptr
);

/**
 * Convert function encoding tensor to block sparse tensors (K2Q: backward pass).
 * 
 * Backward pass: fix kv_block, loop q_blocks.
 * Always performs boundary checking (partial q_blocks cannot have FULL status).
 * 
 * Optimization: If precomputed q_block kv ranges are provided, the kernel will
 * skip q_blocks whose kv range doesn't overlap with the current kv_block.
 * 
 * Output tensors:
 * - mask_block_cnt: [B, H, num_kv_blocks], number of partial q_blocks per kv_block
 * - full_block_cnt: [B, H, num_kv_blocks], number of full q_blocks per kv_block
 * - block_idx: [B, H, num_kv_blocks, num_q_blocks], combined indices of q_blocks
 *              full blocks are stored left-to-right (indices 0, 1, 2, ...)
 *              mask blocks are stored right-to-left (indices num_q_blocks-1, num_q_blocks-2, ...)
 * 
 * @param d_func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param stride_b: stride for batch dimension
 * @param stride_h: stride for head dimension
 * @param stride_f: stride for n_func dimension
 * @param stride_q: stride for query dimension
 * @param B: batch size
 * @param H: number of heads
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param n_func: number of function values per query (must be odd)
 * @param Q_BLOCK_SIZE: block size for query dimension
 * @param KV_BLOCK_SIZE: block size for key/value dimension
 * @param d_q_block_kv_min: [B, H, num_q_blocks], precomputed min kv (can be nullptr to disable optimization)
 * @param d_q_block_kv_max: [B, H, num_q_blocks], precomputed max kv (can be nullptr to disable optimization)
 * @param d_mask_block_cnt: [B, H, num_kv_blocks], output
 * @param d_full_block_cnt: [B, H, num_kv_blocks], output
 * @param d_block_idx: [B, H, num_kv_blocks, num_q_blocks], output (combined full + mask indices)
 * @param stream: CUDA stream
 */
void launch_create_k2q_block_sparse_from_func(
    const int* d_func_tensor,
    int64_t stride_b, int64_t stride_h, int64_t stride_f, int64_t stride_q,
    int B,
    int H,
    int Q_LEN,
    int KV_LEN,
    int n_func,
    int Q_BLOCK_SIZE,
    int KV_BLOCK_SIZE,
    const int* d_q_block_kv_min,  // precomputed, can be nullptr
    const int* d_q_block_kv_max,  // precomputed, can be nullptr
    int* d_mask_block_cnt,
    int* d_full_block_cnt,
    int* d_block_idx,
    cudaStream_t stream = nullptr
);

/**
 * Compact block indices from BHQK format to linear sparse format (CSR-like).
 * 
 * Step 1: Compute prefix sum of mask_block_cnt and full_block_cnt to get offsets
 * Step 2: Extract indices from block_idx to compact mask_block_idx and full_block_idx
 * 
 * Input:
 * - d_mask_block_cnt: [n_blocks], count of mask blocks per row (flattened B*H*num_blocks)
 * - d_full_block_cnt: [n_blocks], count of full blocks per row
 * - d_block_idx: [n_blocks, max_blocks], combined indices
 *     - full blocks stored left-to-right: [0:full_cnt]
 *     - mask blocks stored right-to-left: [max_blocks-mask_cnt:max_blocks]
 * 
 * Output:
 * - d_mask_block_offset: [n_blocks + 1], prefix sum of mask_block_cnt
 * - d_mask_block_idx: [total_mask_blocks], compact mask indices
 * - d_full_block_offset: [n_blocks + 1], prefix sum of full_block_cnt
 * - d_full_block_idx: [total_full_blocks], compact full indices
 * 
 * @param d_mask_block_cnt: [n_blocks], int32 input
 * @param d_full_block_cnt: [n_blocks], int32 input
 * @param d_block_idx: [n_blocks, max_blocks], int32 input
 * @param n_blocks: total number of blocks (B * H * num_q_blocks or num_kv_blocks)
 * @param max_blocks: maximum blocks per row (num_kv_blocks or num_q_blocks)
 * @param d_mask_block_offset: [n_blocks + 1], int32 output
 * @param d_mask_block_idx: [total_mask_blocks], int32 output
 * @param d_full_block_offset: [n_blocks + 1], int32 output
 * @param d_full_block_idx: [total_full_blocks], int32 output
 * @param stream: CUDA stream
 */
/**
 * Extract compact indices from block_idx.
 * Note: Prefix sum is computed using PyTorch cumsum for efficiency.
 */
void launch_extract_compact_indices(
    const int* d_block_idx,
    const int* d_mask_block_cnt,
    const int* d_full_block_cnt,
    const int* d_mask_block_offset,
    const int* d_full_block_offset,
    int64_t n_blocks,
    int64_t max_blocks,
    int* d_mask_block_idx,
    int* d_full_block_idx,
    cudaStream_t stream = nullptr
);

/**
 * Dual inclusive sum - single CTA implementation with exclusive prefix sum output.
 * One CTA processes all B*H*num_blocks elements.
 * No temporary buffer needed.
 * 
 * Input:  cnt[0..n-1]
 * Output: offset[0..n] where offset[0]=0, offset[i+1]=sum(cnt[0..i])
 * 
 * This produces exclusive prefix sum format: [0, c0, c0+c1, ..., total]
 * 
 * @param d_mask_cnt: [n_elements], int32 input - mask block counts (flattened)
 * @param d_full_cnt: [n_elements], int32 input - full block counts (flattened)
 * @param d_mask_offset: [n_elements + 1], int32 output - exclusive prefix sum of mask_cnt
 * @param d_full_offset: [n_elements + 1], int32 output - exclusive prefix sum of full_cnt
 * @param n_elements: Number of count elements (B * H * num_blocks)
 * @param stream: CUDA stream
 */
void launch_dual_inclusive_sum(
    const int* d_mask_cnt,
    const int* d_full_cnt,
    int* d_mask_offset,
    int* d_full_offset,
    int64_t n_elements,
    cudaStream_t stream = nullptr
);

