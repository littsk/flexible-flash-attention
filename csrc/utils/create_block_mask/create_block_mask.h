#pragma once

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
 * - mask_block_idx: [B, H, num_q_blocks, num_kv_blocks], indices of partial kv_blocks
 * - full_block_cnt: [B, H, num_q_blocks], number of full blocks per q_block
 * - full_block_idx: [B, H, num_q_blocks, num_kv_blocks], indices of full kv_blocks
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
 * @param d_mask_block_idx: [B, H, num_q_blocks, num_kv_blocks], output
 * @param d_full_block_cnt: [B, H, num_q_blocks], output
 * @param d_full_block_idx: [B, H, num_q_blocks, num_kv_blocks], output
 * @param stream: CUDA stream
 */
void launch_create_q2k_block_sparse_from_func(
    const int* d_func_tensor,
    int stride_b, int stride_h, int stride_f, int stride_q,
    int B,
    int H,
    int Q_LEN,
    int KV_LEN,
    int n_func,
    int Q_BLOCK_SIZE,
    int KV_BLOCK_SIZE,
    bool check_q_boundary,
    int* d_mask_block_cnt,
    int* d_mask_block_idx,
    int* d_full_block_cnt,
    int* d_full_block_idx,
    cudaStream_t stream = nullptr
);

/**
 * Convert function encoding tensor to block sparse tensors (K2Q: backward pass).
 * 
 * Backward pass: fix kv_block, loop q_blocks.
 * Always performs boundary checking (partial q_blocks cannot have FULL status).
 * 
 * Output tensors:
 * - mask_block_cnt: [B, H, num_kv_blocks], number of partial q_blocks per kv_block
 * - mask_block_idx: [B, H, num_kv_blocks, num_q_blocks], indices of partial q_blocks
 * - full_block_cnt: [B, H, num_kv_blocks], number of full q_blocks per kv_block
 * - full_block_idx: [B, H, num_kv_blocks, num_q_blocks], indices of full q_blocks
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
 * @param d_mask_block_cnt: [B, H, num_kv_blocks], output
 * @param d_mask_block_idx: [B, H, num_kv_blocks, num_q_blocks], output
 * @param d_full_block_cnt: [B, H, num_kv_blocks], output
 * @param d_full_block_idx: [B, H, num_kv_blocks, num_q_blocks], output
 * @param stream: CUDA stream
 */
void launch_create_k2q_block_sparse_from_func(
    const int* d_func_tensor,
    int stride_b, int stride_h, int stride_f, int stride_q,
    int B,
    int H,
    int Q_LEN,
    int KV_LEN,
    int n_func,
    int Q_BLOCK_SIZE,
    int KV_BLOCK_SIZE,
    int* d_mask_block_cnt,
    int* d_mask_block_idx,
    int* d_full_block_cnt,
    int* d_full_block_idx,
    cudaStream_t stream = nullptr
);
