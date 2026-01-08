#include "create_block_mask.h"
#include <cassert>
#include <cub/cub.cuh>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// Block state enumeration
// EMPTY=0, MASK=1, FULL=2
// STATE_NONE=-1 is a neutral state for inactive threads (doesn't affect reduction)
constexpr int STATE_NONE = -1;  // Neutral state for inactive threads
constexpr int STATE_EMPTY = 0;
constexpr int STATE_MASK = 1;
constexpr int STATE_FULL = 2;

// Warp size
constexpr int WARP_SIZE_CONST = 32;

// Maximum Q_BLOCK_SIZE supported (for shared memory allocation)
#ifndef MAX_Q_BLOCK_SIZE
#define MAX_Q_BLOCK_SIZE 256
#endif

// Maximum number of warps per block
#define MAX_NUM_WARPS (MAX_Q_BLOCK_SIZE / WARP_SIZE_CONST)

/**
 * Device function to determine the state of a (q_token, kv_block) pair.
 * 
 * For a given q_token and kv_block [kv_start, kv_end), check against
 * the valid intervals defined by func_vals:
 *   interval 0: [0, F0)
 *   interval 1: [F1, F2)
 *   interval 2: [F3, F4)
 *   ...
 * 
 * @param func_ptr: pointer to func_tensor at position [b, h, 0, q_idx]
 * @param n_func: number of function values
 * @param stride_f: stride for the n_func dimension
 * @param kv_start: start of kv_block (inclusive)
 * @param kv_end: end of kv_block (exclusive)
 * 
 * Returns:
 *   STATE_FULL: kv_block is completely within some valid interval
 *   STATE_MASK: kv_block partially overlaps with valid intervals
 *   STATE_EMPTY: kv_block has no overlap with any valid interval
 */
__device__ __forceinline__ int get_token_kv_block_state(
    const int* func_ptr,
    int n_func,
    int stride_f,
    int kv_start, 
    int kv_end
) {
    int num_intervals = (n_func + 1) / 2;
    
    bool has_any_overlap = false;
    bool is_fully_covered = false;
    
    // Check first interval: [0, F0)
    {
        int interval_start = 0;
        int interval_end = func_ptr[0 * stride_f];  // F0
        
        if (interval_end > interval_start) {
            // Check if kv_block is fully covered by this interval
            if (interval_start <= kv_start && interval_end >= kv_end) {
                is_fully_covered = true;
            }
            // Check if there's any overlap
            else if (interval_start < kv_end && interval_end > kv_start) {
                has_any_overlap = true;
            }
        }
    }
    
    // Check remaining intervals: [F_{2i-1}, F_{2i}) for i = 1, 2, ...
    for (int i = 1; i < num_intervals && !is_fully_covered; i++) {
        int interval_start = func_ptr[(2 * i - 1) * stride_f];  // F_{2i-1}
        int interval_end = func_ptr[(2 * i) * stride_f];        // F_{2i}
        
        if (interval_end > interval_start) {
            // Check if kv_block is fully covered by this interval
            if (interval_start <= kv_start && interval_end >= kv_end) {
                is_fully_covered = true;
            }
            // Check if there's any overlap
            else if (interval_start < kv_end && interval_end > kv_start) {
                has_any_overlap = true;
            }
        }
    }
    
    if (is_fully_covered) {
        return STATE_FULL;
    } else if (has_any_overlap) {
        return STATE_MASK;
    } else {
        return STATE_EMPTY;
    }
}

/**
 * Combine two states into one.
 * Rules:
 * - STATE_NONE is neutral (doesn't affect the other state)
 * - Same state → keep same
 * - One is MASK → MASK
 * - FULL + EMPTY → MASK
 */
__device__ __forceinline__ int combine_states(int state1, int state2) {
    // STATE_NONE is neutral - doesn't affect the result
    if (state1 == STATE_NONE) return state2;
    if (state2 == STATE_NONE) return state1;
    
    if (state1 == state2) {
        return state1;
    }
    // Different states → MASK
    return STATE_MASK;
}

/**
 * Block-level reduction to determine the final state of a block.
 * 
 * Rules:
 * - All threads FULL → FULL block
 * - All threads EMPTY → EMPTY block (skip)
 * - Any thread MASK, or (some FULL and some EMPTY) → MASK block
 * 
 * Optimized with warp shuffle + shared memory two-phase reduction.
 */
__device__ __forceinline__ int reduce_block_state(
    int thread_state,
    int num_active_threads,
    int* warp_results  // shared memory array of size (blockDim.x / WARP_SIZE)
) {
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    // For inactive threads, use STATE_NONE (neutral state that doesn't affect reduction)
    int state = (tid < num_active_threads) ? thread_state : STATE_NONE;
    
    // Phase 1: Warp-level reduction using shuffle
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, tid < num_active_threads);
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int other_state = __shfl_xor_sync(0xFFFFFFFF, state, offset);
        // Only combine if the other thread is active
        if ((lane_id ^ offset) < WARP_SIZE) {
            bool other_active = (tid ^ offset) < num_active_threads;
            if (other_active) {
                state = combine_states(state, other_state);
            }
        }
    }
    
    // Phase 2: Lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        // Check if this warp has any active threads
        bool warp_has_active = (warp_id * WARP_SIZE) < num_active_threads;
        warp_results[warp_id] = warp_has_active ? state : STATE_NONE;
    }
    __syncthreads();
    
    // Phase 3: First warp does final reduction across all warps
    if (warp_id == 0) {
        // Load warp result (or STATE_NONE if this lane doesn't correspond to a valid warp)
        state = (lane_id < num_warps) ? warp_results[lane_id] : STATE_NONE;
        
        // Check which warps actually have active threads
        int threads_in_warp = min(WARP_SIZE, max(0, num_active_threads - lane_id * WARP_SIZE));
        bool warp_valid = (lane_id < num_warps) && (threads_in_warp > 0);
        
        // Warp-level reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            int other_state = __shfl_xor_sync(0xFFFFFFFF, state, offset);
            int other_lane = lane_id ^ offset;
            bool other_valid = (other_lane < num_warps) && 
                               (other_lane * WARP_SIZE < num_active_threads);
            if (other_valid) {
                state = combine_states(state, other_state);
            }
        }
        
        // Lane 0 has final result, write back to shared memory
        if (lane_id == 0) {
            warp_results[0] = state;
        }
    }
    __syncthreads();
    
    return warp_results[0];
}

// =============================================================================
// Q2K Kernel (Forward): fix q_block, loop kv_blocks
// =============================================================================

/**
 * Q2K Kernel: Create block sparse tensors from function encoding.
 * 
 * Design:
 * - One block processes one (b, h, q_block)
 * - Each block has Q_BLOCK_SIZE threads, one thread per q_token
 * - Grid: dim3(num_q_blocks, H, B)
 * 
 * Algorithm:
 * 1. Each thread handles one q_token in the q_block
 * 2. For each kv_block:
 *    a. Each thread determines its (q_token, kv_block) state: FULL/MASK/EMPTY
 *    b. Block-level reduction to get the kv_block's final state
 *    c. Thread 0 records the result
 * 
 * Output layout for block_idx:
 * - Full blocks written left-to-right: block_idx[0], block_idx[1], ...
 * - Mask blocks written right-to-left: block_idx[num_kv_blocks-1], block_idx[num_kv_blocks-2], ...
 * 
 * @param func_tensor: [B, H, n_func, func_q_len], with strides (stride_b, stride_h, stride_f, stride_q)
 */
__global__ void create_q2k_block_sparse_from_func_kernel(
    const int* __restrict__ func_tensor,  // [B, H, n_func, func_q_len]
    int stride_b, int stride_h, int stride_f, int stride_q,  // strides for func_tensor
    int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    int num_q_blocks, int num_kv_blocks,
    bool check_q_boundary,  // If true, partial q_blocks cannot have FULL kv_blocks
    int* __restrict__ mask_block_cnt,     // [B, H, num_q_blocks]
    int* __restrict__ full_block_cnt,     // [B, H, num_q_blocks]
    int* __restrict__ block_idx,          // [B, H, num_q_blocks, num_kv_blocks]
    int* __restrict__ total_mask_blocks,  // Global accumulator for total mask blocks
    int* __restrict__ total_full_blocks   // Global accumulator for total full blocks
) {
    // Shared memory for warp-level reduction results
    __shared__ int warp_results[MAX_NUM_WARPS];
    
    // Get block indices
    int q_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int tid = threadIdx.x;
    
    // Calculate q range for this block
    int q_start = q_block * Q_BLOCK_SIZE;
    int q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
    int num_active_threads = q_end - q_start;
    
    // Check if this q_block is full (has Q_BLOCK_SIZE active tokens)
    bool is_q_block_full = (num_active_threads == Q_BLOCK_SIZE);
    
    // This thread's q_token index
    int q_idx = q_start + tid;
    bool is_active = (tid < num_active_threads);
    
    // Pointer to func_tensor at [b, h, 0, q_idx] using strides
    const int* my_func_ptr = func_tensor + b * stride_b + h * stride_h + q_idx * stride_q;
    
    // Output tensor strides (contiguous layout)
    // cnt tensor has shape [B, H, num_q_blocks + 1] for CSR offset format
    int out_cnt_stride_b = H * (num_q_blocks + 1);
    int out_cnt_stride_h = (num_q_blocks + 1);
    int out_idx_stride_b = H * num_q_blocks * num_kv_blocks;
    int out_idx_stride_h = num_q_blocks * num_kv_blocks;
    int out_idx_stride_q = num_kv_blocks;
    
    // Output offsets
    // cnt is written at offset+1 (offset 0 is reserved for the leading 0 in CSR format)
    int cnt_offset = b * out_cnt_stride_b + h * out_cnt_stride_h + q_block + 1;
    int idx_base = b * out_idx_stride_b + h * out_idx_stride_h + q_block * out_idx_stride_q;
    
    // Counters
    int mask_count = 0;
    int full_count = 0;
    
    // Process each kv_block
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * KV_BLOCK_SIZE;
        int kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);
        
        // Check if this is a partial kv_block (last block with KV_LEN % KV_BLOCK_SIZE != 0)
        // Partial kv_blocks can NEVER be FULL because positions beyond KV_LEN are invalid
        bool is_kv_block_full = (kv_end == kv_start + KV_BLOCK_SIZE);
        
        // Each thread determines its state for this kv_block
        int thread_state = STATE_EMPTY;
        if (is_active) {
            thread_state = get_token_kv_block_state(my_func_ptr, n_func, stride_f, kv_start, kv_end);
        }
        
        // Block-level reduction
        int kv_block_state = reduce_block_state(thread_state, num_active_threads, warp_results);
        
        // Thread 0 records the result
        if (tid == 0) {
            // A block can only be FULL if:
            // 1. All q_tokens in q_block have FULL state for this kv_block
            // 2. q_block itself is full (if check_q_boundary is true)
            // 3. kv_block itself is full (not a partial block at the boundary)
            bool can_be_full = (!check_q_boundary || is_q_block_full) && is_kv_block_full;
            
            if (kv_block_state == STATE_FULL && can_be_full) {
                // Full blocks: write left-to-right
                block_idx[idx_base + full_count] = kv_block;
                full_count++;
            } else if (kv_block_state == STATE_FULL || kv_block_state == STATE_MASK) {
                // Mask blocks: write right-to-left
                block_idx[idx_base + (num_kv_blocks - 1 - mask_count)] = kv_block;
                mask_count++;
            }
        }
        
        __syncthreads();
    }
    
    // Write final counts and atomically accumulate totals
    if (tid == 0) {
        // Write leading 0 for CSR format (first CTA of each (b,h) pair)
        if (q_block == 0) {
            int zero_offset = b * out_cnt_stride_b + h * out_cnt_stride_h;
            mask_block_cnt[zero_offset] = 0;
            full_block_cnt[zero_offset] = 0;
        }
        
        mask_block_cnt[cnt_offset] = mask_count;
        full_block_cnt[cnt_offset] = full_count;
        if (total_mask_blocks != nullptr) {
            atomicAdd(total_mask_blocks, mask_count);
        }
        if (total_full_blocks != nullptr) {
            atomicAdd(total_full_blocks, full_count);
        }
    }
}

// =============================================================================
// K2Q Kernel (Backward): fix kv_block, loop q_blocks
// =============================================================================

/**
 * K2Q Kernel: Create block sparse tensors from function encoding for backward pass.
 * 
 * Design:
 * - One block processes one (b, h, kv_block)
 * - Each block has Q_BLOCK_SIZE threads (to match q_block reduction needs)
 * - Grid: dim3(num_kv_blocks, H, B)
 * 
 * Algorithm:
 * 1. For each q_block:
 *    a. Each thread handles one q_token in the q_block
 *    b. Each thread determines its (q_token, kv_block) state
 *    c. Block-level reduction to get the q_block's final state
 *    d. Thread 0 records the result
 * 
 * Output layout for block_idx:
 * - Full blocks written left-to-right: block_idx[0], block_idx[1], ...
 * - Mask blocks written right-to-left: block_idx[num_q_blocks-1], block_idx[num_q_blocks-2], ...
 * 
 * Note: Always performs boundary checking (partial q_blocks cannot have FULL status).
 * 
 * @param func_tensor: [B, H, n_func, func_q_len], with strides (stride_b, stride_h, stride_f, stride_q)
 */
__global__ void create_k2q_block_sparse_from_func_kernel(
    const int* __restrict__ func_tensor,  // [B, H, n_func, func_q_len]
    int stride_b, int stride_h, int stride_f, int stride_q,  // strides for func_tensor
    int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    int num_q_blocks, int num_kv_blocks,
    int* __restrict__ mask_block_cnt,     // [B, H, num_kv_blocks]
    int* __restrict__ full_block_cnt,     // [B, H, num_kv_blocks]
    int* __restrict__ block_idx,          // [B, H, num_kv_blocks, num_q_blocks]
    int* __restrict__ total_mask_blocks,  // Global accumulator for total mask blocks
    int* __restrict__ total_full_blocks   // Global accumulator for total full blocks
) {
    // Shared memory for warp-level reduction results
    __shared__ int warp_results[MAX_NUM_WARPS];
    
    // Get block indices
    int kv_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int tid = threadIdx.x;
    
    // Calculate kv range for this block
    int kv_start = kv_block * KV_BLOCK_SIZE;
    int kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);
    
    // Check if this is a partial kv_block (last block with KV_LEN % KV_BLOCK_SIZE != 0)
    // Partial kv_blocks can NEVER have FULL q_blocks because positions beyond KV_LEN are invalid
    bool is_kv_block_full = (kv_end == kv_start + KV_BLOCK_SIZE);
    
    // Base pointer for func_tensor at [b, h, 0, 0]
    const int* base_func_ptr = func_tensor + b * stride_b + h * stride_h;
    
    // Output tensor strides (contiguous layout)
    // For K2Q: output is [B, H, num_kv_blocks, num_q_blocks]
    // cnt tensor has shape [B, H, num_kv_blocks + 1] for CSR offset format
    int out_cnt_stride_b = H * (num_kv_blocks + 1);
    int out_cnt_stride_h = (num_kv_blocks + 1);
    int out_idx_stride_b = H * num_kv_blocks * num_q_blocks;
    int out_idx_stride_h = num_kv_blocks * num_q_blocks;
    int out_idx_stride_kv = num_q_blocks;
    
    // Output offsets
    // cnt is written at offset+1 (offset 0 is reserved for the leading 0 in CSR format)
    int cnt_offset = b * out_cnt_stride_b + h * out_cnt_stride_h + kv_block + 1;
    int idx_base = b * out_idx_stride_b + h * out_idx_stride_h + kv_block * out_idx_stride_kv;
    
    // Counters
    int mask_count = 0;
    int full_count = 0;
    
    // Process each q_block
    for (int q_block = 0; q_block < num_q_blocks; q_block++) {
        // Calculate q range for this q_block
        int q_start = q_block * Q_BLOCK_SIZE;
        int q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
        int num_active_threads = q_end - q_start;
        
        // Check if this q_block is full (has Q_BLOCK_SIZE active tokens)
        bool is_q_block_full = (num_active_threads == Q_BLOCK_SIZE);
        
        // This thread's q_token index
        int q_idx = q_start + tid;
        bool is_active = (tid < num_active_threads);
        
        // Pointer to func_tensor at [b, h, 0, q_idx]
        const int* my_func_ptr = base_func_ptr + q_idx * stride_q;
        
        // Each thread determines its state for this (q_token, kv_block)
        int thread_state = STATE_EMPTY;
        if (is_active) {
            thread_state = get_token_kv_block_state(my_func_ptr, n_func, stride_f, kv_start, kv_end);
        }
        
        // Block-level reduction
        int q_block_state = reduce_block_state(thread_state, num_active_threads, warp_results);
        
        // Thread 0 records the result
        // For backward, always check q_boundary: partial q_blocks cannot have FULL status
        // Also, partial kv_blocks can NEVER have FULL q_blocks
        if (tid == 0) {
            // A q_block can only be FULL if:
            // 1. All q_tokens have FULL state for this kv_block
            // 2. q_block itself is full (not partial)
            // 3. kv_block itself is full (not a partial block at the boundary)
            bool can_be_full = is_q_block_full && is_kv_block_full;
            
            if (q_block_state == STATE_FULL && can_be_full) {
                // Full blocks: write left-to-right
                block_idx[idx_base + full_count] = q_block;
                full_count++;
            } else if (q_block_state == STATE_FULL || q_block_state == STATE_MASK) {
                // Mask blocks: write right-to-left
                block_idx[idx_base + (num_q_blocks - 1 - mask_count)] = q_block;
                mask_count++;
            }
        }
        
        __syncthreads();
    }
    
    // Write final counts and atomically accumulate totals
    if (tid == 0) {
        // Write leading 0 for CSR format (first CTA of each (b,h) pair)
        if (kv_block == 0) {
            int zero_offset = b * out_cnt_stride_b + h * out_cnt_stride_h;
            mask_block_cnt[zero_offset] = 0;
            full_block_cnt[zero_offset] = 0;
        }
        
        mask_block_cnt[cnt_offset] = mask_count;
        full_block_cnt[cnt_offset] = full_count;
        if (total_mask_blocks != nullptr) {
            atomicAdd(total_mask_blocks, mask_count);
        }
        if (total_full_blocks != nullptr) {
            atomicAdd(total_full_blocks, full_count);
        }
    }
}

// =============================================================================
// Launch functions
// =============================================================================

void launch_create_q2k_block_sparse_from_func(
    const int* d_func_tensor,
    int stride_b, int stride_h, int stride_f, int stride_q,
    int B, int H, int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    bool check_q_boundary,
    int* d_mask_block_cnt,
    int* d_full_block_cnt,
    int* d_block_idx,
    int* d_total_mask_blocks,
    int* d_total_full_blocks,
    cudaStream_t stream
) {
    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);
    
    // 3D grid: (num_q_blocks, H, B)
    dim3 grid(num_q_blocks, H, B);
    int block_size = Q_BLOCK_SIZE;

    create_q2k_block_sparse_from_func_kernel<<<grid, block_size, 0, stream>>>(
        d_func_tensor, 
        stride_b, stride_h, stride_f, stride_q,
        Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        num_q_blocks, num_kv_blocks,
        check_q_boundary,
        d_mask_block_cnt, d_full_block_cnt,
        d_block_idx,
        d_total_mask_blocks, d_total_full_blocks
    );
}

void launch_create_k2q_block_sparse_from_func(
    const int* d_func_tensor,
    int stride_b, int stride_h, int stride_f, int stride_q,
    int B, int H, int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    int* d_mask_block_cnt,
    int* d_full_block_cnt,
    int* d_block_idx,
    int* d_total_mask_blocks,
    int* d_total_full_blocks,
    cudaStream_t stream
) {
    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);
    
    // 3D grid: (num_kv_blocks, H, B) - note: kv_blocks is the outer loop
    dim3 grid(num_kv_blocks, H, B);
    int block_size = Q_BLOCK_SIZE;  // Still use Q_BLOCK_SIZE for reduction

    create_k2q_block_sparse_from_func_kernel<<<grid, block_size, 0, stream>>>(
        d_func_tensor, 
        stride_b, stride_h, stride_f, stride_q,
        Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        num_q_blocks, num_kv_blocks,
        d_mask_block_cnt, d_full_block_cnt,
        d_block_idx,
        d_total_mask_blocks, d_total_full_blocks
    );
}

// =============================================================================
// Compact Block Idx Kernels
// =============================================================================

/**
 * Kernel to extract compact indices from block_idx.
 * 
 * Each block handles one row of block_idx.
 * - Full indices: stored left-to-right at block_idx[row, 0:full_cnt]
 * - Mask indices: stored right-to-left at block_idx[row, max_blocks-mask_cnt:max_blocks]
 *   Need to reverse when extracting.
 */
__global__ void extract_compact_indices_kernel(
    const int* __restrict__ block_idx,        // [n_blocks, max_blocks]
    const int* __restrict__ mask_cnt,         // [n_blocks]
    const int* __restrict__ full_cnt,         // [n_blocks]
    const int* __restrict__ mask_offset,      // [n_blocks + 1]
    const int* __restrict__ full_offset,      // [n_blocks + 1]
    int n_blocks,
    int max_blocks,
    int* __restrict__ mask_idx_compact,       // [total_mask_blocks]
    int* __restrict__ full_idx_compact        // [total_full_blocks]
) {
    int row = blockIdx.x;
    if (row >= n_blocks) return;
    
    int tid = threadIdx.x;
    
    int fcnt = full_cnt[row];
    int mcnt = mask_cnt[row];
    int foff = full_offset[row];
    int moff = mask_offset[row];
    
    const int* row_ptr = block_idx + row * max_blocks;
    
    // Extract full indices (left-to-right, no reversal needed)
    for (int i = tid; i < fcnt; i += blockDim.x) {
        full_idx_compact[foff + i] = row_ptr[i];
    }
    
    // Extract mask indices (right-to-left, need to reverse)
    // Original: row_ptr[max_blocks - mcnt + i] for i in [0, mcnt)
    // Reversed: row_ptr[max_blocks - 1 - i] for i in [0, mcnt)
    for (int i = tid; i < mcnt; i += blockDim.x) {
        mask_idx_compact[moff + i] = row_ptr[max_blocks - 1 - i];
    }
}

void launch_extract_compact_indices(
    const int* d_block_idx,
    const int* d_mask_block_cnt,
    const int* d_full_block_cnt,
    const int* d_mask_block_offset,
    const int* d_full_block_offset,
    int n_blocks,
    int max_blocks,
    int* d_mask_block_idx,
    int* d_full_block_idx,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    extract_compact_indices_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
        d_block_idx,
        d_mask_block_cnt, d_full_block_cnt,
        d_mask_block_offset, d_full_block_offset,
        n_blocks, max_blocks,
        d_mask_block_idx, d_full_block_idx
    );
}

// ============================================================================
// Dual Inclusive Sum - Single CTA scan using CUB BlockScan
// One CTA processes all B*H*num_blocks_plus_one elements
// ============================================================================

// Block size for scan kernel (must be power of 2 for CUB BlockScan)
#define SCAN_BLOCK_SIZE 256

// Custom int2 addition operator for CUB
struct Int2Sum {
    __device__ __forceinline__ int2 operator()(const int2& a, const int2& b) const {
        return make_int2(a.x + b.x, a.y + b.y);
    }
};

__global__ void dual_inclusive_sum_kernel(
    const int* __restrict__ mask_cnt,      // [total_elements]
    const int* __restrict__ full_cnt,      // [total_elements]
    int* __restrict__ mask_offset,         // [total_elements]
    int* __restrict__ full_offset,         // [total_elements]
    int total_elements                     // B * H * num_blocks_plus_one
) {
    // CUB BlockScan type
    typedef cub::BlockScan<int2, SCAN_BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int2 shared_prefix;
    
    int tid = threadIdx.x;
    
    // Process all elements in tiles using a single CTA
    int2 running_prefix = make_int2(0, 0);
    
    for (int tile_start = 0; tile_start < total_elements; tile_start += SCAN_BLOCK_SIZE) {
        int idx = tile_start + tid;
        
        // Load data (use 0 for out-of-bounds)
        int2 thread_data;
        if (idx < total_elements) {
            thread_data = make_int2(mask_cnt[idx], full_cnt[idx]);
        } else {
            thread_data = make_int2(0, 0);
        }
        
        // Perform block-level inclusive scan
        int2 thread_result;
        BlockScan(temp_storage).InclusiveScan(thread_data, thread_result, Int2Sum());
        __syncthreads();
        
        // Add running prefix from previous tiles
        thread_result.x += running_prefix.x;
        thread_result.y += running_prefix.y;
        
        // Write result
        if (idx < total_elements) {
            mask_offset[idx] = thread_result.x;
            full_offset[idx] = thread_result.y;
        }
        
        // Broadcast running_prefix to all threads for next tile
        // The last valid thread in this tile writes to shared memory
        int last_tid_in_tile = min(SCAN_BLOCK_SIZE - 1, total_elements - 1 - tile_start);
        if (tid == last_tid_in_tile) {
            shared_prefix = thread_result;
        }
        __syncthreads();
        running_prefix = shared_prefix;
    }
}

void launch_dual_inclusive_sum(
    const int* d_mask_cnt,
    const int* d_full_cnt,
    int* d_mask_offset,
    int* d_full_offset,
    int total_elements,         // B * H * num_blocks_plus_one (total elements to scan)
    cudaStream_t stream
) {
    // Launch single CTA to process all elements
    dual_inclusive_sum_kernel<<<1, SCAN_BLOCK_SIZE, 0, stream>>>(
        d_mask_cnt, d_full_cnt,
        d_mask_offset, d_full_offset,
        total_elements
    );
}
