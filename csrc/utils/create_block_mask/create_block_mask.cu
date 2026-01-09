#include "create_block_mask.h"
#include <cassert>
#include <cub/cub.cuh>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// Block state enumeration
// EMPTY=0, MASK=1, FULL=2
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
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 * @param func_ptr: pointer to func_tensor at position [b, h, 0, q_idx]
 * @param n_func: number of function values (only used when N_FUNC == 0)
 * @param stride_f: stride for the n_func dimension
 * @param kv_start: start of kv_block (inclusive)
 * @param kv_end: end of kv_block (exclusive)
 * 
 * Returns:
 *   STATE_FULL: kv_block is completely within some valid interval
 *   STATE_MASK: kv_block partially overlaps with valid intervals
 *   STATE_EMPTY: kv_block has no overlap with any valid interval
 */
template <int N_FUNC>
__device__ __forceinline__ int get_token_kv_block_state(
    const int* func_ptr,
    int n_func,
    int stride_f,
    int kv_start, 
    int kv_end
) {
    // Use compile-time constant if N_FUNC > 0, otherwise use runtime value
    const int num_intervals = (N_FUNC > 0) ? ((N_FUNC + 1) / 2) : ((n_func + 1) / 2);
    
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
    // When N_FUNC is a compile-time constant, the loop can be fully unrolled
    #pragma unroll
    for (int i = 1; i < num_intervals; i++) {
        int interval_start = func_ptr[(2 * i - 1) * stride_f];  // F_{2i-1}
        int interval_end = func_ptr[(2 * i) * stride_f];        // F_{2i}

        if (interval_end <= interval_start) {
            continue;
        }

        // Check if kv_block is fully covered by this interval
        if (interval_start <= kv_start && interval_end >= kv_end) {
            is_fully_covered = true;
        }
        // Check if there's any overlap
        else if (interval_start < kv_end && interval_end > kv_start) {
            has_any_overlap = true;
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
 * - Same state → keep same
 * - One is MASK → MASK
 * - FULL + EMPTY → MASK
 */
__device__ __forceinline__ int combine_states(int state1, int state2) {
    if (state1 == state2) {
        return state1;
    }
    // Different states → MASK
    return STATE_MASK;
}

/**
 * Convert combined state bitmask to state value.
 * 
 * Bitmask encoding: EMPTY(0)->0b001, MASK(1)->0b010, FULL(2)->0b100
 * - 0b001 (1) → all EMPTY → STATE_EMPTY
 * - 0b100 (4) → all FULL → STATE_FULL
 * - otherwise → mixed states → STATE_MASK
 */
__device__ __forceinline__ int bitmask_to_state(unsigned int mask) {
    if (mask == 1u) return STATE_EMPTY;      // Only EMPTY present
    if (mask == 4u) return STATE_FULL;       // Only FULL present
    return STATE_MASK;                       // Mixed or has MASK
}

/**
 * Block-level reduction to determine the final state of a block.
 * 
 * Rules:
 * - All threads FULL → FULL block
 * - All threads EMPTY → EMPTY block (skip)
 * - Any thread MASK, or (some FULL and some EMPTY) → MASK block
 * 
 * Optimized with redux.sync (PTX instruction) for both warp-level and
 * cross-warp reduction, minimizing synchronization overhead.
 * 
 * Note: Inactive threads should have their state pre-set before calling:
 * - check_q_boundary=false: inactive threads contribute STATE_FULL
 * - check_q_boundary=true: inactive threads contribute STATE_EMPTY
 */
__device__ __forceinline__ int reduce_block_state(
    int thread_state,
    int* warp_results  // shared memory array of size MAX_NUM_WARPS
) {
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Convert state to bitmask: EMPTY(0)->0b001, MASK(1)->0b010, FULL(2)->0b100
    unsigned int state_mask = 1u << thread_state;
    
    // Phase 1: Warp-level reduction using redux.sync (single PTX instruction)
    unsigned int warp_mask = __reduce_or_sync(0xFFFFFFFF, state_mask);
    
    // Phase 2: Lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = static_cast<int>(warp_mask);
    }
    __syncthreads();
    
    // Phase 3: Warp 0 performs final reduction using redux.sync
    // Each thread in warp 0 reads one warp result, then reduce across the warp
    unsigned int final_mask = 0;
    if (warp_id == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        // Lane i reads warp_results[i] if valid, else 0 (neutral for OR)
        unsigned int my_result = (lane_id < num_warps) ? 
            static_cast<unsigned int>(warp_results[lane_id]) : 0u;
        final_mask = __reduce_or_sync(0xFFFFFFFF, my_result);
    }
    
    // Convert bitmask back to state (only thread 0's result matters to caller)
    return bitmask_to_state(final_mask);
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
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 * @param func_tensor: [B, H, n_func, func_q_len], with strides (stride_b, stride_h, stride_f, stride_q)
 */
template <int N_FUNC>
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
    #pragma unroll
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * KV_BLOCK_SIZE;
        int kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);
        
        // Check if this is a partial kv_block (last block with KV_LEN % KV_BLOCK_SIZE != 0)
        // Partial kv_blocks can NEVER be FULL because positions beyond KV_LEN are invalid
        bool is_kv_block_full = (kv_end == kv_start + KV_BLOCK_SIZE);
        
        // Each thread determines its state for this kv_block
        // Inactive threads (q out of bounds):
        // - check_q_boundary=false: contribute STATE_FULL (q boundary doesn't affect result)
        // - check_q_boundary=true:  contribute STATE_EMPTY (q boundary requires mask)
        int thread_state;
        if (is_active) {
            thread_state = get_token_kv_block_state<N_FUNC>(my_func_ptr, n_func, stride_f, kv_start, kv_end);
        } else {
            thread_state = check_q_boundary ? STATE_EMPTY : STATE_FULL;
        }
        
        // Block-level reduction
        int kv_block_state = reduce_block_state(thread_state, warp_results);
        
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
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 * @param func_tensor: [B, H, n_func, func_q_len], with strides (stride_b, stride_h, stride_f, stride_q)
 */
template <int N_FUNC>
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
    #pragma unroll
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
        // For K2Q (backward), always check q_boundary: inactive threads contribute STATE_EMPTY
        int thread_state;
        if (is_active) {
            thread_state = get_token_kv_block_state<N_FUNC>(my_func_ptr, n_func, stride_f, kv_start, kv_end);
        } else {
            thread_state = STATE_EMPTY;
        }
        
        // Block-level reduction
        int q_block_state = reduce_block_state(thread_state, warp_results);
        
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

// Macro to dispatch Q2K kernel based on N_FUNC template parameter
#define DISPATCH_Q2K_KERNEL(N_FUNC_VAL) \
    create_q2k_block_sparse_from_func_kernel<N_FUNC_VAL><<<grid, block_size, 0, stream>>>( \
        d_func_tensor, \
        stride_b, stride_h, stride_f, stride_q, \
        Q_LEN, KV_LEN, n_func, \
        Q_BLOCK_SIZE, KV_BLOCK_SIZE, \
        num_q_blocks, num_kv_blocks, \
        check_q_boundary, \
        d_mask_block_cnt, d_full_block_cnt, \
        d_block_idx, \
        d_total_mask_blocks, d_total_full_blocks \
    )

// Macro to dispatch K2Q kernel based on N_FUNC template parameter
#define DISPATCH_K2Q_KERNEL(N_FUNC_VAL) \
    create_k2q_block_sparse_from_func_kernel<N_FUNC_VAL><<<grid, block_size, 0, stream>>>( \
        d_func_tensor, \
        stride_b, stride_h, stride_f, stride_q, \
        Q_LEN, KV_LEN, n_func, \
        Q_BLOCK_SIZE, KV_BLOCK_SIZE, \
        num_q_blocks, num_kv_blocks, \
        d_mask_block_cnt, d_full_block_cnt, \
        d_block_idx, \
        d_total_mask_blocks, d_total_full_blocks \
    )

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

    // Dispatch based on n_func value (only odd values: 1, 3, 5, ..., 33)
    // For n_func > 33, use dynamic version (N_FUNC=0)
    switch (n_func) {
        case 1:  DISPATCH_Q2K_KERNEL(1);  break;
        case 3:  DISPATCH_Q2K_KERNEL(3);  break;
        case 5:  DISPATCH_Q2K_KERNEL(5);  break;
        case 7:  DISPATCH_Q2K_KERNEL(7);  break;
        case 9:  DISPATCH_Q2K_KERNEL(9);  break;
        case 11: DISPATCH_Q2K_KERNEL(11); break;
        case 13: DISPATCH_Q2K_KERNEL(13); break;
        case 15: DISPATCH_Q2K_KERNEL(15); break;
        case 17: DISPATCH_Q2K_KERNEL(17); break;
        case 19: DISPATCH_Q2K_KERNEL(19); break;
        case 21: DISPATCH_Q2K_KERNEL(21); break;
        case 23: DISPATCH_Q2K_KERNEL(23); break;
        case 25: DISPATCH_Q2K_KERNEL(25); break;
        case 27: DISPATCH_Q2K_KERNEL(27); break;
        case 29: DISPATCH_Q2K_KERNEL(29); break;
        case 31: DISPATCH_Q2K_KERNEL(31); break;
        case 33: DISPATCH_Q2K_KERNEL(33); break;
        default: DISPATCH_Q2K_KERNEL(0);  break;  // Dynamic version for n_func > 33
    }
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

    // Dispatch based on n_func value (only odd values: 1, 3, 5, ..., 33)
    // For n_func > 33, use dynamic version (N_FUNC=0)
    switch (n_func) {
        case 1:  DISPATCH_K2Q_KERNEL(1);  break;
        case 3:  DISPATCH_K2Q_KERNEL(3);  break;
        case 5:  DISPATCH_K2Q_KERNEL(5);  break;
        case 7:  DISPATCH_K2Q_KERNEL(7);  break;
        case 9:  DISPATCH_K2Q_KERNEL(9);  break;
        case 11: DISPATCH_K2Q_KERNEL(11); break;
        case 13: DISPATCH_K2Q_KERNEL(13); break;
        case 15: DISPATCH_K2Q_KERNEL(15); break;
        case 17: DISPATCH_K2Q_KERNEL(17); break;
        case 19: DISPATCH_K2Q_KERNEL(19); break;
        case 21: DISPATCH_K2Q_KERNEL(21); break;
        case 23: DISPATCH_K2Q_KERNEL(23); break;
        case 25: DISPATCH_K2Q_KERNEL(25); break;
        case 27: DISPATCH_K2Q_KERNEL(27); break;
        case 29: DISPATCH_K2Q_KERNEL(29); break;
        case 31: DISPATCH_K2Q_KERNEL(31); break;
        case 33: DISPATCH_K2Q_KERNEL(33); break;
        default: DISPATCH_K2Q_KERNEL(0);  break;  // Dynamic version for n_func > 33
    }
}

#undef DISPATCH_Q2K_KERNEL
#undef DISPATCH_K2Q_KERNEL

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
