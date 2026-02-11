#include "magi_to_hstu.h"
#include <cassert>

struct Interval{
    int32_t start;
    int32_t end;
};

// MAX_KV_INTERVALS_PER_Q_TOKEN can be overridden via compile flag -DMAX_MAGI_ATTN_SLICES=N
#ifndef MAX_KV_INTERVALS_PER_Q_TOKEN
#define MAX_KV_INTERVALS_PER_Q_TOKEN 16
#endif

// sort intervals for the token in ascending order by start of each interval
__device__ void sort_intervals(Interval *intervals, int num_intervals) {
    // insert sort since num_intervals is small for each q
    for (int i = 1; i < num_intervals; i++) {
        Interval interval = intervals[i];
        int j = i - 1;
        while (j >= 0 && intervals[j].start > interval.start) {
            intervals[j + 1] = intervals[j];
            j--; 
        }
        intervals[j + 1] = interval;
    }
}

// merge overlapping intervals for the token
// return the number of merged intervals
__device__ int merge_intervals(Interval *intervals, int num_intervals) {
    if (num_intervals <= 1) return num_intervals;
    
    int merged_count = 1;
    for (int i = 1; i < num_intervals; i++) {
        if (intervals[i].start > intervals[merged_count - 1].end) {
            intervals[merged_count] = intervals[i];
            merged_count++;
        } else {
            intervals[merged_count - 1].end = max(intervals[i].end, intervals[merged_count - 1].end);
        }
    }
    return merged_count;
}

constexpr int BLOCK_SIZE = 256;

__global__ void magi_to_hstu_kernel(
    const int32_t* __restrict__ q_ranges, // (num_slices, 2)
    const int32_t* __restrict__ k_ranges, // (num_slices, 2)
    const int* __restrict__ mask_types,   // (num_slices,)
    int num_slices,
    int seqlen_q,
    int seqlen_k,
    int32_t *__restrict__ func_out, // [n_max_func, seqlen_q]
    int n_max_func,
    int32_t *__restrict__ max_func_idx) // output: max func_idx across all tokens
{
    __shared__ int s_max[BLOCK_SIZE];
    
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int func_idx = 0;  // initialize to 0 for threads outside seqlen_q
    
    if (q_idx < seqlen_q) {
        Interval intervals[MAX_KV_INTERVALS_PER_Q_TOKEN];

        // traverse all attention slices, collect the interval list for each q
        int num_intervals = 0;
        for (int i = 0; i < num_slices; i++) {
            int2 q_range = reinterpret_cast<const int2*>(q_ranges)[i];
            int q_start = q_range.x;
            int q_end = q_range.y;
            if (q_idx < q_start || q_idx >= q_end) continue;
            
            int2 k_range = reinterpret_cast<const int2*>(k_ranges)[i];
            int slice_k_start = k_range.x;
            int slice_k_end = k_range.y;

            // mask_type: 0=full, 1=causal, 2=inverse, 3=bi-causal
            int mask_type = mask_types[i]; 
            // inverse or bi-causal
            int offset_start = (mask_type & 2) ? (q_idx - q_start) : 0;  
            // causal or bi-causal
            int offset_end = (mask_type & 1) ? (q_end - q_idx - 1) : 0; 

            int k_interval_start = slice_k_start + offset_start;
            int k_interval_end = slice_k_end - offset_end;

            if (k_interval_start >= k_interval_end) continue;
            assert(k_interval_start >= 0 && k_interval_end <= seqlen_k);
            assert(num_intervals < num_slices);
            assert(num_intervals < MAX_KV_INTERVALS_PER_Q_TOKEN);
            intervals[num_intervals].start = k_interval_start;
            intervals[num_intervals].end = k_interval_end;
            num_intervals++;
        }
        // sort intervals by start of each interval
        sort_intervals(intervals, num_intervals);
        // merge intervals
        int num_merged_intervals = merge_intervals(intervals, num_intervals);
        // convert intervals to function
        // function encoding rule:
        //   interval 0: [0, F0)
        //   interval 1: [F1, F2)
        //   interval 2: [F3, F4)
        //   ...
        int max_func_intervals = (n_max_func + 1) / 2;
        assert(num_merged_intervals <= max_func_intervals);
        
        if (num_merged_intervals > 0) {
            // process first interval for staring function
            if (intervals[0].start == 0) {
                // interval start from 0, use [0, F0)
                func_out[q_idx] = intervals[0].end; 
                func_idx = 1;
            } else {
                // interval does not start from 0, use [0, 0), then use [F0, F1)
                func_out[q_idx] = 0;
                func_out[seqlen_q + q_idx] = intervals[0].start;
                func_out[2 * seqlen_q + q_idx] = intervals[0].end;
                func_idx = 3;
            }
            // process remaining intervals
            for (int i = 1; i < num_merged_intervals; i++) {
                int start = intervals[i].start;
                int end = intervals[i].end;
                func_out[func_idx * seqlen_q + q_idx] = start;
                func_out[(func_idx + 1) * seqlen_q + q_idx] = end;
                func_idx += 2;
            }
        }
    }
    
    // Block-level reduction to find max func_idx
    s_max[threadIdx.x] = func_idx;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    // Only thread 0 updates global max
    if (threadIdx.x == 0) {
        atomicMax(max_func_idx, s_max[0]);
    }
}

void launch_magi_to_hstu(
    const int* d_q_ranges,    
    const int* d_k_ranges,     
    const int* d_mask_types,  
    int num_slices,
    int seqlen_q,
    int seqlen_k,
    int* d_func_out,          
    int n_func,
    int* d_max_func_idx,
    cudaStream_t stream
) {
    const int threads = BLOCK_SIZE;
    const int blocks = (seqlen_q + threads - 1) / threads;
    
    magi_to_hstu_kernel<<<blocks, threads, 0, stream>>>(
        d_q_ranges, d_k_ranges, d_mask_types,
        num_slices, seqlen_q, seqlen_k,
        d_func_out, n_func, d_max_func_idx
    );
}

