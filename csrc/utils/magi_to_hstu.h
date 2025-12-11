#pragma once

#include <cuda_runtime.h>

void launch_magi_to_hstu(
    const int* d_q_ranges,    
    const int* d_k_ranges,     
    const int* d_mask_types,  
    int num_slices,
    int seqlen_q,
    int seqlen_k,
    int* d_func_out,          
    int n_func,
    cudaStream_t stream = nullptr
);

