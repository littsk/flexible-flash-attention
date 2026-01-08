#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "create_block_mask.h"

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

/**
 * Q2K (Forward): Convert function encoding tensor to block sparse tensors.
 * 
 * Forward pass: fix q_block, loop kv_blocks.
 * 
 * Function encoding format (same as magi_to_hstu output):
 *   interval 0: [0, F0)       - valid range (no mask needed)
 *   interval 1: [F1, F2)      - valid range (no mask needed)
 *   interval 2: [F3, F4)      - valid range (no mask needed)
 *   ...
 * 
 * These intervals represent positions where attention scores are VALID (not masked).
 * 
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 *                     func_q_len must be >= Q_LEN + 256 (to avoid bounds checking in kernel)
 *                     n_func must be odd (e.g., 1, 3, 5, 7, ...)
 *                     Supports non-contiguous tensors (uses strides for access)
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 128)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 * @param check_q_boundary: if true (FlexAttention mode), partial q_blocks cannot have FULL kv_blocks;
 *                          if false, partial q_blocks can have FULL kv_blocks (default: false)
 * @param debug: if true, initialize output tensors to -1 for debugging; if false, use uninitialized memory (default: false)
 * 
 * @return tuple of (mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks)
 *         mask_block_cnt: [B, H, num_q_blocks], int32 - number of partial kv_blocks
 *         full_block_cnt: [B, H, num_q_blocks], int32 - number of full kv_blocks
 *         block_idx: [B, H, num_q_blocks, num_kv_blocks], int32 - combined block indices
 *                    full blocks stored left-to-right (0, 1, 2, ...)
 *                    mask blocks stored right-to-left (num_kv_blocks-1, num_kv_blocks-2, ...)
 *         total_mask_blocks: [1], int32 - total number of mask blocks (computed via atomicAdd)
 *         total_full_blocks: [1], int32 - total number of full blocks (computed via atomicAdd)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> create_q2k_block_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 128,
    int KV_BLOCK_SIZE = 128,
    bool check_q_boundary = false,
    bool debug = false
) {
    // Input validation
    TORCH_CHECK(func_tensor.is_cuda(), "func_tensor must be a CUDA tensor");
    TORCH_CHECK(func_tensor.dtype() == torch::kInt32, "func_tensor must be int32");
    TORCH_CHECK(func_tensor.dim() == 4, "func_tensor must be 4D [B, H, n_func, func_q_len]");
    TORCH_CHECK(Q_LEN > 0, "Q_LEN must be positive");
    TORCH_CHECK(KV_LEN > 0, "KV_LEN must be positive");
    TORCH_CHECK(Q_BLOCK_SIZE > 0, "Q_BLOCK_SIZE must be positive");
    TORCH_CHECK(KV_BLOCK_SIZE > 0, "KV_BLOCK_SIZE must be positive");

    int B = func_tensor.size(0);
    int H = func_tensor.size(1);
    int n_func = func_tensor.size(2);
    int func_q_len = func_tensor.size(3);

    // Get strides for non-contiguous tensor support
    int stride_b = func_tensor.stride(0);
    int stride_h = func_tensor.stride(1);
    int stride_f = func_tensor.stride(2);
    int stride_q = func_tensor.stride(3);

    // Check that func_tensor has enough padding to avoid bounds checking in kernel
    TORCH_CHECK(func_q_len >= Q_LEN + 256, 
                "func_tensor.size(3) must be >= Q_LEN + 256 to avoid bounds checking in kernel. "
                "Got func_tensor.size(3)=", func_q_len, ", Q_LEN=", Q_LEN);
    TORCH_CHECK(n_func % 2 == 1, "n_func must be odd (e.g., 1, 3, 5, 7, ...)");

    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);

    // Ensure tensors are on the same device
    at::cuda::CUDAGuard device_guard(func_tensor.device());

    // Allocate output tensors (always contiguous)
    // cnt tensors have size num_q_blocks + 1 for CSR offset format (last element = 0 for cumsum)
    auto opts = func_tensor.options().dtype(torch::kInt32);
    at::Tensor mask_block_cnt, full_block_cnt, block_idx;
    if (debug) {
        // Initialize to -1 for debugging (helps detect unexecuted kernels)
        mask_block_cnt = torch::full({B, H, num_q_blocks + 1}, -1, opts);
        full_block_cnt = torch::full({B, H, num_q_blocks + 1}, -1, opts);
        block_idx = torch::full({B, H, num_q_blocks, num_kv_blocks}, -1, opts);
    } else {
        // Use uninitialized memory for performance
        mask_block_cnt = torch::empty({B, H, num_q_blocks + 1}, opts);
        full_block_cnt = torch::empty({B, H, num_q_blocks + 1}, opts);
        block_idx = torch::empty({B, H, num_q_blocks, num_kv_blocks}, opts);
    }
    
    // Allocate pinned CPU tensors for atomicAdd (GPU can directly access via zero-copy)
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    at::Tensor total_mask_blocks = torch::zeros({1}, cpu_opts);
    at::Tensor total_full_blocks = torch::zeros({1}, cpu_opts);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel (atomicAdd writes directly to pinned memory via zero-copy)
    launch_create_q2k_block_sparse_from_func(
        func_tensor.data_ptr<int>(),
        stride_b, stride_h, stride_f, stride_q,
        B, H, Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        check_q_boundary,
        mask_block_cnt.data_ptr<int>(),
        full_block_cnt.data_ptr<int>(),
        block_idx.data_ptr<int>(),
        total_mask_blocks.data_ptr<int>(),
        total_full_blocks.data_ptr<int>(),
        stream
    );

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks);
}

/**
 * K2Q (Backward): Convert function encoding tensor to block sparse tensors.
 * 
 * Backward pass: fix kv_block, loop q_blocks.
 * Always performs boundary checking (partial q_blocks cannot have FULL status).
 * 
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 128)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 * @param debug: if true, initialize output tensors to -1 for debugging; if false, use uninitialized memory (default: false)
 * 
 * @return tuple of (mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks)
 *         mask_block_cnt: [B, H, num_kv_blocks], int32 - number of partial q_blocks
 *         full_block_cnt: [B, H, num_kv_blocks], int32 - number of full q_blocks
 *         block_idx: [B, H, num_kv_blocks, num_q_blocks], int32 - combined block indices
 *                    full blocks stored left-to-right (0, 1, 2, ...)
 *                    mask blocks stored right-to-left (num_q_blocks-1, num_q_blocks-2, ...)
 *         total_mask_blocks: [1], int32 - total number of mask blocks (computed via atomicAdd)
 *         total_full_blocks: [1], int32 - total number of full blocks (computed via atomicAdd)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> create_k2q_block_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 128,
    int KV_BLOCK_SIZE = 128,
    bool debug = false
) {
    // Input validation
    TORCH_CHECK(func_tensor.is_cuda(), "func_tensor must be a CUDA tensor");
    TORCH_CHECK(func_tensor.dtype() == torch::kInt32, "func_tensor must be int32");
    TORCH_CHECK(func_tensor.dim() == 4, "func_tensor must be 4D [B, H, n_func, func_q_len]");
    TORCH_CHECK(Q_LEN > 0, "Q_LEN must be positive");
    TORCH_CHECK(KV_LEN > 0, "KV_LEN must be positive");
    TORCH_CHECK(Q_BLOCK_SIZE > 0, "Q_BLOCK_SIZE must be positive");
    TORCH_CHECK(KV_BLOCK_SIZE > 0, "KV_BLOCK_SIZE must be positive");

    int B = func_tensor.size(0);
    int H = func_tensor.size(1);
    int n_func = func_tensor.size(2);
    int func_q_len = func_tensor.size(3);

    // Get strides for non-contiguous tensor support
    int stride_b = func_tensor.stride(0);
    int stride_h = func_tensor.stride(1);
    int stride_f = func_tensor.stride(2);
    int stride_q = func_tensor.stride(3);

    // Check that func_tensor has enough padding to avoid bounds checking in kernel
    TORCH_CHECK(func_q_len >= Q_LEN + 256, 
                "func_tensor.size(3) must be >= Q_LEN + 256 to avoid bounds checking in kernel. "
                "Got func_tensor.size(3)=", func_q_len, ", Q_LEN=", Q_LEN);
    TORCH_CHECK(n_func % 2 == 1, "n_func must be odd (e.g., 1, 3, 5, 7, ...)");

    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);

    // Ensure tensors are on the same device
    at::cuda::CUDAGuard device_guard(func_tensor.device());

    // Allocate output tensors (always contiguous)
    // Note: For K2Q, output shape is [B, H, num_kv_blocks, num_q_blocks]
    // cnt tensors have size num_kv_blocks + 1 for CSR offset format (last element = 0 for cumsum)
    auto opts = func_tensor.options().dtype(torch::kInt32);
    at::Tensor mask_block_cnt, full_block_cnt, block_idx;
    if (debug) {
        // Initialize to -1 for debugging (helps detect unexecuted kernels)
        mask_block_cnt = torch::full({B, H, num_kv_blocks + 1}, -1, opts);
        full_block_cnt = torch::full({B, H, num_kv_blocks + 1}, -1, opts);
        block_idx = torch::full({B, H, num_kv_blocks, num_q_blocks}, -1, opts);
    } else {
        // Use uninitialized memory for performance
        mask_block_cnt = torch::empty({B, H, num_kv_blocks + 1}, opts);
        full_block_cnt = torch::empty({B, H, num_kv_blocks + 1}, opts);
        block_idx = torch::empty({B, H, num_kv_blocks, num_q_blocks}, opts);
    }
    
    // Allocate pinned CPU tensors for atomicAdd (GPU can directly access via zero-copy)
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    at::Tensor total_mask_blocks = torch::zeros({1}, cpu_opts);
    at::Tensor total_full_blocks = torch::zeros({1}, cpu_opts);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel (atomicAdd writes directly to pinned memory via zero-copy)
    launch_create_k2q_block_sparse_from_func(
        func_tensor.data_ptr<int>(),
        stride_b, stride_h, stride_f, stride_q,
        B, H, Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        mask_block_cnt.data_ptr<int>(),
        full_block_cnt.data_ptr<int>(),
        block_idx.data_ptr<int>(),
        total_mask_blocks.data_ptr<int>(),
        total_full_blocks.data_ptr<int>(),
        stream
    );

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks);
}

/**
 * Compact block indices from BHQK format to linear sparse format (CSR-like).
 * 
 * Converts the combined block_idx tensor to compact mask_block_idx and full_block_idx tensors
 * using prefix sum (cumulative sum) to compute offsets.
 * 
 * Input format (from create_q2k/k2q_block_sparse_from_func):
 *   - mask_block_cnt: [B, H, num_blocks], count of mask blocks per row
 *   - full_block_cnt: [B, H, num_blocks], count of full blocks per row
 *   - block_idx: [B, H, num_blocks, max_blocks], combined indices
 *       - full blocks stored left-to-right: block_idx[..., 0:full_cnt]
 *       - mask blocks stored right-to-left: block_idx[..., max_blocks-mask_cnt:max_blocks]
 *   - total_mask_blocks: [1], total mask blocks from kernel's atomicAdd
 *   - total_full_blocks: [1], total full blocks from kernel's atomicAdd
 * 
 * Output format (LinearBlockSparseTensorsTorch compatible):
 *   - mask_block_cnt: [n_blocks] (flattened)
 *   - mask_block_offset: [n_blocks + 1], prefix sum of mask_block_cnt
 *   - mask_block_idx: [total_mask_blocks], compact indices
 *   - full_block_cnt: [n_blocks] (flattened)
 *   - full_block_offset: [n_blocks + 1], prefix sum of full_block_cnt
 *   - full_block_idx: [total_full_blocks], compact indices
 * 
 * Memory savings: O(num_blocks * max_blocks) -> O(total_mask_blocks + total_full_blocks)
 * 
 * @param mask_block_cnt: [B, H, num_blocks], int32
 * @param full_block_cnt: [B, H, num_blocks], int32
 * @param block_idx: [B, H, num_blocks, max_blocks], int32
 * @param total_mask_blocks: [1], int32 - total mask blocks from atomicAdd
 * @param total_full_blocks: [1], int32 - total full blocks from atomicAdd
 * 
 * @return tuple of (mask_block_cnt_flat, mask_block_offset, mask_block_idx_compact,
 *                   full_block_cnt_flat, full_block_offset, full_block_idx_compact)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> 
compact_block_idx(
    const at::Tensor& mask_block_cnt,
    const at::Tensor& full_block_cnt,
    const at::Tensor& block_idx,
    const at::Tensor& total_mask_blocks,
    const at::Tensor& total_full_blocks
) {
    // Input validation
    TORCH_CHECK(mask_block_cnt.is_cuda(), "mask_block_cnt must be a CUDA tensor");
    TORCH_CHECK(full_block_cnt.is_cuda(), "full_block_cnt must be a CUDA tensor");
    TORCH_CHECK(block_idx.is_cuda(), "block_idx must be a CUDA tensor");
    // total_mask_blocks and total_full_blocks are pinned CPU tensors (from async D2H)
    TORCH_CHECK(!total_mask_blocks.is_cuda() && total_mask_blocks.is_pinned(), 
                "total_mask_blocks must be a CPU tensor");
    TORCH_CHECK(!total_full_blocks.is_cuda() && total_full_blocks.is_pinned(), 
                "total_full_blocks must be a CPU tensor");
    TORCH_CHECK(mask_block_cnt.dtype() == torch::kInt32, "mask_block_cnt must be int32");
    TORCH_CHECK(full_block_cnt.dtype() == torch::kInt32, "full_block_cnt must be int32");
    TORCH_CHECK(block_idx.dtype() == torch::kInt32, "block_idx must be int32");
    TORCH_CHECK(total_mask_blocks.dtype() == torch::kInt32, "total_mask_blocks must be int32");
    TORCH_CHECK(total_full_blocks.dtype() == torch::kInt32, "total_full_blocks must be int32");
    TORCH_CHECK(mask_block_cnt.dim() == 3, "mask_block_cnt must be 3D [B, H, num_blocks+1]");
    TORCH_CHECK(full_block_cnt.dim() == 3, "full_block_cnt must be 3D [B, H, num_blocks+1]");
    TORCH_CHECK(block_idx.dim() == 4, "block_idx must be 4D [B, H, num_blocks, max_blocks]");
    
    // Ensure same device
    at::cuda::CUDAGuard device_guard(mask_block_cnt.device());
    
    int64_t B = mask_block_cnt.size(0);
    int64_t H = mask_block_cnt.size(1);
    int64_t num_blocks_plus_one = mask_block_cnt.size(2);  // num_blocks + 1 for CSR format
    int64_t num_blocks = block_idx.size(2);
    int64_t max_blocks = block_idx.size(3);
    int64_t n_blocks_flat = B * H * num_blocks_plus_one;
    
    auto opts = mask_block_cnt.options();
    
    // Ensure contiguous (no flatten needed, data_ptr accesses underlying memory directly)
    at::Tensor mask_cnt = mask_block_cnt.contiguous();
    at::Tensor full_cnt = full_block_cnt.contiguous();
    at::Tensor block_idx_cont = block_idx.contiguous();
    
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Synchronize stream to ensure atomicAdd to pinned memory is complete
    cudaStreamSynchronize(stream);

    // Read total counts directly from CPU memory (no .item() overhead)
    int total_mask = static_cast<int>(*total_mask_blocks.data_ptr<int>());
    int total_full = static_cast<int>(*total_full_blocks.data_ptr<int>());

    // Allocate compact idx tensors (empty if total is 0)
    at::Tensor mask_block_idx_compact = torch::empty({total_mask}, opts);
    at::Tensor full_block_idx_compact = torch::empty({total_full}, opts);
    
    // Step 1: Compute inclusive prefix sums (offset format)
    // cnt = [0, c0, c1, ..., cn-1], cumsum = [0, c0, c0+c1, ..., sum] which is the offset
    // Use simple single-CTA per (b,h) scan - no temp buffer needed
    // Output shape: (B, H, num_blocks_plus_one) to match input cnt shape
    at::Tensor mask_block_offset = torch::empty({B, H, num_blocks_plus_one}, opts);
    at::Tensor full_block_offset = torch::empty({B, H, num_blocks_plus_one}, opts);
    
    launch_dual_inclusive_sum(
        mask_cnt.data_ptr<int>(),
        full_cnt.data_ptr<int>(),
        mask_block_offset.data_ptr<int>(),
        full_block_offset.data_ptr<int>(),
        B * H * num_blocks_plus_one,  // total elements to scan
        stream
    );
        
    // Step 2: Extract indices using CUDA kernel
    // Note: n_blocks is B*H*num_blocks (not including +1), offset is indexed 1:n+1
    int64_t n_blocks = B * H * num_blocks;
    if (total_mask > 0 || total_full > 0) {
        launch_extract_compact_indices(
            block_idx_cont.data_ptr<int>(),
            mask_cnt.data_ptr<int>() + 1,  // Skip leading 0, start from actual counts
            full_cnt.data_ptr<int>() + 1,  // Skip leading 0, start from actual counts
            mask_block_offset.data_ptr<int>(),  // offset[0]=0, offset[i+1]=sum(cnt[0:i])
            full_block_offset.data_ptr<int>(),
            n_blocks,
            max_blocks,
            mask_block_idx_compact.data_ptr<int>(),
            full_block_idx_compact.data_ptr<int>(),
            stream
        );
    }
    
    // Return tensors with shape (B, H, num_blocks_plus_one)
    return std::make_tuple(
        mask_cnt, mask_block_offset, mask_block_idx_compact,
        full_cnt, full_block_offset, full_block_idx_compact
    );
}

/**
 * Q2K CSR (Forward): Convert function encoding tensor directly to CSR sparse format.
 * 
 * This is a convenience function that combines create_q2k_block_sparse_from_func and compact_block_idx.
 * 
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 256)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 * @param check_q_boundary: if true (FlexAttention mode), partial q_blocks cannot have FULL kv_blocks
 * 
 * @return tuple of (mask_block_cnt, mask_block_offset, mask_block_idx,
 *                   full_block_cnt, full_block_offset, full_block_idx)
 *         All tensors in CSR format for efficient sparse operations.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
create_q2k_csr_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 256,
    int KV_BLOCK_SIZE = 128,
    bool check_q_boundary = false
) {
    // Step 1: Create block sparse tensors
    auto [mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks] = 
        create_q2k_block_sparse_from_func(
            func_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, 
            check_q_boundary, false  // debug=false for performance
        );
    
    // Step 2: Compact to CSR format
    return compact_block_idx(
        mask_block_cnt, full_block_cnt, block_idx,
        total_mask_blocks, total_full_blocks
    );
}

/**
 * K2Q CSR (Backward): Convert function encoding tensor directly to CSR sparse format.
 * 
 * This is a convenience function that combines create_k2q_block_sparse_from_func and compact_block_idx.
 * 
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 128)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 * 
 * @return tuple of (mask_block_cnt, mask_block_offset, mask_block_idx,
 *                   full_block_cnt, full_block_offset, full_block_idx)
 *         All tensors in CSR format for efficient sparse operations.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
create_k2q_csr_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 128,
    int KV_BLOCK_SIZE = 128
) {
    // Step 1: Create block sparse tensors
    auto [mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks] = 
        create_k2q_block_sparse_from_func(
            func_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, 
            false  // debug=false for performance
        );
    
    // Step 2: Compact to CSR format
    return compact_block_idx(
        mask_block_cnt, full_block_cnt, block_idx,
        total_mask_blocks, total_full_blocks
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for creating block sparse tensors (FlexAttention/BlockSparseTensorsTorch format)";
    
    m.def("create_q2k_block_sparse_from_func", &create_q2k_block_sparse_from_func,
          "Q2K (Forward): Convert function encoding tensor to block sparse tensors. "
          "Fix q_block, loop kv_blocks. "
          "Returns (mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks) "
          "where block_idx contains full blocks left-to-right and mask blocks right-to-left. "
          "total counts are computed via atomicAdd in kernel.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE")  = 256,  // default: 256x128 for forward pass
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("check_q_boundary") = false,
          py::arg("debug") = false);
    
    m.def("create_k2q_block_sparse_from_func", &create_k2q_block_sparse_from_func,
          "K2Q (Backward): Convert function encoding tensor to block sparse tensors. "
          "Fix kv_block, loop q_blocks. Always checks boundary. "
          "Returns (mask_block_cnt, full_block_cnt, block_idx, total_mask_blocks, total_full_blocks) "
          "where block_idx contains full blocks left-to-right and mask blocks right-to-left. "
          "total counts are computed via atomicAdd in kernel.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 128,  // default: 128x128 for backward pass
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("debug") = false);
    
    m.def("compact_block_idx", &compact_block_idx,
          "Compact block indices from BHQK format to linear sparse format (CSR-like). "
          "Converts combined block_idx tensor to compact mask_block_idx and full_block_idx. "
          "Uses total counts from kernel's atomicAdd to avoid CPU-GPU sync. "
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx).",
          py::arg("mask_block_cnt"),
          py::arg("full_block_cnt"),
          py::arg("block_idx"),
          py::arg("total_mask_blocks"),
          py::arg("total_full_blocks"));
    
    m.def("create_q2k_csr_sparse_from_func", &create_q2k_csr_sparse_from_func,
          "Q2K CSR (Forward): Convert function encoding tensor directly to CSR sparse format. "
          "Combines create_q2k_block_sparse_from_func and compact_block_idx in one call. "
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx).",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 256,
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("check_q_boundary") = false);
    
    m.def("create_k2q_csr_sparse_from_func", &create_k2q_csr_sparse_from_func,
          "K2Q CSR (Backward): Convert function encoding tensor directly to CSR sparse format. "
          "Combines create_k2q_block_sparse_from_func and compact_block_idx in one call. "
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx).",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 128,
          py::arg("KV_BLOCK_SIZE") = 128);
}
