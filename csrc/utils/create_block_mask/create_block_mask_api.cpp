#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "create_block_mask.h"

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// TODO: Combining the full and mask tensors into one full_mask tensor can reduce the temporary memory overhead by half.
// TODO: Count the num in kernel with persistent mode to avoid d2h copy.

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
 * @return tuple of (mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx)
 *         mask_block_cnt: [B, H, num_q_blocks], int32 - number of partial kv_blocks
 *         mask_block_idx: [B, H, num_q_blocks, num_kv_blocks], int32 - partial kv_block indices
 *         full_block_cnt: [B, H, num_q_blocks], int32 - number of full kv_blocks
 *         full_block_idx: [B, H, num_q_blocks, num_kv_blocks], int32 - full kv_block indices
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> create_q2k_block_sparse_from_func(
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
    auto opts = func_tensor.options().dtype(torch::kInt32);
    at::Tensor mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx;
    if (debug) {
        // Initialize to -1 for debugging (helps detect unexecuted kernels)
        mask_block_cnt = torch::full({B, H, num_q_blocks}, -1, opts);
        mask_block_idx = torch::full({B, H, num_q_blocks, num_kv_blocks}, -1, opts);
        full_block_cnt = torch::full({B, H, num_q_blocks}, -1, opts);
        full_block_idx = torch::full({B, H, num_q_blocks, num_kv_blocks}, -1, opts);
    } else {
        // Use uninitialized memory for performance
        mask_block_cnt = torch::empty({B, H, num_q_blocks}, opts);
        mask_block_idx = torch::empty({B, H, num_q_blocks, num_kv_blocks}, opts);
        full_block_cnt = torch::empty({B, H, num_q_blocks}, opts);
        full_block_idx = torch::empty({B, H, num_q_blocks, num_kv_blocks}, opts);
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    launch_create_q2k_block_sparse_from_func(
        func_tensor.data_ptr<int>(),
        stride_b, stride_h, stride_f, stride_q,
        B, H, Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        check_q_boundary,
        mask_block_cnt.data_ptr<int>(),
        mask_block_idx.data_ptr<int>(),
        full_block_cnt.data_ptr<int>(),
        full_block_idx.data_ptr<int>(),
        stream
    );

    return std::make_tuple(mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx);
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
 * @return tuple of (mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx)
 *         mask_block_cnt: [B, H, num_kv_blocks], int32 - number of partial q_blocks
 *         mask_block_idx: [B, H, num_kv_blocks, num_q_blocks], int32 - partial q_block indices
 *         full_block_cnt: [B, H, num_kv_blocks], int32 - number of full q_blocks
 *         full_block_idx: [B, H, num_kv_blocks, num_q_blocks], int32 - full q_block indices
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> create_k2q_block_sparse_from_func(
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
    auto opts = func_tensor.options().dtype(torch::kInt32);
    at::Tensor mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx;
    if (debug) {
        // Initialize to -1 for debugging (helps detect unexecuted kernels)
        mask_block_cnt = torch::full({B, H, num_kv_blocks}, -1, opts);
        mask_block_idx = torch::full({B, H, num_kv_blocks, num_q_blocks}, -1, opts);
        full_block_cnt = torch::full({B, H, num_kv_blocks}, -1, opts);
        full_block_idx = torch::full({B, H, num_kv_blocks, num_q_blocks}, -1, opts);
    } else {
        // Use uninitialized memory for performance
        mask_block_cnt = torch::empty({B, H, num_kv_blocks}, opts);
        mask_block_idx = torch::empty({B, H, num_kv_blocks, num_q_blocks}, opts);
        full_block_cnt = torch::empty({B, H, num_kv_blocks}, opts);
        full_block_idx = torch::empty({B, H, num_kv_blocks, num_q_blocks}, opts);
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    launch_create_k2q_block_sparse_from_func(
        func_tensor.data_ptr<int>(),
        stride_b, stride_h, stride_f, stride_q,
        B, H, Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        mask_block_cnt.data_ptr<int>(),
        mask_block_idx.data_ptr<int>(),
        full_block_cnt.data_ptr<int>(),
        full_block_idx.data_ptr<int>(),
        stream
    );

    return std::make_tuple(mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for creating block sparse tensors (FlexAttention/BlockSparseTensorsTorch format)";
    
    m.def("create_q2k_block_sparse_from_func", &create_q2k_block_sparse_from_func,
          "Q2K (Forward): Convert function encoding tensor to block sparse tensors. "
          "Fix q_block, loop kv_blocks.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 128,
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("check_q_boundary") = false,
          py::arg("debug") = false);
    
    m.def("create_k2q_block_sparse_from_func", &create_k2q_block_sparse_from_func,
          "K2Q (Backward): Convert function encoding tensor to block sparse tensors. "
          "Fix kv_block, loop q_blocks. Always checks boundary.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 128,
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("debug") = false);
}
