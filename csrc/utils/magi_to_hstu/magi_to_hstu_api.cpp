#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "magi_to_hstu.h"

/**
 * Convert MagiAttention mask format to HSTU function encoding format.
 *
 * @param q_ranges: [num_slices, 2], int32, query ranges for each attention slice
 * @param k_ranges: [num_slices, 2], int32, key ranges for each attention slice
 * @param mask_types: [num_slices], int32, mask type for each slice
 *                    (0=full, 1=causal, 2=inverse, 3=bi-causal)
 * @param seqlen_q: query sequence length
 * @param seqlen_k: key sequence length
 * @param n_max_func: maximum number of functions in output (default: 5)
 *
 * @return func_out: [n_max_func, seqlen_q], int32, function encoding
 *         Function encoding rule:
 *           interval 0: [0, F0)
 *           interval 1: [F1, F2)
 *           interval 2: [F3, F4)
 *           ...
 *         -1 indicates unused slot
 */
at::Tensor magi_to_hstu(
    const at::Tensor& q_ranges,
    const at::Tensor& k_ranges,
    const at::Tensor& mask_types,
    int seqlen_q,
    int seqlen_k,
    int n_max_func = 5
) {
    // Input validation
    TORCH_CHECK(q_ranges.is_cuda(), "q_ranges must be a CUDA tensor");
    TORCH_CHECK(k_ranges.is_cuda(), "k_ranges must be a CUDA tensor");
    TORCH_CHECK(mask_types.is_cuda(), "mask_types must be a CUDA tensor");

    TORCH_CHECK(q_ranges.is_contiguous(), "q_ranges must be contiguous");
    TORCH_CHECK(k_ranges.is_contiguous(), "k_ranges must be contiguous");
    TORCH_CHECK(mask_types.is_contiguous(), "mask_types must be contiguous");

    TORCH_CHECK(q_ranges.dtype() == torch::kInt32, "q_ranges must be int32");
    TORCH_CHECK(k_ranges.dtype() == torch::kInt32, "k_ranges must be int32");
    TORCH_CHECK(mask_types.dtype() == torch::kInt32, "mask_types must be int32");

    TORCH_CHECK(q_ranges.dim() == 2, "q_ranges must be 2D [num_slices, 2]");
    TORCH_CHECK(k_ranges.dim() == 2, "k_ranges must be 2D [num_slices, 2]");
    TORCH_CHECK(mask_types.dim() == 1, "mask_types must be 1D [num_slices]");

    int num_slices = q_ranges.size(0);
    TORCH_CHECK(q_ranges.size(1) == 2, "q_ranges second dim must be 2");
    TORCH_CHECK(k_ranges.size(0) == num_slices, "k_ranges first dim must match num_slices");
    TORCH_CHECK(k_ranges.size(1) == 2, "k_ranges second dim must be 2");
    TORCH_CHECK(mask_types.size(0) == num_slices, "mask_types must have num_slices elements");

    TORCH_CHECK(seqlen_q > 0, "seqlen_q must be positive");
    TORCH_CHECK(seqlen_k > 0, "seqlen_k must be positive");
    TORCH_CHECK(n_max_func > 0, "n_max_func must be positive");

    // Ensure all tensors are on the same device
    at::cuda::CUDAGuard device_guard(q_ranges.device());

    // Allocate output tensor, initialized to -1 (unused)
    auto opts = q_ranges.options().dtype(torch::kInt32);
    at::Tensor func_out = torch::full({n_max_func, seqlen_q}, 0, opts);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    launch_magi_to_hstu(
        q_ranges.data_ptr<int>(),
        k_ranges.data_ptr<int>(),
        mask_types.data_ptr<int>(),
        num_slices,
        seqlen_q,
        seqlen_k,
        func_out.data_ptr<int>(),
        n_max_func,
        stream
    );

    return func_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA Magi-to-HSTU mask conversion";
    m.def("magi_to_hstu", &magi_to_hstu,
          "Convert MagiAttention mask format to HSTU function encoding",
          py::arg("q_ranges"),
          py::arg("k_ranges"),
          py::arg("mask_types"),
          py::arg("seqlen_q"),
          py::arg("seqlen_k"),
          py::arg("n_max_func") = 5);
}
