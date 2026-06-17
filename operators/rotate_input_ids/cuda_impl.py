"""Rotate Input IDs Operator - CUDA Implementation.

This module provides a CUDA implementation of the rotate_input_ids kernel,
which shifts input_ids sequences and inserts new tokens at specific positions.

This is commonly used in speculative decoding and multi-layer Eagle models
for efficient sequence manipulation.
"""

from __future__ import annotations

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA kernel source code
CUDA_KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void rotate_input_ids_cuda_kernel(
    scalar_t* __restrict__ input_ids_ptr,
    const int64_t* __restrict__ extend_start_loc_ptr,
    const int64_t* __restrict__ extend_seq_lens_ptr,
    const scalar_t* __restrict__ topk_index_ptr,
    const int64_t* __restrict__ select_index_ptr,
    const int BLOCK_SIZE
) {
    const int pid = blockIdx.x;

    const int64_t start_loc = extend_start_loc_ptr[pid];
    const int64_t seq_len = extend_seq_lens_ptr[pid];
    const scalar_t new_token = topk_index_ptr[pid];

    const int64_t num_elements_to_shift = seq_len - 1;

    // Shift elements: each element moves to position - 1
    for (int off = 0; off < num_elements_to_shift; off += BLOCK_SIZE) {
        for (int i = threadIdx.x; i < BLOCK_SIZE && (off + i) < num_elements_to_shift; i += blockDim.x) {
            const int64_t read_offset = off + i;
            const scalar_t val = input_ids_ptr[start_loc + read_offset + 1];
            input_ids_ptr[start_loc + read_offset] = val;
        }
        __syncthreads();
    }

    // Write new token at the last position
    if (seq_len > 0 && threadIdx.x == 0) {
        int64_t last_pos;
        if (select_index_ptr != nullptr) {
            last_pos = select_index_ptr[pid];
        } else {
            last_pos = start_loc + seq_len - 1;
        }
        input_ids_ptr[last_pos] = new_token;
    }
}


torch::Tensor rotate_input_ids_cuda(
    torch::Tensor input_ids,
    torch::Tensor extend_start_loc,
    torch::Tensor extend_seq_lens,
    torch::Tensor topk_index,
    c10::optional<torch::Tensor> select_index_opt
) {
    const int64_t batch_size = extend_seq_lens.size(0);
    const int BLOCK_SIZE = 128;

    // Ensure tensors are on the same device
    auto device = input_ids.device();
    extend_start_loc = extend_start_loc.to(device).to(torch::kInt64);
    extend_seq_lens = extend_seq_lens.to(device).to(torch::kInt64);
    topk_index = topk_index.to(device).to(input_ids.scalar_type());

    const int64_t* select_index_ptr = nullptr;
    torch::Tensor select_index;
    if (select_index_opt.has_value()) {
        select_index = select_index_opt.value().to(device).to(torch::kInt64);
        select_index_ptr = select_index.data_ptr<int64_t>();
    }

    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);

    AT_DISPATCH_INT_TYPES_AND_HALF(input_ids.scalar_type(), "rotate_input_ids_cuda", [&] {
        rotate_input_ids_cuda_kernel<scalar_t><<<grid, block>>>(
            input_ids.data_ptr<scalar_t>(),
            extend_start_loc.data_ptr<int64_t>(),
            extend_seq_lens.data_ptr<int64_t>(),
            topk_index.data_ptr<scalar_t>(),
            select_index_ptr,
            BLOCK_SIZE
        );
    });

    return input_ids;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotate_input_ids_cuda", &rotate_input_ids_cuda, "Rotate input IDs CUDA kernel");
}
"""


def load_cuda_extension():
    """Load or compile the CUDA extension."""
    return cpp_extension.load(
        name="rotate_input_ids_cuda",
        sources=[],
        extra_cxx_flags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        source_code=CUDA_KERNEL_SOURCE,
        is_python_module=True,
    )


# Lazy-loaded module
_cuda_module = None


def get_cuda_module():
    """Get or compile the CUDA module."""
    global _cuda_module
    if _cuda_module is None:
        _cuda_module = load_cuda_extension()
    return _cuda_module


BACKEND_NAME = "cuda"


def is_available() -> tuple[bool, str]:
    """Check if CUDA backend is available."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    return True, ""


def run(inputs: dict) -> torch.Tensor:
    """Run rotate_input_ids using CUDA kernel.

    Args:
        inputs: Dictionary containing:
            - input_ids: Flattened input IDs tensor
            - extend_start_loc: Start locations for each sequence
            - extend_seq_lens: Sequence lengths
            - topk_index: New tokens to insert
            - select_index: Optional custom positions for new tokens

    Returns:
        Modified input_ids tensor
    """
    input_ids = inputs["input_ids"]
    extend_start_loc = inputs["extend_start_loc"]
    extend_seq_lens = inputs["extend_seq_lens"]
    topk_index = inputs["topk_index"]
    select_index = inputs.get("select_index", None)

    module = get_cuda_module()
    result = module.rotate_input_ids_cuda(
        input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index
    )

    return result