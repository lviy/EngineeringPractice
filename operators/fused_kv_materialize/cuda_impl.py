"""Fused KV Materialize Operator - CUDA Implementation.

This module provides a CUDA implementation of the fused KV materialization kernel,
which combines RMSNorm(K) + RoPE(K) operations for efficient KV cache construction.

The kernel processes:
- K projection output: applies RMSNorm then RoPE (neox-style rotation)
- V projection output: passes through directly
"""

from __future__ import annotations

import torch
import torch.utils.cpp_extension as cpp_extension

# CUDA kernel source code
CUDA_KERNEL_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

template <typename scalar_t>
__global__ void fused_norm_rope_cuda_kernel(
    const scalar_t* __restrict__ kv_ptr,        // [total_ctx, kv_size * 2]
    const scalar_t* __restrict__ k_norm_weight, // [head_dim]
    const scalar_t* __restrict__ cos_sin_cache, // [max_pos, rotary_dim]
    const int64_t* __restrict__ positions,      // [total_ctx]
    scalar_t* __restrict__ k_out,               // [total_ctx, num_kv_heads, head_dim]
    scalar_t* __restrict__ v_out,               // [total_ctx, num_kv_heads, head_dim]
    const int64_t kv_stride_ctx,
    const int64_t cos_sin_stride_pos,
    const int64_t k_out_stride_ctx,
    const int64_t k_out_stride_head,
    const int64_t v_out_stride_ctx,
    const int64_t v_out_stride_head,
    const int total_ctx,
    const int num_kv_heads,
    const int head_dim,
    const int kv_size,
    const int rotary_dim,
    const int half_rotary_dim,
    const float eps
) {
    const int ctx_id = blockIdx.x;
    const int head_id = blockIdx.y;

    if (ctx_id >= total_ctx) return;

    // Load position
    const int64_t position = positions[ctx_id];

    // Compute base pointers
    const scalar_t* kv_base = kv_ptr + ctx_id * kv_stride_ctx;
    const scalar_t* k_base = kv_base + head_id * head_dim;
    const scalar_t* v_base = kv_base + kv_size + head_id * head_dim;
    scalar_t* k_write = k_out + ctx_id * k_out_stride_ctx + head_id * k_out_stride_head;
    scalar_t* v_write = v_out + ctx_id * v_out_stride_ctx + head_id * v_out_stride_head;

    // Load K and compute RMSNorm
    float k_sum_sq = 0.0f;
    float k_raw_f[256];  // Assuming head_dim <= 256

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = static_cast<float>(k_base[i]);
        k_raw_f[i] = val;
        k_sum_sq += val * val;
    }

    // Reduce sum of squares
    __shared__ float s_sum_sq;
    if (threadIdx.x == 0) s_sum_sq = 0.0f;
    __syncthreads();

    // Atomic add for reduction
    atomicAdd(&s_sum_sq, k_sum_sq);
    __syncthreads();

    // Compute inverse RMS
    float inv_rms = rsqrtf(s_sum_sq / static_cast<float>(head_dim) + eps);

    // Load norm weights and apply normalization
    const scalar_t* cos_sin_base = cos_sin_cache + position * cos_sin_stride_pos;

    // Process in two halves for RoPE
    // First half: load, normalize, rotate
    for (int i = threadIdx.x; i < half_rotary_dim; i += blockDim.x) {
        // Load norm weight
        float norm_w = static_cast<float>(k_norm_weight[i]);

        // Normalize first half
        float k_first = k_raw_f[i] * inv_rms * norm_w;

        // Load and normalize second half
        float k_second_raw = static_cast<float>(k_base[half_rotary_dim + i]);
        float norm_w_second = static_cast<float>(k_norm_weight[half_rotary_dim + i]);
        float k_second = k_second_raw * inv_rms * norm_w_second;

        // Load cos/sin
        float cos_v = static_cast<float>(cos_sin_base[i]);
        float sin_v = static_cast<float>(cos_sin_base[half_rotary_dim + i]);

        // Apply rotation (neox style)
        float k_rot_first = k_first * cos_v - k_second * sin_v;
        float k_rot_second = k_second * cos_v + k_first * sin_v;

        // Store rotated values
        k_write[i] = static_cast<scalar_t>(k_rot_first);
        k_write[half_rotary_dim + i] = static_cast<scalar_t>(k_rot_second);
    }

    // Pass-through for dimensions beyond rotary_dim
    for (int i = threadIdx.x + rotary_dim; i < head_dim; i += blockDim.x) {
        float norm_w = static_cast<float>(k_norm_weight[i]);
        float k_normed = k_raw_f[i] * inv_rms * norm_w;
        k_write[i] = static_cast<scalar_t>(k_normed);
    }

    // Store V (no transform)
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        v_write[i] = v_base[i];
    }
}


std::tuple<torch::Tensor, torch::Tensor> fused_norm_rope_cuda(
    const torch::Tensor& kv,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    double eps
) {
    const int64_t total_ctx = kv.size(0);
    const int64_t kv_size = num_kv_heads * head_dim;

    // Create output tensors
    auto options = kv.options();
    auto k_out = torch::empty({total_ctx, num_kv_heads, head_dim}, options);
    auto v_out = torch::empty_like(k_out);

    if (total_ctx == 0) {
        return std::make_tuple(k_out, v_out);
    }

    // Ensure positions is int64 on correct device
    auto positions_tensor = positions.to(torch::kInt64);
    if (positions_tensor.device() != kv.device()) {
        positions_tensor = positions_tensor.to(kv.device());
    }

    const int half_rotary_dim = rotary_dim / 2;
    const int block_size = (head_dim + 31) / 32 * 32;  // Round up to multiple of 32

    dim3 grid(total_ctx, num_kv_heads);
    dim3 block(block_size > 256 ? 256 : block_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(kv.scalar_type(), "fused_norm_rope_cuda", [&] {
        fused_norm_rope_cuda_kernel<scalar_t><<<grid, block>>>(
            kv.data_ptr<scalar_t>(),
            k_norm_weight.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<scalar_t>(),
            positions_tensor.data_ptr<int64_t>(),
            k_out.data_ptr<scalar_t>(),
            v_out.data_ptr<scalar_t>(),
            kv.stride(0),
            cos_sin_cache.stride(0),
            k_out.stride(0),
            k_out.stride(1),
            v_out.stride(0),
            v_out.stride(1),
            total_ctx,
            num_kv_heads,
            head_dim,
            kv_size,
            rotary_dim,
            half_rotary_dim,
            static_cast<float>(eps)
        );
    });

    return std::make_tuple(k_out, v_out);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_norm_rope_cuda", &fused_norm_rope_cuda, "Fused norm and RoPE CUDA kernel");
}
"""


def load_cuda_extension():
    """Load or compile the CUDA extension."""
    return cpp_extension.load(
        name="fused_kv_materialize_cuda",
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
    """Run fused KV materialization using CUDA kernel.

    Args:
        inputs: Dictionary containing:
            - kv: [total_ctx, kv_size*2] tensor
            - k_norm_weight: [head_dim] tensor
            - cos_sin_cache: [max_pos, rotary_dim] tensor
            - positions: [total_ctx] tensor
            - num_kv_heads: int
            - head_dim: int
            - rotary_dim: int
            - eps: float (optional, default 1e-6)

    Returns:
        Tuple of (k_out, v_out) tensors, each [total_ctx, num_kv_heads, head_dim]
    """
    kv = inputs["kv"]
    k_norm_weight = inputs["k_norm_weight"]
    cos_sin_cache = inputs["cos_sin_cache"]
    positions = inputs["positions"]
    num_kv_heads = inputs["num_kv_heads"]
    head_dim = inputs["head_dim"]
    rotary_dim = inputs["rotary_dim"]
    eps = inputs.get("eps", 1e-6)

    module = get_cuda_module()
    k_out, v_out = module.fused_norm_rope_cuda(
        kv, k_norm_weight, cos_sin_cache, positions,
        num_kv_heads, head_dim, rotary_dim, eps
    )

    return k_out, v_out
