from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

BACKEND_NAME = "triton"


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


if triton is not None:

    def _fp16_configs() -> list[triton.Config]:
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        ]


    def _fp8_configs() -> list[triton.Config]:
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=5, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=5, num_warps=4),
        ]


    def _select_configs(meta):
        return _fp8_configs() if meta.get("FP8_OUTPUT", False) else _fp16_configs()


    @triton.autotune(
        configs=_fp16_configs() + _fp8_configs(),
        key=["m", "n", "k", "FP8_OUTPUT", "B_TRANSPOSED"],
    )
    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        FP8_OUTPUT: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        grid_m = tl.cdiv(m, BLOCK_M)
        grid_n = tl.cdiv(n, BLOCK_N)

        width = GROUP_M * grid_n
        group_id = pid // width
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % width) % group_size_m)
        pid_n = (pid % width) // group_size_m

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        if B_TRANSPOSED:
            # FP8 path stores B as a contiguous transposed matrix with logical shape (N, K).
            # To fetch the original B[K, N] tile, we index B_t[N, K].
            b_ptrs = b_ptr + (offs_bn[None, :] * stride_bk + offs_k[:, None] * stride_bn)
        else:
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(k, BLOCK_K)):
            offs_k_iter = k_block * BLOCK_K + offs_k
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < m) & (offs_k_iter[None, :] < k), other=0.0)
            if B_TRANSPOSED:
                b = tl.load(b_ptrs, mask=(offs_bn[None, :] < n) & (offs_k_iter[:, None] < k), other=0.0)
            else:
                b = tl.load(b_ptrs, mask=(offs_k_iter[:, None] < k) & (offs_bn[None, :] < n), other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_K * stride_ak
            if B_TRANSPOSED:
                b_ptrs += BLOCK_K * stride_bn
            else:
                b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)


def is_available(inputs: dict[str, torch.Tensor] | None = None) -> tuple[bool, str]:
    if triton is None:
        return False, "Triton is not installed."
    if not torch.cuda.is_available():
        return False, "CUDA device is not available."
    if inputs is not None and _is_fp8_dtype(inputs["a"].dtype):
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            return False, "Triton FP8 GEMM requires compute capability >= 9.0."
    return True, ""


def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    a = inputs["a"]
    b = inputs["b_triton"]
    b_is_transposed = inputs.get("b_is_transposed", False)
    if a.ndim != 2 or b.ndim != 2:
        raise TypeError("GEMM expects 2D inputs.")

    m, k = a.shape
    n = b.shape[0] if b_is_transposed else b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    fp8_output = _is_fp8_dtype(a.dtype)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        B_TRANSPOSED=b_is_transposed,
        FP8_OUTPUT=fp8_output,
    )

    if fp8_output:
        scaled = c.float() * inputs["scale_a"] * inputs["scale_b"]
        return scaled.to(torch.float16)
    return c
