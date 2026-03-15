from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

BACKEND_NAME = "triton"


if triton is not None:

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
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(k, BLOCK_K)):
            offs_k_iter = k_block * BLOCK_K + offs_k
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < m) & (offs_k_iter[None, :] < k), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k_iter[:, None] < k) & (offs_bn[None, :] < n), other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)


def is_available() -> tuple[bool, str]:
    if triton is None:
        return False, "Triton is not installed."
    if not torch.cuda.is_available():
        return False, "CUDA device is not available."
    return True, ""


def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    a = inputs["a"]
    b = inputs["b"]
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("This demo Triton GEMM kernel currently expects float16 inputs.")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

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
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        GROUP_M=8,
    )
    return c
