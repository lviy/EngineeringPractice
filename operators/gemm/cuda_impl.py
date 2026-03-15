from __future__ import annotations

import torch

BACKEND_NAME = "cuda"


def is_available(inputs: dict[str, torch.Tensor] | None = None) -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA device is not available."
    return True, ""


def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    a = inputs["a"]
    b = inputs["b"]

    if inputs.get("b_is_transposed", False):
        # Torch matmul does not provide a direct FP8 baseline here, so we dequantize the
        # quantized inputs back to FP16 and use a CUDA-backed GEMM as the reference path.
        return torch.matmul(a.to(torch.float16), b.to(torch.float16).transpose(0, 1).contiguous())

    return torch.matmul(a, b)
