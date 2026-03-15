from __future__ import annotations

import torch

from operators.base import OperatorCase
from operators.gemm import cuda_impl, triton_impl


def build_cases(profile: str = "default") -> list[OperatorCase]:
    if profile == "default":
        return [
            OperatorCase("gemm_s", "M=512, N=512, K=512, dtype=float16", {"m": 512, "n": 512, "k": 512, "dtype": torch.float16}),
            OperatorCase("gemm_m", "M=1024, N=1024, K=1024, dtype=float16", {"m": 1024, "n": 1024, "k": 1024, "dtype": torch.float16}),
            OperatorCase("gemm_l", "M=2048, N=2048, K=1024, dtype=float16", {"m": 2048, "n": 2048, "k": 1024, "dtype": torch.float16}),
            OperatorCase("gemm_rect", "M=4096, N=1024, K=2048, dtype=float16", {"m": 4096, "n": 1024, "k": 2048, "dtype": torch.float16}),
        ]

    raise ValueError(f"Unknown profile for GEMM: {profile}")


def prepare_inputs(case: OperatorCase, device: str = "cuda") -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    m = case.params["m"]
    n = case.params["n"]
    k = case.params["k"]
    dtype = case.params["dtype"]

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    return {"a": a, "b": b}


def get_backends():
    return [cuda_impl, triton_impl]
