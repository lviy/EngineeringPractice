from __future__ import annotations

import torch

from operators.base import OperatorCase
from operators.gemm import cuda_impl, triton_impl


def _make_case(
    name: str,
    m: int,
    n: int,
    k: int,
    dtype_name: str,
    family: str,
    sweep_value: int,
    x_label: str,
) -> OperatorCase:
    return OperatorCase(
        name=name,
        summary=f"M={m}, N={n}, K={k}, dtype={dtype_name}",
        params={
            "m": m,
            "n": n,
            "k": k,
            "dtype_name": dtype_name,
            "family": family,
            "sweep_value": sweep_value,
            "x_label": x_label,
        },
    )


def _resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise RuntimeError(f"Current PyTorch build does not expose dtype '{dtype_name}'.")


def build_cases(profile: str = "default") -> list[OperatorCase]:
    if profile == "default":
        square_sizes = [1024, 2048, 3072, 4096, 6144, 8192]
        mlp_token_sizes = [1024, 2048, 4096, 6144]
        attn_token_sizes = [1024, 2048, 4096, 6144]
        cases: list[OperatorCase] = []

        for dtype_name in ("float16", "float8_e5m2"):
            dtype_tag = "fp16" if dtype_name == "float16" else "fp8"

            for size in square_sizes:
                cases.append(
                    _make_case(
                        name=f"gemm_{dtype_tag}_square_{size}",
                        m=size,
                        n=size,
                        k=size,
                        dtype_name=dtype_name,
                        family="square",
                        sweep_value=size,
                        x_label="Matrix Size (M=N=K)",
                    )
                )

            for tokens in mlp_token_sizes:
                cases.append(
                    _make_case(
                        name=f"gemm_{dtype_tag}_mlp_{tokens}",
                        m=tokens,
                        n=11008,
                        k=4096,
                        dtype_name=dtype_name,
                        family="mlp",
                        sweep_value=tokens,
                        x_label="Token Count (M)",
                    )
                )

            for tokens in attn_token_sizes:
                cases.append(
                    _make_case(
                        name=f"gemm_{dtype_tag}_attn_{tokens}",
                        m=tokens,
                        n=4096,
                        k=4096,
                        dtype_name=dtype_name,
                        family="attn",
                        sweep_value=tokens,
                        x_label="Token Count (M)",
                    )
                )

        return cases

    if profile == "smoke":
        return [
            _make_case("gemm_fp16_smoke", 1024, 1024, 1024, "float16", "square", 1024, "Matrix Size (M=N=K)"),
            _make_case("gemm_fp8_smoke", 2048, 2048, 2048, "float8_e5m2", "square", 2048, "Matrix Size (M=N=K)"),
        ]

    raise ValueError(f"Unknown profile for GEMM: {profile}")


def prepare_inputs(case: OperatorCase, device: str = "cuda") -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    m = case.params["m"]
    n = case.params["n"]
    k = case.params["k"]
    dtype_name = case.params["dtype_name"]
    dtype = _resolve_dtype(dtype_name)

    base_a = torch.randn((m, k), device=device, dtype=torch.float16)
    base_b = torch.randn((k, n), device=device, dtype=torch.float16)

    if dtype == torch.float16:
        a = base_a
        b = base_b
        b_is_transposed = False
    else:
        a = base_a.to(dtype)
        b = base_b.transpose(0, 1).contiguous().to(dtype)
        b_is_transposed = True

    return {
        "a": a,
        "b": b,
        "input_dtype_name": dtype_name,
        "b_is_transposed": b_is_transposed,
        "x_label": case.params["x_label"],
    }


def get_backends():
    return [cuda_impl, triton_impl]
