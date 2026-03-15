from __future__ import annotations

import torch

BACKEND_NAME = "cuda"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


def _native_fp8_available() -> bool:
    return hasattr(torch, "_scaled_mm")


def is_available(inputs: dict[str, torch.Tensor] | None = None) -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA device is not available."
    if inputs is not None and _is_fp8_dtype(inputs["a"].dtype):
        if not _native_fp8_available():
            return False, "Native FP8 GEMM requires torch._scaled_mm in the current PyTorch build."
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            return False, "Native FP8 GEMM requires compute capability >= 9.0."
        _, n = inputs["b_native"].shape
        k = inputs["a"].shape[1]
        if n % 16 != 0 or k % 16 != 0:
            return False, "Native FP8 GEMM requires N and K to be multiples of 16."
    return True, ""


def _call_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    call_variants = [
        lambda: torch._scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
            use_fast_accum=True,
        ),
        lambda: torch._scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
        ),
        lambda: torch._scaled_mm(
            a,
            b,
            bias=None,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_result=None,
            out_dtype=out_dtype,
            use_fast_accum=True,
        ),
        lambda: torch._scaled_mm(
            a,
            b,
            bias=None,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_result=None,
            out_dtype=out_dtype,
        ),
    ]

    last_type_error: TypeError | None = None
    for attempt in call_variants:
        try:
            result = attempt()
            return result[0] if isinstance(result, tuple) else result
        except TypeError as exc:
            last_type_error = exc

    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("Unable to dispatch torch._scaled_mm for native FP8 GEMM.")


def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    a = inputs["a"]
    b_native = inputs["b_native"]

    if _is_fp8_dtype(a.dtype):
        return _call_scaled_mm(
            a,
            b_native,
            scale_a=inputs["scale_a"],
            scale_b=inputs["scale_b"],
            out_dtype=torch.float16,
        )

    return torch.matmul(a, b_native)
