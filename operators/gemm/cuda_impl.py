from __future__ import annotations

import torch

BACKEND_NAME = "cuda"


def is_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA device is not available."
    return True, ""


def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.matmul(inputs["a"], inputs["b"])
