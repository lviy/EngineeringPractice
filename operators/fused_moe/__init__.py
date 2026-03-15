from __future__ import annotations

import torch

from operators.base import OperatorCase
from operators.fused_moe import cuda_impl, triton_impl


def build_cases(profile: str = "default") -> list[OperatorCase]:
    if profile == "default":
        return [
            OperatorCase(
                "fused_moe_s",
                "tokens=256, hidden=1024, experts=8, topk=2, dtype=float16",
                {"tokens": 256, "hidden": 1024, "experts": 8, "topk": 2, "dtype": torch.float16},
            ),
            OperatorCase(
                "fused_moe_m",
                "tokens=1024, hidden=2048, experts=16, topk=2, dtype=float16",
                {"tokens": 1024, "hidden": 2048, "experts": 16, "topk": 2, "dtype": torch.float16},
            ),
        ]

    raise ValueError(f"Unknown profile for fused_moe: {profile}")


def prepare_inputs(case: OperatorCase, device: str = "cuda") -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    tokens = case.params["tokens"]
    hidden = case.params["hidden"]
    experts = case.params["experts"]
    topk = case.params["topk"]
    dtype = case.params["dtype"]

    x = torch.randn((tokens, hidden), device=device, dtype=dtype)
    expert_weight = torch.randn((experts, hidden, hidden), device=device, dtype=dtype)
    router_logits = torch.randn((tokens, experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), k=topk, dim=-1)

    return {
        "x": x,
        "expert_weight": expert_weight,
        "topk_weights": topk_weights.to(dtype),
        "topk_ids": topk_ids,
    }


def get_backends():
    return [cuda_impl, triton_impl]
