from __future__ import annotations

import torch

BACKEND_NAME = "cuda"


def is_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA device is not available."
    return True, ""


def run(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    x = inputs["x"]
    expert_weight = inputs["expert_weight"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]

    tokens, hidden = x.shape
    topk = topk_ids.shape[1]
    output = torch.zeros((tokens, hidden), device=x.device, dtype=x.dtype)

    for token_idx in range(tokens):
        token_out = torch.zeros((hidden,), device=x.device, dtype=x.dtype)
        for route_idx in range(topk):
            expert_id = int(topk_ids[token_idx, route_idx].item())
            weight = topk_weights[token_idx, route_idx]
            token_out += weight * torch.matmul(x[token_idx], expert_weight[expert_id])
        output[token_idx] = token_out

    return output
