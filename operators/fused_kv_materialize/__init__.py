"""Fused KV Materialize Operator.

This operator implements fused KV cache materialization, which combines:
1. KV projection output processing
2. RMSNorm on K
3. RoPE (Rotary Position Embedding) on K
4. V passthrough

This is commonly used in LLM inference for efficient KV cache construction.
"""

from __future__ import annotations

import torch

from operators.base import OperatorCase
from operators.fused_kv_materialize import cuda_impl, torch_impl, triton_impl


def build_cases(profile: str = "default") -> list[OperatorCase]:
    """Build benchmark test cases for fused KV materialization.

    Args:
        profile: Benchmark profile name
            - "default": Standard test cases with various sizes
            - "regular": Uniform sizes for regular benchmark
            - "irregular": Variable sequence lengths with padding patterns

    Returns:
        List of OperatorCase objects
    """
    cases: list[OperatorCase] = []

    if profile == "default":
        # Standard cases covering typical LLM dimensions
        configs = [
            # (total_ctx, num_kv_heads, head_dim, rotary_dim, name)
            (128, 8, 128, 128, "small"),
            (256, 16, 128, 128, "medium"),
            (512, 32, 128, 128, "large"),
            (1024, 8, 64, 64, "long_seq_small_head"),
            (512, 8, 256, 128, "big_head_dim"),
        ]
        for total_ctx, num_kv_heads, head_dim, rotary_dim, name in configs:
            cases.append(OperatorCase(
                name=f"fused_kv_{name}",
                summary=f"ctx={total_ctx}, kv_heads={num_kv_heads}, head_dim={head_dim}, rotary_dim={rotary_dim}",
                params={
                    "total_ctx": total_ctx,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "rotary_dim": rotary_dim,
                    "dtype": torch.float16,
                    "family": "default",
                    "sweep_value": total_ctx,
                    "x_label": "Total Context",
                },
            ))

    elif profile == "regular":
        # Regular test cases - uniform sizes
        ctx_sizes = [64, 128, 256, 512, 1024, 2048]
        for ctx in ctx_sizes:
            cases.append(OperatorCase(
                name=f"fused_kv_regular_ctx{ctx}",
                summary=f"ctx={ctx}, kv_heads=8, head_dim=128, rotary_dim=128",
                params={
                    "total_ctx": ctx,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "rotary_dim": 128,
                    "dtype": torch.float16,
                    "family": "regular",
                    "sweep_value": ctx,
                    "x_label": "Total Context",
                },
            ))

        # Sweep over head dimensions
        head_dims = [64, 96, 128, 160, 192, 256]
        for hd in head_dims:
            rotary = min(hd, 128)
            cases.append(OperatorCase(
                name=f"fused_kv_regular_hd{hd}",
                summary=f"ctx=512, kv_heads=8, head_dim={hd}, rotary_dim={rotary}",
                params={
                    "total_ctx": 512,
                    "num_kv_heads": 8,
                    "head_dim": hd,
                    "rotary_dim": rotary,
                    "dtype": torch.float16,
                    "family": "head_dim_sweep",
                    "sweep_value": hd,
                    "x_label": "Head Dimension",
                },
            ))

    elif profile == "irregular":
        # Irregular test cases - variable lengths with padding patterns
        # Simulating real-world scenarios with batched sequences of different lengths

        # Case 1: Highly variable sequence lengths (like prefill batching)
        cases.append(OperatorCase(
            name="fused_kv_irregular_prefill_batch",
            summary="Simulated prefill batch with variable seq lengths (padding)",
            params={
                "total_ctx": 1024,
                "num_kv_heads": 8,
                "head_dim": 128,
                "rotary_dim": 128,
                "dtype": torch.float16,
                "family": "irregular",
                "sweep_value": 1,
                "x_label": "Irregularity Level",
                "irregular_mode": "prefill_batch",
            },
        ))

        # Case 2: Decode scenario with many small sequences
        cases.append(OperatorCase(
            name="fused_kv_irregular_decode_batch",
            summary="Decode batch with many small sequences",
            params={
                "total_ctx": 256,
                "num_kv_heads": 8,
                "head_dim": 128,
                "rotary_dim": 128,
                "dtype": torch.float16,
                "family": "irregular",
                "sweep_value": 2,
                "x_label": "Irregularity Level",
                "irregular_mode": "decode_batch",
            },
        ))

        # Case 3: Mixed sizes with padding ratio variations
        for padding_ratio in [0.1, 0.3, 0.5, 0.7]:
            cases.append(OperatorCase(
                name=f"fused_kv_irregular_padding_{int(padding_ratio*100)}",
                summary=f"Mixed batch with {int(padding_ratio*100)}% padding",
                params={
                    "total_ctx": 512,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "rotary_dim": 128,
                    "dtype": torch.float16,
                    "family": "irregular_padding",
                    "sweep_value": padding_ratio,
                    "x_label": "Padding Ratio",
                    "irregular_mode": "mixed_padding",
                    "padding_ratio": padding_ratio,
                },
            ))

    else:
        raise ValueError(f"Unknown profile for fused_kv_materialize: {profile}")

    return cases


def prepare_inputs(case: OperatorCase, device: str = "cuda") -> dict:
    """Prepare input tensors for fused KV materialization.

    Args:
        case: OperatorCase with benchmark parameters
        device: Device to place tensors on

    Returns:
        Dictionary of input tensors
    """
    torch.manual_seed(42)

    total_ctx = case.params["total_ctx"]
    num_kv_heads = case.params["num_kv_heads"]
    head_dim = case.params["head_dim"]
    rotary_dim = case.params["rotary_dim"]
    dtype = case.params["dtype"]

    kv_size = num_kv_heads * head_dim

    # KV tensor: [total_ctx, kv_size * 2]
    # Contains concatenated K and V projections
    kv = torch.randn((total_ctx, kv_size * 2), device=device, dtype=dtype)

    # K norm weight: [head_dim]
    k_norm_weight = torch.randn((head_dim,), device=device, dtype=dtype)

    # Cos/sin cache for RoPE: [max_pos, rotary_dim]
    # Precomputed cos/sin values for all possible positions
    max_pos = total_ctx + 100  # Add buffer
    cos_sin_cache = torch.randn((max_pos, rotary_dim), device=device, dtype=dtype)

    # Positions: [total_ctx]
    # Position indices for each token
    if case.params.get("irregular_mode") == "prefill_batch":
        # Simulate variable-length sequences in a batch
        # Create positions that look like they come from different sequences
        positions = torch.zeros(total_ctx, device=device, dtype=torch.int64)
        idx = 0
        seq_len = 1
        while idx < total_ctx:
            # Variable sequence lengths
            cur_len = min(seq_len, total_ctx - idx)
            positions[idx:idx + cur_len] = torch.arange(cur_len, device=device)
            idx += cur_len
            seq_len = (seq_len * 2) % 256 + 1
    elif case.params.get("irregular_mode") == "decode_batch":
        # Many small sequences, each with position = 0 or 1
        positions = torch.randint(0, 2, (total_ctx,), device=device, dtype=torch.int64)
    elif case.params.get("irregular_mode") == "mixed_padding":
        # Positions with some gaps (simulating padding)
        padding_ratio = case.params.get("padding_ratio", 0.3)
        positions = torch.arange(total_ctx, device=device, dtype=torch.int64)
        # Add some discontinuities to simulate padding effects
        num_gaps = int(total_ctx * padding_ratio / 10)
        for _ in range(num_gaps):
            gap_start = torch.randint(0, total_ctx - 10, (1,)).item()
            positions[gap_start:] += 5
    else:
        # Regular sequential positions
        positions = torch.arange(total_ctx, device=device, dtype=torch.int64)

    return {
        "kv": kv,
        "k_norm_weight": k_norm_weight,
        "cos_sin_cache": cos_sin_cache,
        "positions": positions,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "eps": 1e-6,
        "input_dtype_name": "float16" if dtype == torch.float16 else str(dtype),
    }


def get_backends():
    """Get available backend implementations.

    Returns:
        List of backend modules
    """
    backends = [torch_impl, triton_impl]
    try:
        # CUDA impl may not be available on all systems
        import importlib.util
        if importlib.util.find_spec("torch.utils.cpp_extension"):
            backends.insert(0, cuda_impl)
    except ImportError:
        pass
    return backends
