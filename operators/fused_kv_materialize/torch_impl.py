"""Fused KV Materialize Operator - PyTorch Implementation.

This module provides a pure PyTorch reference implementation of the fused KV
materialization kernel. It performs RMSNorm + RoPE fusion using standard PyTorch
operations, serving as both a reference implementation and a baseline for benchmarks.
"""

from __future__ import annotations

import torch

BACKEND_NAME = "torch"


def is_available() -> tuple[bool, str]:
    """Check if PyTorch backend is available."""
    return True, ""


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization.

    Args:
        x: Input tensor [..., head_dim]
        weight: Weight tensor [head_dim]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as x
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def apply_rotary_pos_emb(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings (neox-style).

    Args:
        k: Input tensor [..., rotary_dim]
        cos: Cosine values [..., half_rotary_dim]
        sin: Sine values [..., half_rotary_dim]

    Returns:
        Rotated tensor with same shape as k
    """
    if k.shape[-1] == 0:
        return k

    half_rotary_dim = k.shape[-1] // 2
    k1 = k[..., :half_rotary_dim]
    k2 = k[..., half_rotary_dim:]

    # Neox-style rotation
    k_rotated = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin,
    ], dim=-1)

    return k_rotated


def run(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused KV materialization using PyTorch operations.

    This implements the same logic as the Triton/CUDA kernels:
    1. Split KV into K and V components
    2. Apply RMSNorm to K
    3. Apply RoPE to the rotary portion of K
    4. Pass through V unchanged

    Args:
        inputs: Dictionary containing:
            - kv: [total_ctx, kv_size*2] tensor
            - k_norm_weight: [head_dim] tensor
            - cos_sin_cache: [max_pos, rotary_dim] tensor
            - positions: [total_ctx] tensor
            - num_kv_heads: int
            - head_dim: int
            - rotary_dim: int
            - eps: float (optional, default 1e-6)

    Returns:
        Tuple of (k_out, v_out) tensors, each [total_ctx, num_kv_heads, head_dim]
    """
    kv = inputs["kv"]
    k_norm_weight = inputs["k_norm_weight"]
    cos_sin_cache = inputs["cos_sin_cache"]
    positions = inputs["positions"]
    num_kv_heads = inputs["num_kv_heads"]
    head_dim = inputs["head_dim"]
    rotary_dim = inputs["rotary_dim"]
    eps = inputs.get("eps", 1e-6)

    total_ctx = kv.shape[0]
    kv_size = num_kv_heads * head_dim

    if total_ctx == 0:
        k_out = torch.empty((0, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device)
        v_out = torch.empty_like(k_out)
        return k_out, v_out

    # Split KV into K and V
    # kv shape: [total_ctx, kv_size * 2]
    # K: first kv_size elements, V: second kv_size elements
    kv_reshaped = kv.view(total_ctx, 2, num_kv_heads, head_dim)
    k_raw = kv_reshaped[:, 0, :, :]  # [total_ctx, num_kv_heads, head_dim]
    v_raw = kv_reshaped[:, 1, :, :]  # [total_ctx, num_kv_heads, head_dim]

    # Reshape for per-head processing
    k_flat = k_raw.view(-1, head_dim)  # [total_ctx * num_kv_heads, head_dim]

    # Apply RMSNorm
    k_normed = rms_norm(k_flat.float(), k_norm_weight.float(), eps)
    k_normed = k_normed.to(kv.dtype)

    # Reshape back
    k_normed = k_normed.view(total_ctx, num_kv_heads, head_dim)

    # Apply RoPE
    # Get cos/sin for each position
    positions_1d = positions.view(-1)

    # cos_sin_cache: [max_pos, rotary_dim]
    # Extract cos and sin for positions
    cos_sin_for_pos = cos_sin_cache[positions_1d]  # [total_ctx, rotary_dim]

    half_rotary_dim = rotary_dim // 2
    cos = cos_sin_for_pos[:, :half_rotary_dim].unsqueeze(1)  # [total_ctx, 1, half_rotary_dim]
    sin = cos_sin_for_pos[:, half_rotary_dim:].unsqueeze(1)  # [total_ctx, 1, half_rotary_dim]

    # Apply rotation to the rotary portion
    k_rotary = k_normed[:, :, :rotary_dim]  # [total_ctx, num_kv_heads, rotary_dim]
    k_rotary = k_rotary.float()

    k1 = k_rotary[:, :, :half_rotary_dim]
    k2 = k_rotary[:, :, half_rotary_dim:]

    k_rotated = torch.cat([
        k1 * cos - k2 * sin,
        k2 * cos + k1 * sin,
    ], dim=-1)

    k_rotated = k_rotated.to(kv.dtype)

    # Combine rotated and pass-through portions
    if rotary_dim < head_dim:
        k_pass = k_normed[:, :, rotary_dim:]
        k_out = torch.cat([k_rotated, k_pass], dim=-1)
    else:
        k_out = k_rotated

    v_out = v_raw

    return k_out, v_out
