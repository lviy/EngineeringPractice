"""Fused KV Materialize Operator - Triton Implementation.

Port of SGLang's fused KV materialization Triton kernel.
Combines: KV projection + RMSNorm + RoPE (Triton), then KV writes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BACKEND_NAME = "triton"


@triton.jit
def _fused_norm_rope_kernel(
    kv_ptr,
    k_norm_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    k_out_ptr,
    v_out_ptr,
    kv_stride_ctx,
    cos_sin_stride_pos,
    k_out_stride_ctx,
    k_out_stride_head,
    v_out_stride_ctx,
    v_out_stride_head,
    total_ctx,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """Fused RMSNorm(K) + RoPE(K) materialization. Grid: (total_ctx, num_kv_heads)."""
    ctx_id = tl.program_id(0)
    head_id = tl.program_id(1)
    if ctx_id >= total_ctx:
        return

    # Load metadata
    position = tl.load(positions_ptr + ctx_id)

    # Compute base pointers
    kv_base = kv_ptr + ctx_id * kv_stride_ctx
    k_base = kv_base + head_id * head_dim
    v_base = kv_base + kv_size + head_id * head_dim
    k_write = k_out_ptr + ctx_id * k_out_stride_ctx + head_id * k_out_stride_head
    v_write = v_out_ptr + ctx_id * v_out_stride_ctx + head_id * v_out_stride_head

    # Load K and V
    offs = tl.arange(0, BLOCK_HD)
    mask_hd = offs < head_dim
    mask_half = offs < half_rotary_dim

    k_raw = tl.load(k_base + offs, mask=mask_hd, other=0.0).to(tl.float32)
    v_raw = tl.load(v_base + offs, mask=mask_hd, other=0.0)

    # RMSNorm on K
    inv_rms = tl.rsqrt(tl.sum(k_raw * k_raw) / head_dim + eps)
    norm_w = tl.load(k_norm_weight_ptr + offs, mask=mask_hd, other=1.0).to(tl.float32)
    k_normed = k_raw * inv_rms * norm_w

    # RoPE (neox style)
    cos_sin_base = cos_sin_cache_ptr + position * cos_sin_stride_pos
    cos_v = tl.load(cos_sin_base + offs, mask=mask_half, other=1.0).to(tl.float32)
    sin_v = tl.load(
        cos_sin_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)

    # Extract first/second halves of K for rotation
    k_first = tl.where(mask_half, k_normed, 0.0)
    k_second_raw = tl.load(
        k_base + half_rotary_dim + offs, mask=mask_half, other=0.0
    ).to(tl.float32)
    norm_w_second = tl.load(
        k_norm_weight_ptr + half_rotary_dim + offs, mask=mask_half, other=1.0
    ).to(tl.float32)
    k_second = k_second_raw * inv_rms * norm_w_second

    # Apply rotation
    k_rot_first = k_first * cos_v - k_second * sin_v
    k_rot_second = k_second * cos_v + k_first * sin_v

    # Store V (no transform)
    tl.store(v_write + offs, v_raw, mask=mask_hd)

    # Store K: rotated halves + pass-through
    tl.store(k_write + offs, k_rot_first.to(v_raw.dtype), mask=mask_half)
    tl.store(
        k_write + half_rotary_dim + offs, k_rot_second.to(v_raw.dtype), mask=mask_half
    )
    mask_pass = (offs >= rotary_dim) & (offs < head_dim)
    tl.store(k_write + offs, k_normed.to(v_raw.dtype), mask=mask_pass)


def _fused_norm_rope(
    kv: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + RoPE materialization for a single layer."""
    total_ctx = kv.shape[0]
    if total_ctx == 0:
        empty = torch.empty(
            (0, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
        )
        return empty, empty

    kv_size = num_kv_heads * head_dim
    half_rotary_dim = rotary_dim // 2
    BLOCK_HD = triton.next_power_of_2(head_dim)

    # Ensure int64 for indexing
    if positions.device != kv.device:
        positions = positions.to(device=kv.device, dtype=torch.int64)
    elif positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    k_out = torch.empty(
        (total_ctx, num_kv_heads, head_dim), dtype=kv.dtype, device=kv.device
    )
    v_out = torch.empty_like(k_out)

    _fused_norm_rope_kernel[(total_ctx, num_kv_heads)](
        kv,
        k_norm_weight,
        cos_sin_cache,
        positions,
        k_out,
        v_out,
        kv.stride(0),
        cos_sin_cache.stride(0),
        k_out.stride(0),
        k_out.stride(1),
        v_out.stride(0),
        v_out.stride(1),
        total_ctx,
        num_kv_heads,
        head_dim,
        kv_size,
        rotary_dim,
        half_rotary_dim,
        eps,
        BLOCK_HD,
    )
    return k_out, v_out


def is_available() -> tuple[bool, str]:
    """Check if Triton backend is available."""
    try:
        import triton  # noqa: F401
        return True, ""
    except ImportError:
        return False, "Triton is not installed"


def run(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused KV materialization using Triton kernel.

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

    k_out, v_out = _fused_norm_rope(
        kv, k_norm_weight, cos_sin_cache, positions,
        num_kv_heads, head_dim, rotary_dim, eps
    )
    return k_out, v_out
