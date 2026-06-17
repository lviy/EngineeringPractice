"""Rotate Input IDs Operator - Triton Implementation.

Port of SGLang's rotate_input_ids Triton kernel.
Shifts input IDs sequences and inserts new tokens at specific positions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BACKEND_NAME = "triton"


@triton.jit
def rotate_input_ids_kernel(
    input_ids_ptr,
    extend_start_loc_ptr,
    extend_seq_lens_ptr,
    topk_index_ptr,
    select_index_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Rotate input IDs kernel.

    For each sequence:
    1. Shift all elements left by 1 position
    2. Insert new token at the last position
    """
    pid = tl.program_id(0)

    start_loc = tl.load(extend_start_loc_ptr + pid)
    seq_len = tl.load(extend_seq_lens_ptr + pid)
    new_token = tl.load(topk_index_ptr + pid)

    num_elements_to_shift = seq_len - 1

    # Process shifts in blocks
    for off in range(0, num_elements_to_shift, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements_to_shift

        read_ptr = input_ids_ptr + start_loc + offsets + 1
        val = tl.load(read_ptr, mask=mask)
        tl.debug_barrier()

        write_ptr = input_ids_ptr + start_loc + offsets
        tl.store(write_ptr, val, mask=mask)
        tl.debug_barrier()

    # Write new token at last position
    if seq_len > 0:
        if select_index_ptr is not None:
            last_pos_ptr = input_ids_ptr + tl.load(select_index_ptr + pid)
        else:
            last_pos_ptr = input_ids_ptr + start_loc + seq_len - 1
        tl.store(last_pos_ptr, new_token)


def rotate_input_ids_triton(
    input_ids: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    topk_index: torch.Tensor,
    select_index: torch.Tensor = None
) -> torch.Tensor:
    """Run rotate_input_ids using Triton kernel.

    Args:
        input_ids: Flattened input IDs tensor
        extend_start_loc: Start locations for each sequence
        extend_seq_lens: Sequence lengths
        topk_index: New tokens to insert
        select_index: Optional custom positions for new tokens

    Returns:
        Modified input_ids tensor
    """
    batch_size = extend_seq_lens.shape[0]
    BLOCK_SIZE = 4096 if select_index is not None else 8
    grid = (batch_size,)

    # Ensure proper types
    extend_start_loc = extend_start_loc.to(torch.int64)
    extend_seq_lens = extend_seq_lens.to(torch.int64)

    rotate_input_ids_kernel[grid](
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return input_ids


def is_available() -> tuple[bool, str]:
    """Check if Triton backend is available."""
    try:
        import triton  # noqa: F401
        return True, ""
    except ImportError:
        return False, "Triton is not installed"


def run(inputs: dict) -> torch.Tensor:
    """Run rotate_input_ids using Triton kernel.

    Args:
        inputs: Dictionary containing:
            - input_ids: Flattened input IDs tensor
            - extend_start_loc: Start locations for each sequence
            - extend_seq_lens: Sequence lengths
            - topk_index: New tokens to insert
            - select_index: Optional custom positions for new tokens

    Returns:
        Modified input_ids tensor
    """
    input_ids = inputs["input_ids"]
    extend_start_loc = inputs["extend_start_loc"]
    extend_seq_lens = inputs["extend_seq_lens"]
    topk_index = inputs["topk_index"]
    select_index = inputs.get("select_index", None)

    return rotate_input_ids_triton(
        input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index
    )
