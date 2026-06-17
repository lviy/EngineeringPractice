"""Rotate Input IDs Operator - PyTorch Implementation.

This module provides a pure PyTorch reference implementation of the rotate_input_ids
kernel. It shifts elements in each sequence and inserts new tokens at specific positions.

This is commonly used in speculative decoding for efficient sequence manipulation.
"""

from __future__ import annotations

import torch

BACKEND_NAME = "torch"


def is_available() -> tuple[bool, str]:
    """Check if PyTorch backend is available."""
    return True, ""


def run(inputs: dict) -> torch.Tensor:
    """Run rotate_input_ids using PyTorch operations.

    For each sequence in the batch:
    1. Shift all elements left by 1 position (element at pos+1 moves to pos)
    2. Insert the new token at the last position

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
    input_ids = inputs["input_ids"].clone()
    extend_start_loc = inputs["extend_start_loc"]
    extend_seq_lens = inputs["extend_seq_lens"]
    topk_index = inputs["topk_index"]
    select_index = inputs.get("select_index", None)

    batch_size = extend_seq_lens.shape[0]

    for pid in range(batch_size):
        start_loc = int(extend_start_loc[pid].item())
        seq_len = int(extend_seq_lens[pid].item())
        new_token = topk_index[pid]

        if seq_len <= 0:
            continue

        num_elements_to_shift = seq_len - 1

        # Shift elements: each element moves to position - 1
        if num_elements_to_shift > 0:
            input_ids[start_loc:start_loc + num_elements_to_shift] = \
                input_ids[start_loc + 1:start_loc + seq_len].clone()

        # Write new token at the last position
        if select_index is not None:
            last_pos = int(select_index[pid].item())
        else:
            last_pos = start_loc + seq_len - 1
        input_ids[last_pos] = new_token

    return input_ids