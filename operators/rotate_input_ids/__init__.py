"""Rotate Input IDs Operator.

This operator implements sequence rotation for input_ids, which is used in
speculative decoding and multi-layer Eagle models for efficient sequence
manipulation.

The operation:
1. Shifts all elements in each sequence left by 1 position
2. Inserts a new token at the last position

This enables efficient token prediction updates without full sequence reprocessing.
"""

from __future__ import annotations

import torch

from operators.base import OperatorCase
from operators.rotate_input_ids import cuda_impl, torch_impl, triton_impl


def build_cases(profile: str = "default") -> list[OperatorCase]:
    """Build benchmark test cases for rotate_input_ids.

    Args:
        profile: Benchmark profile name
            - "default": Standard test cases
            - "regular": Uniform sizes
            - "irregular": Variable lengths with padding

    Returns:
        List of OperatorCase objects
    """
    cases: list[OperatorCase] = []

    if profile == "default":
        # Standard cases covering typical scenarios
        configs = [
            # (batch_size, total_tokens, max_seq_len, use_select_index, name)
            (4, 512, 128, False, "small_batch"),
            (16, 1024, 64, False, "medium_batch"),
            (32, 2048, 64, False, "large_batch"),
            (8, 256, 32, True, "with_select_index"),
        ]
        for batch_size, total_tokens, max_seq_len, use_select, name in configs:
            cases.append(OperatorCase(
                name=f"rotate_input_ids_{name}",
                summary=f"batch={batch_size}, total_tokens={total_tokens}, max_seq_len={max_seq_len}, select_index={use_select}",
                params={
                    "batch_size": batch_size,
                    "total_tokens": total_tokens,
                    "max_seq_len": max_seq_len,
                    "use_select_index": use_select,
                    "dtype": torch.int64,
                    "family": "default",
                    "sweep_value": batch_size,
                    "x_label": "Batch Size",
                },
            ))

    elif profile == "regular":
        # Regular test cases - uniform sizes
        # Sweep over batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for bs in batch_sizes:
            cases.append(OperatorCase(
                name=f"rotate_input_ids_regular_batch{bs}",
                summary=f"batch={bs}, seq_len=64, total_tokens={bs*64}",
                params={
                    "batch_size": bs,
                    "total_tokens": bs * 64,
                    "max_seq_len": 64,
                    "use_select_index": False,
                    "dtype": torch.int64,
                    "family": "batch_sweep",
                    "sweep_value": bs,
                    "x_label": "Batch Size",
                },
            ))

        # Sweep over sequence lengths
        seq_lens = [8, 16, 32, 64, 128, 256, 512, 1024]
        for sl in seq_lens:
            cases.append(OperatorCase(
                name=f"rotate_input_ids_regular_seq{sl}",
                summary=f"batch=8, seq_len={sl}, total_tokens={8*sl}",
                params={
                    "batch_size": 8,
                    "total_tokens": 8 * sl,
                    "max_seq_len": sl,
                    "use_select_index": False,
                    "dtype": torch.int64,
                    "family": "seq_len_sweep",
                    "sweep_value": sl,
                    "x_label": "Sequence Length",
                },
            ))

    elif profile == "irregular":
        # Irregular test cases - variable lengths with padding patterns

        # Case 1: Highly variable sequence lengths (like real-world batching)
        cases.append(OperatorCase(
            name="rotate_input_ids_irregular_variable",
            summary="Variable sequence lengths (1 to max)",
            params={
                "batch_size": 32,
                "total_tokens": 2048,
                "max_seq_len": 128,
                "use_select_index": False,
                "dtype": torch.int64,
                "family": "irregular",
                "sweep_value": 1,
                "x_label": "Irregularity Level",
                "irregular_mode": "variable_lengths",
            },
        ))

        # Case 2: Power-of-2 sequence lengths (common in padded batches)
        cases.append(OperatorCase(
            name="rotate_input_ids_irregular_power2",
            summary="Power-of-2 sequence lengths",
            params={
                "batch_size": 16,
                "total_tokens": 1024,
                "max_seq_len": 64,
                "use_select_index": False,
                "dtype": torch.int64,
                "family": "irregular",
                "sweep_value": 2,
                "x_label": "Irregularity Level",
                "irregular_mode": "power2_lengths",
            },
        ))

        # Case 3: With padding simulation (some sequences are padded)
        for padding_ratio in [0.1, 0.3, 0.5, 0.7]:
            cases.append(OperatorCase(
                name=f"rotate_input_ids_irregular_pad_{int(padding_ratio*100)}",
                summary=f"Sequences with {int(padding_ratio*100)}% padding overhead",
                params={
                    "batch_size": 16,
                    "total_tokens": int(1024 * (1 + padding_ratio)),
                    "max_seq_len": 64,
                    "use_select_index": False,
                    "dtype": torch.int64,
                    "family": "irregular_padding",
                    "sweep_value": padding_ratio,
                    "x_label": "Padding Ratio",
                    "irregular_mode": "padding_simulation",
                    "padding_ratio": padding_ratio,
                },
            ))

        # Case 4: Decode scenario - many tiny sequences
        cases.append(OperatorCase(
            name="rotate_input_ids_irregular_decode",
            summary="Decode scenario (many tiny sequences)",
            params={
                "batch_size": 64,
                "total_tokens": 128,  # Each sequence is just 1-2 tokens
                "max_seq_len": 4,
                "use_select_index": True,
                "dtype": torch.int64,
                "family": "irregular",
                "sweep_value": 3,
                "x_label": "Irregularity Level",
                "irregular_mode": "decode_tiny",
            },
        ))

    else:
        raise ValueError(f"Unknown profile for rotate_input_ids: {profile}")

    return cases


def prepare_inputs(case: OperatorCase, device: str = "cuda") -> dict:
    """Prepare input tensors for rotate_input_ids.

    Args:
        case: OperatorCase with benchmark parameters
        device: Device to place tensors on

    Returns:
        Dictionary of input tensors
    """
    torch.manual_seed(42)

    batch_size = case.params["batch_size"]
    total_tokens = case.params["total_tokens"]
    max_seq_len = case.params["max_seq_len"]
    use_select_index = case.params["use_select_index"]
    dtype = case.params["dtype"]

    irregular_mode = case.params.get("irregular_mode", None)

    # Create input_ids tensor with random token IDs
    input_ids = torch.randint(0, 32000, (total_tokens,), device=device, dtype=dtype)

    # Create sequence metadata
    extend_start_loc = torch.zeros(batch_size, device=device, dtype=torch.int64)
    extend_seq_lens = torch.zeros(batch_size, device=device, dtype=torch.int64)

    if irregular_mode == "variable_lengths":
        # Variable lengths from 1 to max_seq_len
        lengths = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
        # Adjust to fit within total_tokens
        cumsum = lengths.cumsum(0)
        if cumsum[-1] > total_tokens:
            scale = total_tokens / cumsum[-1].item()
            lengths = (lengths.float() * scale).int().clamp(min=1)
            cumsum = lengths.cumsum(0)
        extend_seq_lens = lengths.to(torch.int64)
        extend_start_loc[0] = 0
        extend_start_loc[1:] = cumsum[:-1]

    elif irregular_mode == "power2_lengths":
        # Power-of-2 sequence lengths
        powers = torch.randint(0, 4, (batch_size,), device=device)  # 0-3 for lengths 1,2,4,8
        lengths = (2 ** powers).clamp(min=1, max=max_seq_len)
        # Adjust to fit
        cumsum = lengths.cumsum(0)
        if cumsum[-1] > total_tokens:
            lengths = torch.ones(batch_size, device=device, dtype=torch.int64)
        extend_seq_lens = lengths.to(torch.int64)
        extend_start_loc[0] = 0
        extend_start_loc[1:] = lengths.cumsum(0)[:-1]

    elif irregular_mode == "padding_simulation":
        # Simulate padding by having some sequences shorter than allocated space
        padding_ratio = case.params.get("padding_ratio", 0.3)
        base_length = max_seq_len // 2
        lengths = torch.full((batch_size,), base_length, device=device, dtype=torch.int64)
        # Some sequences are shorter (padding)
        num_padded = int(batch_size * padding_ratio)
        lengths[:num_padded] = torch.randint(1, base_length, (num_padded,), device=device)
        extend_seq_lens = lengths
        extend_start_loc[0] = 0
        extend_start_loc[1:] = lengths.cumsum(0)[:-1]

    elif irregular_mode == "decode_tiny":
        # Very small sequences (1-3 tokens each)
        lengths = torch.randint(1, 4, (batch_size,), device=device)
        extend_seq_lens = lengths.to(torch.int64)
        extend_start_loc[0] = 0
        extend_start_loc[1:] = lengths.cumsum(0)[:-1]

    else:
        # Regular: evenly distributed sequences
        seq_len_per_batch = total_tokens // batch_size
        lengths = torch.full((batch_size,), seq_len_per_batch, device=device, dtype=torch.int64)
        # Make sure last batch gets any remaining tokens
        remainder = total_tokens - seq_len_per_batch * batch_size
        if remainder > 0:
            lengths[-1] += remainder
        extend_seq_lens = lengths
        extend_start_loc[0] = 0
        extend_start_loc[1:] = lengths.cumsum(0)[:-1]

    # New tokens to insert
    topk_index = torch.randint(0, 32000, (batch_size,), device=device, dtype=dtype)

    # Optional select_index
    select_index = None
    if use_select_index:
        # Select custom positions for new tokens
        select_index = torch.zeros(batch_size, device=device, dtype=torch.int64)
        for i in range(batch_size):
            start = extend_start_loc[i].item()
            length = extend_seq_lens[i].item()
            select_index[i] = start + length - 1

    return {
        "input_ids": input_ids,
        "extend_start_loc": extend_start_loc,
        "extend_seq_lens": extend_seq_lens,
        "topk_index": topk_index,
        "select_index": select_index,
        "input_dtype_name": "int64",
        "batch_size": batch_size,
        "total_tokens": total_tokens,
    }


def get_backends():
    """Get available backend implementations.

    Returns:
        List of backend modules
    """
    backends = [torch_impl, triton_impl]
    try:
        import importlib.util
        if importlib.util.find_spec("torch.utils.cpp_extension"):
            backends.insert(0, cuda_impl)
    except ImportError:
        pass
    return backends