from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run this project on a CUDA-enabled machine.")


def benchmark_cuda_callable(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0

    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)

    return total_ms / repeat


def max_abs_diff(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference - candidate).abs().max().item())


def write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(output_path: Path, rows: list[dict[str, Any]], operator_name: str) -> None:
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Please install requirements.txt first.") from exc

    case_names = [row["case_name"] for row in rows if row["backend"] == rows[0]["backend"]]
    backends = sorted({row["backend"] for row in rows})

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = list(range(len(case_names)))
    width = 0.8 / max(len(backends), 1)

    for index, backend in enumerate(backends):
        backend_rows = [row for row in rows if row["backend"] == backend]
        backend_times = [row["avg_ms"] for row in backend_rows]
        bar_positions = [x + index * width for x in x_positions]
        ax.bar(bar_positions, backend_times, width=width, label=backend)

    center_offset = width * (len(backends) - 1) / 2
    ax.set_xticks([x + center_offset for x in x_positions])
    ax.set_xticklabels(case_names, rotation=20, ha="right")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title(f"{operator_name.upper()} Backend Benchmark")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
