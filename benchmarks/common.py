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


def compute_tflops(m: int, n: int, k: int, avg_ms: float) -> float:
    if avg_ms <= 0:
        return 0.0
    flops = 2.0 * m * n * k
    return flops / (avg_ms * 1e-3) / 1e12


def attach_speedup(rows: list[dict[str, Any]], baseline_backend: str = "cuda") -> list[dict[str, Any]]:
    baseline_map: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["case_name"]), str(row["dtype"]))
        if row["backend"] == baseline_backend:
            baseline_map[key] = float(row["avg_ms"])

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["case_name"]), str(row["dtype"]))
        row_copy = dict(row)
        baseline_ms = baseline_map.get(key)
        current_ms = float(row["avg_ms"])
        if baseline_ms is None or current_ms <= 0:
            row_copy["speedup_vs_cuda"] = ""
        else:
            row_copy["speedup_vs_cuda"] = round(baseline_ms / current_ms, 4)
        enriched_rows.append(row_copy)

    return enriched_rows


def write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(output_path: Path, rows: list[dict[str, Any]], operator_name: str, metric_key: str, ylabel: str, title: str) -> None:
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Please install requirements.txt first.") from exc

    families = list(dict.fromkeys(row.get("family", "default") for row in rows))
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 4.5 * len(families)), squeeze=False)

    for axis, family in zip(axes.flatten(), families):
        family_rows = [row for row in rows if row.get("family", "default") == family]
        series_keys = sorted({(row["backend"], row["dtype"]) for row in family_rows})
        x_label = next((str(row.get("x_label")) for row in family_rows if row.get("x_label")), "Sweep Value")

        for backend, dtype_name in series_keys:
            series_rows = [
                row
                for row in family_rows
                if row["backend"] == backend and row["dtype"] == dtype_name and row.get(metric_key, "") != ""
            ]
            series_rows.sort(key=lambda row: row.get("sweep_value", 0))
            x_values = [row.get("sweep_value", 0) for row in series_rows]
            y_values = [row[metric_key] for row in series_rows]
            if not x_values:
                continue
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=f"{backend}-{dtype_name}")

        axis.set_xlabel(x_label)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{operator_name.upper()} {title} | {family}")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
