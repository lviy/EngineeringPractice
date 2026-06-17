#!/usr/bin/env python3
"""Benchmark runner for all operators.

This script runs comprehensive benchmarks for all operators across
multiple backends (CUDA, PyTorch, Triton) with both regular and
irregular test cases.

Usage:
    python benchmarks/benchmark_all.py [--warmup N] [--repeat N] [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from benchmarks.common import (
    benchmark_cuda_callable,
    max_abs_diff,
    write_csv,
    plot_metric,
    require_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA, PyTorch, and Triton operator implementations."
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["fused_kv_materialize", "rotate_input_ids"],
        help="Operator names to benchmark",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["regular", "irregular"],
        help="Benchmark profiles to run",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=50,
        help="Benchmark repetitions",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save comparison plots into output/",
    )
    return parser.parse_args()


def load_operator_module(operator_name: str):
    """Dynamically load an operator module."""
    import importlib
    return importlib.import_module(f"operators.{operator_name}")


def run_benchmark(
    operator_name: str,
    profile: str,
    warmup: int,
    repeat: int,
) -> list[dict[str, Any]]:
    """Run benchmark for a single operator and profile.

    Args:
        operator_name: Name of the operator module
        profile: Benchmark profile name
        warmup: Number of warmup iterations
        repeat: Number of benchmark repetitions

    Returns:
        List of benchmark result dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Operator: {operator_name} | Profile: {profile}")
    print(f"{'='*60}")

    operator_module = load_operator_module(operator_name)

    try:
        cases = operator_module.build_cases(profile=profile)
    except ValueError as e:
        print(f"  [skip] Unknown profile '{profile}': {e}")
        return []

    if not cases:
        print(f"  [skip] No cases generated for profile '{profile}'")
        return []

    backends = operator_module.get_backends()
    rows: list[dict[str, Any]] = []

    for case in cases:
        print(f"\n[case] {case.name} | {case.summary}")

        try:
            prepared_inputs = operator_module.prepare_inputs(case, device="cuda")
        except Exception as exc:
            print(f"  - skip case: {exc}")
            continue

        reference_output = None

        for backend in backends:
            try:
                available, reason = backend.is_available(prepared_inputs)
            except TypeError:
                available, reason = backend.is_available()

            if not available:
                print(f"  - skip {backend.BACKEND_NAME}: {reason}")
                continue

            def runner():
                return backend.run(prepared_inputs)

            try:
                output = runner()
                avg_ms = benchmark_cuda_callable(runner, warmup=warmup, repeat=repeat)
            except Exception as exc:
                print(f"  - error {backend.BACKEND_NAME}: {exc}")
                continue

            # Compute correctness difference
            if reference_output is None:
                diff = 0.0
                reference_output = output
            else:
                if isinstance(output, tuple):
                    # Handle tuple outputs (like k_out, v_out)
                    if isinstance(reference_output, tuple):
                        diffs = [
                            max_abs_diff(ref, out)
                            for ref, out in zip(reference_output, output)
                        ]
                        diff = max(diffs)
                    else:
                        diff = max_abs_diff(reference_output, output[0])
                else:
                    diff = max_abs_diff(reference_output, output)

            # Build result row
            row = {
                "operator": operator_name,
                "profile": profile,
                "case_name": case.name,
                "summary": case.summary,
                "dtype": prepared_inputs.get("input_dtype_name", "unknown"),
                "family": case.params.get("family", "default"),
                "sweep_value": case.params.get("sweep_value", -1),
                "x_label": case.params.get("x_label", "Value"),
                "backend": backend.BACKEND_NAME,
                "avg_ms": round(avg_ms, 4),
                "max_abs_diff": round(diff, 8),
            }

            # Add operator-specific metrics
            if operator_name == "fused_kv_materialize":
                row.update({
                    "total_ctx": case.params.get("total_ctx", -1),
                    "num_kv_heads": case.params.get("num_kv_heads", -1),
                    "head_dim": case.params.get("head_dim", -1),
                    "rotary_dim": case.params.get("rotary_dim", -1),
                })
            elif operator_name == "rotate_input_ids":
                row.update({
                    "batch_size": case.params.get("batch_size", -1),
                    "total_tokens": case.params.get("total_tokens", -1),
                    "max_seq_len": case.params.get("max_seq_len", -1),
                })

            rows.append(row)
            print(f"  - {backend.BACKEND_NAME}: {avg_ms:.4f} ms | max_diff={diff:.8f}")

    return rows


def attach_speedup(rows: list[dict[str, Any]], baseline_backend: str = "torch") -> list[dict[str, Any]]:
    """Attach speedup metrics relative to baseline backend.

    Args:
        rows: List of benchmark result dictionaries
        baseline_backend: Backend to use as baseline for speedup calculation

    Returns:
        Enriched rows with speedup metrics
    """
    baseline_map: dict[tuple[str, str, str], float] = {}

    for row in rows:
        key = (str(row["operator"]), str(row["case_name"]), str(row["dtype"]))
        if row["backend"] == baseline_backend:
            baseline_map[key] = float(row["avg_ms"])

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["operator"]), str(row["case_name"]), str(row["dtype"]))
        row_copy = dict(row)
        baseline_ms = baseline_map.get(key)
        current_ms = float(row["avg_ms"])

        if baseline_ms is None or current_ms <= 0:
            row_copy["speedup_vs_torch"] = ""
        else:
            row_copy["speedup_vs_torch"] = round(baseline_ms / current_ms, 4)

        enriched_rows.append(row_copy)

    return enriched_rows


def main() -> None:
    args = parse_args()
    require_cuda()

    all_rows: list[dict[str, Any]] = []

    for operator_name in args.operators:
        for profile in args.profiles:
            rows = run_benchmark(
                operator_name=operator_name,
                profile=profile,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            all_rows.extend(rows)

    if not all_rows:
        print("\n[error] No benchmark results collected!")
        return

    # Add speedup metrics
    all_rows = attach_speedup(all_rows, baseline_backend="torch")

    # Save results
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined results
    combined_csv_path = output_dir / "combined_benchmark.csv"
    write_csv(combined_csv_path, all_rows)
    print(f"\n[saved] {combined_csv_path}")

    # Save per-operator results
    for operator_name in args.operators:
        operator_rows = [r for r in all_rows if r["operator"] == operator_name]
        if operator_rows:
            csv_path = output_dir / f"{operator_name}_benchmark.csv"
            write_csv(csv_path, operator_rows)
            print(f"[saved] {csv_path}")

    # Generate plots if requested
    if args.plot:
        for operator_name in args.operators:
            operator_rows = [r for r in all_rows if r["operator"] == operator_name]
            if not operator_rows:
                continue

            latency_plot_path = output_dir / f"{operator_name}_latency.png"
            speedup_plot_path = output_dir / f"{operator_name}_speedup.png"

            plot_metric(
                latency_plot_path,
                operator_rows,
                operator_name,
                metric_key="avg_ms",
                ylabel="Average Latency (ms)",
                title="Latency Benchmark",
            )
            print(f"[saved] {latency_plot_path}")

            plot_metric(
                speedup_plot_path,
                operator_rows,
                operator_name,
                metric_key="speedup_vs_torch",
                ylabel="Speedup vs PyTorch",
                title="Speedup Benchmark",
            )
            print(f"[saved] {speedup_plot_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    for operator_name in args.operators:
        operator_rows = [r for r in all_rows if r["operator"] == operator_name]
        if not operator_rows:
            continue

        print(f"\n{operator_name.upper()}:")

        # Group by backend
        backends = sorted(set(r["backend"] for r in operator_rows))
        for backend in backends:
            backend_rows = [r for r in operator_rows if r["backend"] == backend]
            avg_ms = sum(float(r["avg_ms"]) for r in backend_rows) / len(backend_rows)
            print(f"  {backend}: avg={avg_ms:.4f}ms across {len(backend_rows)} cases")


if __name__ == "__main__":
    main()
