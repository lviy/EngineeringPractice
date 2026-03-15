from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.common import benchmark_cuda_callable, max_abs_diff, maybe_plot, require_cuda, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CUDA and Triton operator implementations.")
    parser.add_argument("--operator", required=True, help="Operator name, e.g. gemm or fused_moe.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before timing.")
    parser.add_argument("--repeat", type=int, default=50, help="Benchmark repetitions.")
    parser.add_argument("--profile", default="default", help="Benchmark case profile.")
    parser.add_argument("--plot", action="store_true", help="Save a comparison plot into output/.")
    return parser.parse_args()


def load_operator_module(operator_name: str):
    return importlib.import_module(f"operators.{operator_name}")


def main() -> None:
    args = parse_args()
    require_cuda()

    operator_module = load_operator_module(args.operator)
    cases = operator_module.build_cases(profile=args.profile)
    backends = operator_module.get_backends()

    if not cases:
        raise RuntimeError(f"No benchmark cases were generated for operator '{args.operator}'.")

    rows: list[dict[str, object]] = []

    for case in cases:
        print(f"[case] {case.name} | {case.summary}")
        prepared_inputs = operator_module.prepare_inputs(case, device="cuda")
        reference_output = None

        for backend in backends:
            available, reason = backend.is_available()
            if not available:
                print(f"  - skip {backend.BACKEND_NAME}: {reason}")
                continue

            def runner():
                return backend.run(prepared_inputs)

            output = runner()
            avg_ms = benchmark_cuda_callable(runner, warmup=args.warmup, repeat=args.repeat)

            if reference_output is None:
                diff = 0.0
                reference_output = output
            else:
                diff = max_abs_diff(reference_output, output)

            row = {
                "operator": args.operator,
                "case_name": case.name,
                "summary": case.summary,
                "backend": backend.BACKEND_NAME,
                "avg_ms": round(avg_ms, 4),
                "max_abs_diff": round(diff, 8),
            }
            rows.append(row)
            print(f"  - {backend.BACKEND_NAME}: {avg_ms:.4f} ms | max_abs_diff={diff:.8f}")

    output_dir = PROJECT_ROOT / "output"
    csv_path = output_dir / f"{args.operator}_benchmark.csv"
    write_csv(csv_path, rows)
    print(f"[saved] {csv_path}")

    if args.plot:
        plot_path = output_dir / f"{args.operator}_benchmark.png"
        maybe_plot(plot_path, rows, args.operator)
        print(f"[saved] {plot_path}")


if __name__ == "__main__":
    main()
