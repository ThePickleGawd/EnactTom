#!/usr/bin/env python3
"""Print a compact summary for a single benchmark run."""

from __future__ import annotations

import argparse
from pathlib import Path

from enacttom.api_costs import (
    BOLD,
    CYAN,
    GREEN,
    RESET,
    WHITE,
    YELLOW,
    format_cost_table,
    merge_summaries,
    summarize_path_costs,
)

from enacttom.benchmark_metrics import load_repeat_summary, repeat_summary_path
from enacttom.benchmark_results import (
    parse_benchmark_results,
    parse_parallel_benchmark_results,
)

RED = "\033[31m"


def _style(text: str, *codes: str) -> str:
    return "".join(code for code in codes if code) + text + RESET


def collect_benchmark_cost_summary(
    output_dir: str,
    *,
    parallel: bool = False,
    repeat: bool = False,
) -> dict:
    output_path = Path(output_dir)
    if repeat or parallel:
        return summarize_path_costs(output_path)

    roots = []
    for summary_file in sorted(
        output_path.parent.glob(f"{output_path.name}-*agents/results/benchmark_summary.json")
    ):
        roots.append(summary_file.parent)

    exact = output_path / "results" / "benchmark_summary.json"
    if exact.exists():
        roots.append(exact.parent)

    merged = {}
    for root in roots:
        merged = merge_summaries(merged, summarize_path_costs(root)) if merged else summarize_path_costs(root)
    return merged or {}


def _print_cost_summary(cost_summary: dict) -> None:
    print("")
    for line in format_cost_table(cost_summary, heading="BENCHMARK API COSTS"):
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact summary for one benchmark output.")
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    parser.add_argument("--model", required=True, help="Benchmark model label.")
    parser.add_argument("--parallel", action="store_true", default=False, help="Parse as a parallel benchmark output.")
    parser.add_argument("--repeat", action="store_true", default=False, help="Parse as a repeated benchmark output.")
    args = parser.parse_args()

    if args.repeat or repeat_summary_path(args.output_dir).exists():
        summary = load_repeat_summary(args.output_dir)
        cost_summary = collect_benchmark_cost_summary(
            args.output_dir,
            parallel=args.parallel,
            repeat=True,
        )
        print("")
        print(_style("=" * 46, BOLD, CYAN))
        print(_style("ENACTTOM REPEATED BENCHMARK SUMMARY", BOLD, WHITE))
        print(_style("=" * 46, BOLD, CYAN))
        print(f"Model: {_style(summary.model, BOLD, CYAN)}")
        print(f"Runs: {_style(str(summary.num_times), BOLD, WHITE)}")
        print(f"Completed runs: {_style(str(summary.completed_runs), BOLD, WHITE)}")
        print(
            f"Average pass rate: {_style(f'{summary.average_pass_rate:.1f}%', BOLD, GREEN)}"
            if summary.average_pass_rate is not None
            else f"Average pass rate: {_style('--', BOLD, YELLOW)}"
        )
        print(f"Pass-rate std dev: {_style(f'{summary.std_pass_rate:.1f}%', BOLD, WHITE)}")
        print(
            f"Pass@{summary.k}: {_style(f'{summary.pass_at_k:.1f}%', BOLD, GREEN)}"
            if summary.pass_at_k is not None
            else f"Pass@{summary.k}: {_style('--', BOLD, YELLOW)}"
        )
        print(
            f"Pass^{summary.k}: {_style(f'{summary.pass_power_k:.1f}%', BOLD, GREEN)}"
            if summary.pass_power_k is not None
            else f"Pass^{summary.k}: {_style('--', BOLD, YELLOW)}"
        )
        _print_cost_summary(cost_summary)
        return 0
    elif args.parallel:
        results = parse_parallel_benchmark_results(args.output_dir, args.model)
    else:
        results = parse_benchmark_results(args.output_dir, args.model)
    cost_summary = collect_benchmark_cost_summary(
        args.output_dir,
        parallel=args.parallel,
        repeat=False,
    )

    print("")
    print(_style("=" * 46, BOLD, CYAN))
    print(_style("ENACTTOM BENCHMARK SUMMARY", BOLD, WHITE))
    print(_style("=" * 46, BOLD, CYAN))
    print(f"Model: {_style(results.model, BOLD, CYAN)}")
    print(f"Tasks: {_style(str(results.total), BOLD, WHITE)}")
    print(f"Passed: {_style(str(results.passed), BOLD, GREEN)}")
    print(f"Failed: {_style(str(results.failed), BOLD, RED)}")
    rate_color = GREEN if results.pass_rate >= 50.0 else YELLOW if results.pass_rate >= 25.0 else RED
    print(f"Pass rate: {_style(f'{results.pass_rate:.1f}%', BOLD, rate_color)}")
    _print_cost_summary(cost_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
