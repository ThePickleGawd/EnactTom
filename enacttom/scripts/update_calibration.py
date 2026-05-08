"""Write benchmark results back into task JSONs as calibration metadata.

Usage:
    python -m enacttom.scripts.update_calibration \
        --tasks-dir data/enacttom/tasks \
        --benchmark-output-base ./outputs/enacttom/2026-02-07-benchmark \
        --model gpt-5-mini

"""

import argparse

from enacttom.benchmark_results import (
    parse_benchmark_results,
    update_calibration_from_benchmark,
)


def main():
    parser = argparse.ArgumentParser(
        description="Write benchmark results back into task JSONs as calibration metadata."
    )
    parser.add_argument("--tasks-dir", "--task-dir", dest="tasks_dir", required=True, help="Directory containing source task JSONs")
    parser.add_argument("--benchmark-output-base", required=True, help="Benchmark output base path (before -Nagents suffix)")
    parser.add_argument("--model", required=True, help="Model short name used in benchmark")
    args = parser.parse_args()

    results = parse_benchmark_results(args.benchmark_output_base, args.model)
    print(
        f"[calibration] Parsed {results.total} results "
        f"({results.passed} passed, {results.failed} failed)"
    )
    update_calibration_from_benchmark(results, args.tasks_dir)


if __name__ == "__main__":
    main()
