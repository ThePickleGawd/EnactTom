"""CLI entry point for parallel benchmark execution.

Called by run_enacttom.sh when --max-workers is specified.
Runs one benchmark subprocess per task JSON with GPU round-robin.
Calibration is skipped by default; pass --calibration to write per-task updates.

Usage:
    python -m enacttom.scripts.run_benchmark_parallel \
        --tasks-dir data/enacttom/tasks \
        --model gpt-5.2 \
        --output-dir ./outputs/enacttom/2026-02-25-benchmark \
        --max-workers 8

"""

import argparse

from enacttom.benchmark_results import run_benchmark_parallel


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark in parallel (one process per task)."
    )
    parser.add_argument("--tasks-dir", "--task-dir", dest="tasks_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument("--no-video", action="store_true", default=True)
    parser.add_argument("--category", default=None)
    parser.set_defaults(no_calibration=True)
    parser.add_argument(
        "--no-calibration",
        dest="no_calibration",
        action="store_true",
        help="Do not write calibration back into source task JSONs (default).",
    )
    parser.add_argument(
        "--calibration",
        dest="no_calibration",
        action="store_false",
        help="Write calibration back into source task JSONs.",
    )
    parser.add_argument(
        "--benchmark-run-mode",
        "--run-mode",
        dest="benchmark_run_mode",
        default="standard",
        choices=["standard", "baseline"],
    )
    parser.add_argument("--observation-mode", default="text", choices=["text", "vision"])
    parser.add_argument("--selector-min-frames", type=int, default=1)
    parser.add_argument("--selector-max-frames", type=int, default=5)
    parser.add_argument("--selector-max-candidates", type=int, default=12)

    args = parser.parse_args()

    results = run_benchmark_parallel(
        tasks_dir=args.tasks_dir,
        model=args.model,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        workers_per_gpu=args.workers_per_gpu,
        no_video=args.no_video,
        category=args.category,
        run_mode=args.benchmark_run_mode,
        observation_mode=args.observation_mode,
        selector_min_frames=args.selector_min_frames,
        selector_max_frames=args.selector_max_frames,
        selector_max_candidates=args.selector_max_candidates,
        write_calibration=not args.no_calibration,
        log_prefix="[parallel]",
    )

    print(
        f"[parallel] Done: {results.total} tasks "
        f"({results.passed} passed, {results.failed} failed, "
        f"{results.pass_rate:.1f}%)"
    )


if __name__ == "__main__":
    main()
