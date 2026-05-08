#!/usr/bin/env python3
"""Run the same benchmark command multiple times and aggregate reliability metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from enacttom.api_costs import (
    BOLD,
    CYAN,
    GREEN,
    RESET,
    WHITE,
    YELLOW,
    format_cost_table,
    summarize_path_costs,
)
from enacttom.benchmark_metrics import (
    BenchmarkRepeatRun,
    build_repeat_summary,
    write_repeat_summary,
)
from enacttom.benchmark_results import (
    BenchmarkResults,
    kill_proc_group,
    parse_benchmark_results,
    parse_parallel_benchmark_results,
    update_calibration_from_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_EnactToM = PROJECT_ROOT / "enacttom" / "run_enacttom.sh"


def _style(text: str, *codes: str) -> str:
    return "".join(code for code in codes if code) + text + RESET


@dataclass
class ActiveRun:
    run_index: int
    output_dir: Path
    log_path: Path
    log_handle: object
    proc: subprocess.Popen


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat EnactToM benchmark runs and aggregate pass@k / pass^k metrics."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks-dir", "--task-dir", dest="tasks_dir", help="Task directory to benchmark.")
    source.add_argument("--task", help="Single task JSON to benchmark.")
    parser.add_argument("--model", required=True, help="Model to benchmark.")
    parser.add_argument("--output-dir", required=True, help="Parent output directory for all repeats.")
    parser.add_argument("--num-times", type=int, default=3, help="Number of repeated runs.")
    parser.add_argument("--max-sim-steps", type=int, default=200000)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--observation-mode", default="text", choices=["text", "vision"])
    parser.add_argument(
        "--benchmark-run-mode",
        default="standard",
        choices=["standard", "baseline"],
    )
    parser.add_argument("--selector-min-frames", type=int, default=1)
    parser.add_argument("--selector-max-frames", type=int, default=5)
    parser.add_argument("--selector-max-candidates", type=int, default=12)
    parser.add_argument("--video", action="store_true", default=False)
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
        help="Write calibration back into source task JSONs after repeats finish.",
    )
    return parser.parse_args(argv)


def _load_task_dicts(task_file: Path) -> List[dict]:
    with open(task_file, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return [entry for entry in payload["tasks"] if isinstance(entry, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _matches_category(task_dict: dict, category: Optional[str]) -> bool:
    if not category:
        return True
    return str(task_dict.get("category", "cooperative")).strip().lower() == category


def _task_id_from_dict(task_dict: dict, fallback: str) -> str:
    task_id = str(task_dict.get("task_id", "")).strip()
    if task_id and task_id != "REPLACE_WITH_UNIQUE_ID":
        return task_id
    return fallback


def _collect_expected_task_ids(args: argparse.Namespace) -> List[str]:
    task_files: List[Path]
    if args.task:
        task_files = [Path(args.task)]
    else:
        task_files = sorted(Path(args.tasks_dir).glob("*.json"))

    expected_task_ids: List[str] = []
    seen = set()
    for task_file in task_files:
        try:
            task_dicts = _load_task_dicts(task_file)
        except Exception:
            continue

        for idx, task_dict in enumerate(task_dicts):
            if not _matches_category(task_dict, args.category):
                continue
            fallback = task_file.stem if len(task_dicts) == 1 else f"{task_file.stem}:{idx}"
            task_id = _task_id_from_dict(task_dict, fallback=fallback)
            if task_id in seen:
                continue
            seen.add(task_id)
            expected_task_ids.append(task_id)

    return expected_task_ids


def _build_run_command(args: argparse.Namespace, run_output_dir: Path) -> list[str]:
    cmd = [
        str(RUN_EnactToM),
        "benchmark",
        "--model",
        args.model,
        "--output-dir",
        str(run_output_dir),
        "--max-sim-steps",
        str(args.max_sim_steps),
        "--benchmark-run-mode",
        args.benchmark_run_mode,
        "--observation-mode",
        args.observation_mode,
        "--selector-min-frames",
        str(args.selector_min_frames),
        "--selector-max-frames",
        str(args.selector_max_frames),
        "--selector-max-candidates",
        str(args.selector_max_candidates),
        "--no-calibration",
        "--num-times",
        "1",
    ]

    if args.tasks_dir:
        cmd.extend(["--tasks-dir", args.tasks_dir])
    else:
        cmd.extend(["--task", args.task])

    if args.max_llm_calls is not None:
        cmd.extend(["--max-llm-calls", str(args.max_llm_calls)])
    if args.workers_per_gpu is not None:
        cmd.extend(["--workers-per-gpu", str(args.workers_per_gpu)])
    elif args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.num_gpus is not None:
        cmd.extend(["--num-gpus", str(args.num_gpus)])
    if args.category:
        cmd.extend(["--category", args.category])
    if args.video:
        cmd.append("--video")

    return cmd


def _parse_run_results(args: argparse.Namespace, run_output_dir: Path) -> BenchmarkResults:
    if args.task:
        return parse_benchmark_results(str(run_output_dir), args.model)
    if args.max_workers is not None or args.workers_per_gpu is not None:
        try:
            return parse_parallel_benchmark_results(str(run_output_dir), args.model)
        except Exception:
            return parse_benchmark_results(str(run_output_dir), args.model)
    return parse_benchmark_results(str(run_output_dir), args.model)


def _run_log_path(run_output_dir: Path) -> Path:
    return run_output_dir / "benchmark.log"


def _launch_run(
    args: argparse.Namespace,
    run_index: int,
    run_output_dir: Path,
) -> ActiveRun:
    cmd = _build_run_command(args, run_output_dir)
    log_path = _run_log_path(run_output_dir)
    log_handle = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise

    print(f"[repeat] starting run {run_index}/{args.num_times}: {' '.join(cmd)}")
    print(f"[repeat] run {run_index}/{args.num_times} log: {log_path}")
    return ActiveRun(
        run_index=run_index,
        output_dir=run_output_dir,
        log_path=log_path,
        log_handle=log_handle,
        proc=proc,
    )


def _print_summary(summary_path: Path, summary, output_dir: Path) -> None:
    print("")
    print(_style("=" * 46, BOLD, CYAN))
    print(_style("ENACTTOM REPEATED BENCHMARK SUMMARY", BOLD, WHITE))
    print(_style("=" * 46, BOLD, CYAN))
    print(f"Model: {_style(summary.model, BOLD, CYAN)}")
    print(f"Runs requested: {_style(str(summary.num_times), BOLD, WHITE)}")
    print(f"Runs completed with results: {_style(str(summary.completed_runs), BOLD, WHITE)}")
    if summary.average_pass_rate is not None:
        print(f"Average pass rate: {_style(f'{summary.average_pass_rate:.1f}%', BOLD, GREEN)}")
    else:
        print(f"Average pass rate: {_style('--', BOLD, YELLOW)}")
    print(f"Pass-rate std dev: {_style(f'{summary.std_pass_rate:.1f}%', BOLD, WHITE)}")
    if summary.pass_at_k is not None:
        print(f"Pass@{summary.k}: {_style(f'{summary.pass_at_k:.1f}%', BOLD, GREEN)}")
    else:
        print(f"Pass@{summary.k}: {_style('--', BOLD, YELLOW)}")
    if summary.pass_power_k is not None:
        print(f"Pass^{summary.k}: {_style(f'{summary.pass_power_k:.1f}%', BOLD, GREEN)}")
    else:
        print(f"Pass^{summary.k}: {_style('--', BOLD, YELLOW)}")
    print("")
    for line in format_cost_table(summarize_path_costs(output_dir), heading="BENCHMARK API COSTS"):
        print(line)
    print(f"Results saved to: {_style(str(summary_path), BOLD, CYAN)}")


def main() -> int:
    args = parse_args()
    if args.num_times < 1:
        print("ERROR: --num-times must be at least 1.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_task_ids = _collect_expected_task_ids(args)

    # Launch all repeat runs in parallel
    active_runs: List[ActiveRun] = []
    for run_index in range(1, args.num_times + 1):
        run_output_dir = output_dir / f"run_{run_index}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        active_runs.append(_launch_run(args, run_index, run_output_dir))

    runs = []
    parsed_runs: Dict[int, BenchmarkResults] = {}
    exit_codes = []

    try:
        for active_run in active_runs:
            return_code = active_run.proc.wait()
            active_run.log_handle.close()
            exit_codes.append(return_code)

            parsed = None
            error = ""
            try:
                parsed = _parse_run_results(args, active_run.output_dir)
            except Exception as exc:
                error = str(exc)

            if parsed is None:
                status = "failed"
                run = BenchmarkRepeatRun(
                    run_index=active_run.run_index,
                    output_dir=str(active_run.output_dir),
                    status=status,
                    return_code=return_code,
                    error=error or f"benchmark exited with code {return_code}",
                )
            else:
                status = "complete" if return_code == 0 else "partial"
                run = BenchmarkRepeatRun(
                    run_index=active_run.run_index,
                    output_dir=str(active_run.output_dir),
                    status=status,
                    return_code=return_code,
                    total=parsed.total,
                    passed=parsed.passed,
                    failed=parsed.failed,
                    pass_rate=parsed.pass_rate,
                    error=error,
                )
                parsed_runs[active_run.run_index] = parsed

            runs.append(run)
            if run.pass_rate is not None:
                print(
                    f"[repeat] finished run {active_run.run_index}/{args.num_times}: "
                    f"status={run.status} pass_rate={run.pass_rate:.1f}% "
                    f"({run.passed}/{run.total}) log={active_run.log_path}"
                )
            else:
                print(
                    f"[repeat] finished run {active_run.run_index}/{args.num_times}: "
                    f"status={run.status} error={run.error or 'unknown error'} "
                    f"log={active_run.log_path}"
                )
    finally:
        for active_run in active_runs:
            if active_run.proc.poll() is None:
                kill_proc_group(active_run.proc)
            try:
                active_run.log_handle.close()
            except Exception:
                pass

    summary = build_repeat_summary(
        model=args.model,
        num_times=args.num_times,
        runs=sorted(runs, key=lambda run: run.run_index),
        parsed_runs=parsed_runs,
        expected_task_ids=expected_task_ids,
    )
    summary_path = write_repeat_summary(
        output_dir,
        summary,
        extra_fields={
            "tasks_dir": args.tasks_dir,
            "task": args.task,
            "observation_mode": args.observation_mode,
            "run_mode": args.benchmark_run_mode,
            "category": args.category,
            "max_workers": args.max_workers,
        },
    )

    if not args.no_calibration and args.tasks_dir:
        calibration_source = None
        for run_index in sorted(parsed_runs.keys(), reverse=True):
            run = next((item for item in runs if item.run_index == run_index), None)
            if run and run.return_code == 0:
                calibration_source = parsed_runs[run_index]
                break
        if calibration_source is None and parsed_runs:
            calibration_source = parsed_runs[max(parsed_runs.keys())]

        if calibration_source is not None:
            update_calibration_from_benchmark(
                calibration_source,
                args.tasks_dir,
            )

    _print_summary(summary_path, summary, output_dir)
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
