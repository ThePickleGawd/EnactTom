"""Benchmark result parsing and calibration helpers."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from enacttom.cli.validate_task import validate_runtime_grounding


def find_calibration_entry(
    calibration,
    model: Optional[str] = None,
    agent_models: Optional[dict] = None,
    run_mode: str = "standard",
) -> Optional[dict]:
    """Find a calibration entry matching the given criteria.

    If agent_models is provided, find exact match on agent_models dict.
    If only model is provided, find entry where ALL agents use that model.
    Returns the most recent match (last in list).

    """
    if not isinstance(calibration, list) or not calibration:
        return None

    best = None
    for entry in calibration:
        entry_agent_models = entry.get("agent_models", {})
        entry_run_mode = str(entry.get("run_mode", "standard") or "standard")
        if entry_run_mode != run_mode:
            continue

        if agent_models is not None:
            if entry_agent_models == agent_models:
                best = entry
        elif model is not None:
            if entry_agent_models and all(
                v == model for v in entry_agent_models.values()
            ):
                best = entry
    return best


def _build_trajectory_from_log(results_dir: str) -> List[Dict[str, Any]]:
    """Build trajectory from planner log files in a benchmark results directory.

    Reads planner-log-*.json files, extracts action_history, and groups by turn.
    Same logic as agent.py:_build_trajectory.
    """
    from collections import defaultdict

    results_path = Path(results_dir)
    log_files = sorted(results_path.glob("planner-log/planner-log-*.json"))
    if not log_files:
        return []

    # Merge action_history from all planner logs
    all_actions: List[Dict[str, Any]] = []
    for log_file in log_files:
        try:
            with open(log_file) as f:
                log_data = json.load(f)
            all_actions.extend(log_data.get("action_history", []))
        except Exception:
            continue

    if not all_actions:
        return []

    # Group by turn
    turns: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "agents": {},
        "subtasks_completed": []
    })

    for record in all_actions:
        turn = record.get("turn", 0)
        if record.get("type") == "subtask_completion":
            turns[turn]["subtasks_completed"].extend(
                record.get("subtasks_completed", [])
            )
        else:
            agent_id = record.get("agent", "unknown")
            agent_entry: Dict[str, Any] = {
                "action": record.get("action", ""),
                "observation": record.get("result", ""),
            }
            if record.get("thought"):
                agent_entry["thought"] = record["thought"]
            turns[turn]["agents"][agent_id] = agent_entry

    trajectory = []
    for turn_num in sorted(turns.keys()):
        entry = turns[turn_num]
        trajectory.append({
            "turn": turn_num,
            "agents": entry["agents"],
            "subtasks_completed": entry["subtasks_completed"],
        })

    return trajectory


def cal_passed(entry: dict) -> bool:
    """Extract passed status from a calibration entry (any category)."""
    results = entry.get("results", {})
    if "main_goal" in results:
        return results["main_goal"].get("passed", False)
    return results.get("passed", False)


def cal_progress(entry: dict) -> float:
    """Extract overall progress from a calibration entry (any category)."""
    results = entry.get("results", {})
    if "main_goal" in results:
        return results["main_goal"].get("progress", 0.0)
    return results.get("progress", 0.0)


@dataclass
class TaskResult:
    task_id: str
    title: str
    task_path: str
    success: bool
    steps: int
    turns: int
    percent_complete: float
    skipped: bool
    error: Optional[str]
    evaluation: dict
    category: str = ""
    run_mode: str = "standard"
    agent_model_mapping: Optional[dict] = None
    results_dir: str = ""


@dataclass
class BenchmarkResults:
    model: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: List[TaskResult] = field(default_factory=list)


def _screen_benchmarkable_tasks(task_files: List[Path]) -> tuple[List[Path], List[str]]:
    """Split task files into benchmarkable and invalid subsets."""
    valid: List[Path] = []
    errors: List[str] = []

    for task_file in task_files:
        try:
            with open(task_file) as f:
                task_data = json.load(f)
        except Exception as e:
            errors.append(f"{task_file.name}: invalid JSON ({e})")
            continue

        grounding_error = validate_runtime_grounding(task_data, scene_data=None)
        if grounding_error:
            task_id = task_data.get("task_id") or task_file.stem
            errors.append(f"{task_id}: {grounding_error}")
            continue

        valid.append(task_file)

    return valid, errors


def run_benchmark(
    tasks_dir: str,
    model: str,
    output_dir: str,
    no_video: bool = True,
    category: Optional[str] = None,
    run_mode: str = "standard",
    observation_mode: str = "text",
    selector_min_frames: int = 1,
    selector_max_frames: int = 5,
    selector_max_candidates: int = 12,
) -> BenchmarkResults:
    """Run benchmark via run_enacttom.sh and parse results.

    Args:
        tasks_dir: Directory containing task JSONs to benchmark.
        model: Model short name (e.g. "haiku", "gpt-5-mini").
        output_dir: Base output directory. Results land at
            <output_dir>-{N}agents/results/benchmark_summary.json.
        no_video: Disable video recording.
        category: Optional task category filter.

    Returns:
        BenchmarkResults with parsed per-task results.
    """
    cmd = [
        "./enacttom/run_enacttom.sh", "benchmark",
        "--tasks-dir", str(tasks_dir),
        "--model", model,
        "--output-dir", str(output_dir),
        "--benchmark-run-mode", run_mode,
        "--observation-mode", observation_mode,
        "--selector-min-frames", str(selector_min_frames),
        "--selector-max-frames", str(selector_max_frames),
        "--selector-max-candidates", str(selector_max_candidates),
    ]
    if not no_video:
        cmd.append("--video")
    if category:
        cmd.extend(["--category", category])

    print(f"[benchmark] Running benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark command failed (exit={result.returncode}) for model={model}, tasks_dir={tasks_dir}"
        )

    return parse_benchmark_results(output_dir, model)


def parse_benchmark_results(output_dir: str, model: str) -> BenchmarkResults:
    """Parse benchmark_summary.json files from output directory.

    Merges results across agent-count groups by globbing
    <output_dir>-*agents/results/benchmark_summary.json.
    """
    output_path = Path(output_dir)
    summary_files = sorted(output_path.parent.glob(f"{output_path.name}-*agents/results/benchmark_summary.json"))

    if not summary_files:
        # Try exact path as well (single-agent-count case)
        exact = output_path / "results" / "benchmark_summary.json"
        if exact.exists():
            summary_files = [exact]

    if not summary_files:
        raise FileNotFoundError(
            f"No benchmark_summary.json found under '{output_dir}'. "
            "Benchmark likely failed before writing results."
        )

    all_task_results: List[TaskResult] = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for sf in summary_files:
        with open(sf) as f:
            summary = json.load(f)

        # Derive results directory from the summary file location
        results_base_dir = str(sf.parent)

        for r in summary.get("results", []):
            if r.get("skipped", False):
                total_skipped += 1
                continue

            evaluation = r.get("evaluation", {})
            task_id = r.get("task_id", "")
            task_result = TaskResult(
                task_id=task_id,
                title=r.get("title", ""),
                task_path=r.get("task_id", ""),  # Will be resolved later
                success=r.get("success", False),
                steps=r.get("steps", 0),
                turns=r.get("turns", 0),
                percent_complete=evaluation.get("percent_complete", 0.0),
                skipped=False,
                error=r.get("error"),
                evaluation=evaluation,
                category=r.get("category", ""),
                run_mode=r.get("run_mode", summary.get("run_mode", "standard")),
                agent_model_mapping=r.get("agent_model_mapping"),
                results_dir=f"{results_base_dir}/{task_id}" if task_id else "",
            )
            all_task_results.append(task_result)

            if task_result.success:
                total_passed += 1
            else:
                total_failed += 1

    total = total_passed + total_failed
    if total == 0:
        raise RuntimeError(
            f"Benchmark produced zero non-skipped results for '{output_dir}' "
            f"(skipped={total_skipped}, summaries={len(summary_files)})."
        )

    pass_rate = (total_passed / total * 100) if total > 0 else 0.0

    return BenchmarkResults(
        model=model,
        total=total,
        passed=total_passed,
        failed=total_failed,
        pass_rate=pass_rate,
        results=all_task_results,
    )


def kill_proc_group(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Send SIGTERM to the process group, then SIGKILL if still alive after timeout."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return  # process already gone

    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


def _detect_gpu_ids() -> List[int]:
    """Detect available CUDA GPU IDs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(x.strip()) for x in result.stdout.strip().split("\n")]
    except Exception:
        pass
    return [0]


def _build_subprocess_env(base_env: Dict[str, str], out_path: Path, stem: str, gpu_id: int) -> Dict[str, str]:
    """Build a subprocess environment that keeps temp files off the root filesystem."""
    tmp_root = out_path / "_tmp" / stem
    cache_root = out_path / "_cache" / stem

    for path in (
        tmp_root,
        cache_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    env = dict(base_env)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "TMPDIR": str(tmp_root),
            "TMP": str(tmp_root),
            "TEMP": str(tmp_root),
            "XDG_CACHE_HOME": str(cache_root),
        }
    )
    return env


def run_benchmark_parallel(
    tasks_dir: str,
    model: str,
    output_dir: str,
    max_workers: int = 50,
    workers_per_gpu: Optional[int] = None,
    no_video: bool = True,
    category: Optional[str] = None,
    run_mode: str = "standard",
    extra_args: Optional[List[str]] = None,
    write_calibration: bool = True,
    observation_mode: str = "text",
    selector_min_frames: int = 1,
    selector_max_frames: int = 5,
    selector_max_candidates: int = 12,
    log_prefix: str = "[benchmark]",
) -> BenchmarkResults:
    """Run benchmark in parallel — one process per task JSON.

    For each task file, creates a temp single-task directory and spawns
    a separate benchmark process. Manages a pool of up to max_workers
    concurrent processes. Merges all results into a single BenchmarkResults.

    Each process is assigned a GPU via round-robin over all available CUDA
    devices (set through CUDA_VISIBLE_DEVICES).

    Args:
        tasks_dir: Directory containing task JSONs.
        model: Model short name.
        output_dir: Base output directory for per-task results.
        max_workers: Maximum concurrent benchmark processes.
        no_video: Disable video recording.
        category: Optional category filter.
        extra_args: Additional args forwarded to each subprocess benchmark call.
        write_calibration: Write calibration to source task JSONs as each task completes.
        log_prefix: Prefix used for progress and warning logs.

    Returns:
        Merged BenchmarkResults across all tasks.
    """
    tasks_path = Path(tasks_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_dir = out_path / "logs"
    log_dir.mkdir(exist_ok=True)

    task_files = sorted(tasks_path.glob("*.json"))
    if not task_files:
        print(f"{log_prefix} WARNING: no task files in {tasks_dir}", file=sys.stderr)
        return BenchmarkResults(model=model, total=0, passed=0, failed=0, pass_rate=0.0)

    # Pre-filter by category so we don't spawn unnecessary subprocesses
    if category:
        filtered = []
        for tf in task_files:
            try:
                with open(tf) as f:
                    task_data = json.load(f)
                if task_data.get("category") == category:
                    filtered.append(tf)
            except Exception:
                filtered.append(tf)  # include on error, let subprocess handle it
        task_files = filtered
        if not task_files:
            print(f"{log_prefix} WARNING: no {category} tasks in {tasks_dir}", file=sys.stderr)
            return BenchmarkResults(model=model, total=0, passed=0, failed=0, pass_rate=0.0)

    task_files, invalid_task_errors = _screen_benchmarkable_tasks(task_files)
    if invalid_task_errors:
        preview = "\n".join(f"  - {msg}" for msg in invalid_task_errors[:10])
        print(
            f"{log_prefix} WARNING: skipping {len(invalid_task_errors)} unbenchmarkable task(s):\n{preview}",
            file=sys.stderr,
        )
    if not task_files:
        raise RuntimeError(
            "No benchmarkable tasks found. "
            "All selected tasks use placeholder or invalid runtime scene metadata."
        )

    # Detect GPUs for round-robin distribution
    gpu_ids = _detect_gpu_ids()
    if workers_per_gpu is not None:
        max_workers = len(gpu_ids) * workers_per_gpu

    # Prepare per-task jobs: (task_stem, task_input_dir, benchmark_output_dir)
    jobs = []
    for tf in task_files:
        stem = tf.stem
        task_input_dir = out_path / stem / "task_input"
        task_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tf, task_input_dir / tf.name)
        bench_out = str(out_path / stem / "benchmark")
        jobs.append((stem, str(task_input_dir), bench_out))

    # Skip tasks that already have results (resume after partial run)
    skipped_stems: List[str] = []
    resumable_jobs = []
    for stem, task_input, bench_out in jobs:
        # Check all possible benchmark-Nagents subdirs for a summary
        bench_path = Path(bench_out)
        parent = bench_path.parent
        has_result = False
        if parent.exists():
            for d in parent.iterdir():
                if d.is_dir() and d.name.startswith("benchmark-"):
                    sf = d / "results" / "benchmark_summary.json"
                    if sf.exists():
                        has_result = True
                        break
        if has_result:
            skipped_stems.append(stem)
        else:
            resumable_jobs.append((stem, task_input, bench_out))

    if skipped_stems:
        print(
            f"{log_prefix} Resuming: skipping {len(skipped_stems)} task(s) with existing results",
            file=sys.stderr,
        )
    jobs = resumable_jobs

    total_tasks = len(jobs)
    job_idx = 0
    active: List[tuple] = []  # (stem, bench_out, Popen, log_file_handle)
    completed_stems: List[str] = []
    failed_stems: List[str] = []

    spinner_chars = ["|", "/", "-", "\\"]
    spinner_idx = 0
    last_status = None

    print(f"{log_prefix} Parallel benchmark: {total_tasks} tasks, max_workers={max_workers}")
    print(f"{log_prefix} GPUs detected: {gpu_ids} ({len(gpu_ids)} devices)")
    print(f"{log_prefix} Benchmark output dir: {out_path.resolve()}")
    print(f"{log_prefix} Benchmark logs: {log_dir.resolve()}")

    try:
        while True:
            # Reap finished processes
            still_active = []
            for stem, bench_out, proc, fh in active:
                if proc.poll() is not None:
                    fh.close()
                    completed_stems.append(stem)
                    if proc.returncode != 0:
                        failed_stems.append(stem)
                        print(
                            f"{log_prefix} WARNING: benchmark for {stem} exited with code {proc.returncode}",
                            file=sys.stderr,
                        )
                    elif write_calibration:
                        # Write calibration to original tasks dir immediately
                        try:
                            per_task = parse_benchmark_results(bench_out, model)
                            update_calibration_from_benchmark(
                                per_task, str(tasks_path)
                            )
                        except FileNotFoundError:
                            pass  # task produced no results (skipped)
                        except Exception as e:
                            print(
                                f"{log_prefix} WARNING: calibration failed for {stem}: {e}",
                                file=sys.stderr,
                            )
                else:
                    still_active.append((stem, bench_out, proc, fh))
            active = still_active

            # Spawn new processes
            while len(active) < max_workers and job_idx < total_tasks:
                stem, task_input, bench_out = jobs[job_idx]
                cmd = [
                    "./enacttom/run_enacttom.sh", "benchmark",
                    "--tasks-dir", task_input,
                    "--model", model,
                    "--output-dir", bench_out,
                    "--benchmark-run-mode", run_mode,
                    "--observation-mode", observation_mode,
                    "--selector-min-frames", str(selector_min_frames),
                    "--selector-max-frames", str(selector_max_frames),
                    "--selector-max-candidates", str(selector_max_candidates),
                    "--no-calibration",
                    "--num-times", "1",
                ]
                if not no_video:
                    cmd.append("--video")
                if category:
                    cmd.extend(["--category", category])
                if extra_args:
                    cmd.extend(extra_args)

                # Round-robin GPU assignment and keep temp files under the benchmark output tree.
                gpu_id = gpu_ids[job_idx % len(gpu_ids)]
                env = _build_subprocess_env(os.environ, out_path, stem, gpu_id)

                log_file = log_dir / f"bench_{stem}.log"
                fh = open(log_file, "w")
                proc = subprocess.Popen(
                    cmd, stdout=fh, stderr=fh, env=env, start_new_session=True
                )
                active.append((stem, bench_out, proc, fh))
                job_idx += 1

            done = len(completed_stems)
            if done >= total_tasks and not active:
                if sys.stdout.isatty():
                    print()  # finish in-place status line
                print(f"{log_prefix} Benchmark: {done}/{total_tasks} tasks complete — done!")
                break

            spinner = spinner_chars[spinner_idx % len(spinner_chars)]
            spinner_idx += 1
            status = (
                f"{log_prefix} {spinner} benchmarking: {done}/{total_tasks} tasks complete "
                f"(active={len(active)})"
            )
            if sys.stdout.isatty():
                print(f"\r{status}", end="", flush=True)
            elif status != last_status:
                print(status)
                last_status = status

            if not active and job_idx >= total_tasks:
                if sys.stdout.isatty():
                    print()  # finish in-place status line
                break

            time.sleep(10)
    finally:
        for stem, bench_out, proc, fh in active:
            kill_proc_group(proc)
            fh.close()

    if failed_stems:
        print(
            f"{log_prefix} WARNING: {len(failed_stems)} task(s) failed and will be excluded "
            f"from results: {', '.join(sorted(failed_stems)[:10])}. "
            f"See logs in {log_dir}",
            file=sys.stderr,
        )

    # Merge results from all per-task benchmark outputs (new + resumed)
    all_task_results: List[TaskResult] = []
    total_passed = 0
    total_failed = 0

    # Include results from tasks skipped due to resume (already had results)
    all_jobs_for_results = list(jobs)
    for stem in skipped_stems:
        task_input = str(out_path / stem / "task_input")
        bench_out = str(out_path / stem / "benchmark")
        all_jobs_for_results.append((stem, task_input, bench_out))

    no_result_stems: List[str] = []
    for stem, task_input, bench_out in all_jobs_for_results:
        try:
            per_task = parse_benchmark_results(bench_out, model)
        except FileNotFoundError:
            # Task was skipped (e.g. category mismatch) or produced no results
            no_result_stems.append(stem)
            continue
        except Exception as e:
            print(
                f"{log_prefix} WARNING: failed parsing results for '{stem}': {e}",
                file=sys.stderr,
            )
            no_result_stems.append(stem)
            continue
        all_task_results.extend(per_task.results)
        total_passed += per_task.passed
        total_failed += per_task.failed

    if no_result_stems:
        print(
            f"{log_prefix} {len(no_result_stems)} task(s) produced no results (skipped or failed)"
        )

    total = total_passed + total_failed
    pass_rate = (total_passed / total * 100) if total > 0 else 0.0

    return BenchmarkResults(
        model=model,
        total=total,
        passed=total_passed,
        failed=total_failed,
        pass_rate=pass_rate,
        results=all_task_results,
    )


def parse_parallel_benchmark_results(output_dir: str, model: str) -> BenchmarkResults:
    """Parse a parallel benchmark output directory without rerunning anything."""
    out_path = Path(output_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"Parallel benchmark output dir not found: {output_dir}")

    all_task_results: List[TaskResult] = []
    total_passed = 0
    total_failed = 0

    no_result_stems: List[str] = []
    for task_dir in sorted(path for path in out_path.iterdir() if path.is_dir() and path.name != "logs"):
        bench_out = str(task_dir / "benchmark")
        try:
            per_task = parse_benchmark_results(bench_out, model)
        except FileNotFoundError:
            no_result_stems.append(task_dir.name)
            continue
        except Exception:
            no_result_stems.append(task_dir.name)
            continue
        all_task_results.extend(per_task.results)
        total_passed += per_task.passed
        total_failed += per_task.failed

    total = total_passed + total_failed
    pass_rate = (total_passed / total * 100) if total > 0 else 0.0

    return BenchmarkResults(
        model=model,
        total=total,
        passed=total_passed,
        failed=total_failed,
        pass_rate=pass_rate,
        results=all_task_results,
    )


def _build_agent_models_from_result(
    result: TaskResult,
    model: str,
    task_data: Optional[dict] = None,
) -> Dict[str, str]:
    """Build per-agent model mapping from a TaskResult.

    Priority:
    1. result.agent_model_mapping (from benchmark_summary.json)
    2. Fall back to all agents using `model`
    """
    # 1. Direct agent_model_mapping from benchmark results
    if result.agent_model_mapping:
        return {
            agent_id: info.get("model", model)
            for agent_id, info in result.agent_model_mapping.items()
        }

    # 2. All agents use the same model
    num_agents = task_data.get("num_agents", 2) if task_data else 2
    return {f"agent_{i}": model for i in range(num_agents)}


def update_calibration_from_benchmark(
    benchmark_results: BenchmarkResults,
    tasks_dir: str,
) -> None:
    """Write benchmark results back into task JSONs as calibration entries.

    Uses the unified array-based calibration format. Each entry records
    per-agent model mappings, category-specific fields, and trajectory.
    Deduplicates by agent_models (replaces existing entry with same matchup).

    Args:
        benchmark_results: Parsed benchmark results.
        tasks_dir: Directory containing source task JSONs.
    """
    tasks_path = Path(tasks_dir)
    model = benchmark_results.model

    # Build lookup: task_id -> TaskResult
    result_map = {r.task_id: r for r in benchmark_results.results}

    updated = 0
    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task_data = json.load(f)
        except Exception:
            continue

        task_id = task_data.get("task_id", "")
        # Skip placeholder IDs — they are shared across many tasks and would corrupt calibration
        if task_id == "REPLACE_WITH_UNIQUE_ID":
            task_id = ""
        # Try matching by task_id or by filename stem
        result = result_map.get(task_id) or result_map.get(task_file.stem)
        if result is None:
            continue

        raw_cal = task_data.get("calibration", [])
        calibration = raw_cal if isinstance(raw_cal, list) else []

        # Build per-agent model mapping
        agent_models = _build_agent_models_from_result(
            result, model, task_data
        )

        # Build calibration entry with structured results
        evaluation = result.evaluation
        category = result.category or task_data.get("category", "")

        if category == "mixed":
            agents = {
                aid: {"subgoal_passed": passed}
                for aid, passed in evaluation.get("agent_subgoal_status", {}).items()
            }
            results_block = {
                "main_goal": {
                    "passed": evaluation.get("main_goal_success", False),
                    "progress": evaluation.get("main_goal_progress", result.percent_complete),
                },
                "agents": agents,
            }

        else:
            # Cooperative / default
            results_block = {
                "passed": result.success,
                "progress": result.percent_complete,
            }

        entry: Dict[str, Any] = {
            "tested_at": datetime.now().isoformat(),
            "run_mode": result.run_mode or "standard",
            "agent_models": agent_models,
            "steps": result.steps,
            "results": results_block,
        }

        # Build trajectory from planner logs if available
        if result.results_dir:
            trajectory = _build_trajectory_from_log(result.results_dir)
            if trajectory:
                entry["trajectory"] = trajectory

        # Deduplicate: replace existing entry with same agent_models, else append
        replaced = False
        for i, existing in enumerate(calibration):
            existing_run_mode = str(existing.get("run_mode", "standard") or "standard")
            if (
                existing.get("agent_models") == agent_models
                and existing_run_mode == entry["run_mode"]
            ):
                calibration[i] = entry
                replaced = True
                break
        if not replaced:
            calibration.append(entry)

        task_data["calibration"] = calibration

        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)
        updated += 1

    print(f"[calibration] Updated {updated} task(s) in {tasks_dir} for model {model}")
