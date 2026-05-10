#!/usr/bin/env python3
"""
Entry point for external-agent task generation.

Usage:
    python enacttom/task_gen/runner.py --config-name examples/enacttom_2_robots +model=gpt-5

Or via shell script:
    ./enacttom/run.sh generate --model gpt-5 --task-gen-agent mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from enacttom.task_gen.event_log import (
    append_event,
    load_run_manifest,
    load_worker_snapshot,
    maybe_int,
    write_run_manifest,
    write_worker_snapshot,
)
from enacttom.api_costs import format_cost_summary, summarize_worker_costs
from enacttom.task_gen.external_agent import ExternalAgentError, ExternalAgentLauncher
from enacttom.task_gen.authoring_surface import (
    AUTHORING_CONSTRAINTS_NOTICE,
    get_authoring_action_descriptions,
    get_authoring_default_actions,
    get_authoring_mechanics,
    get_authoring_predicates,
)
from enacttom.task_gen.prompts import build_external_taskgen_prompt
from enacttom.task_gen.seed_selector import (
    SeedSelectionConfig,
    is_task_like_json,
    resolve_seed_tasks_dir,
    select_seed_tasks,
)
from enacttom.task_gen.session import TaskGenSession, default_state


DEFAULT_SAMPLED_TASK_COUNT = 30


def parse_extra_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--retry-verification", type=str, default=None)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--calibration-model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--target-pass-rate", type=float, default=0.10)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["cooperative", "mixed"],
    )
    parser.add_argument("--seed-tasks-dir", type=str, default=None)
    parser.add_argument("--seed-pass-ratio", type=float, default=0.20)
    parser.add_argument("--seed-fail-ratio", type=float, default=0.80)
    parser.add_argument("--difficulty", type=str, default="standard", choices=["standard", "hard"])
    parser.add_argument("--sampled-tasks-dir", type=str, default=None)
    parser.add_argument("--no-icl", action="store_true",
                        help="Disable ICL: do not prepare calibration-based sampled tasks")
    parser.add_argument("--judge-threshold", type=float, default=None)
    parser.add_argument(
        "--judge-difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument("--test-model", type=str, default=None)
    parser.add_argument("--k-level", type=int, nargs="*", default=None)
    parser.add_argument(
        "--task-gen-agent",
        type=str,
        default="mini",
        choices=["mini", "claude", "codex"],
        help="External agent CLI used for task generation.",
    )
    parser.add_argument(
        "--remove",
        type=str,
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _build_run_manifest_update(
    existing_manifest: Dict[str, Any],
    *,
    run_id: str,
    generation_mode: str,
    generation_total_workers: Optional[int],
    generation_requested_tasks: Optional[int],
    output_dir: str,
    task_gen_agent: str,
    model: str,
) -> Dict[str, Any]:
    return {
        "run_id": existing_manifest.get("run_id") or run_id,
        "started_at": existing_manifest.get("started_at") or datetime.now().isoformat(),
        "mode": existing_manifest.get("mode") or generation_mode,
        "total_workers": existing_manifest.get("total_workers") or generation_total_workers,
        "requested_tasks": existing_manifest.get("requested_tasks") or generation_requested_tasks,
        "output_dir": existing_manifest.get("output_dir") or output_dir,
        "task_gen_agent": existing_manifest.get("task_gen_agent") or task_gen_agent,
        "model": existing_manifest.get("model") or model,
    }


def parse_runner_args(argv: list[str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "num_tasks": 1,
        "model": None,
        "llm_provider": None,
        "output_dir": "data/enacttom/tasks",
        "subtasks_min": 2,
        "subtasks_max": 5,
        "agents_min": 2,
        "agents_max": 2,
        "quiet": False,
        "config_name": None,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config-name":
            if i + 1 >= len(argv):
                raise SystemExit("Error: --config-name requires a value.")
            config["config_name"] = argv[i + 1]
            i += 2
            continue
        if arg.startswith("+") and "=" in arg:
            key, value = arg[1:].split("=", 1)
            if key in {"num_tasks", "subtasks_min", "subtasks_max", "agents_min", "agents_max"}:
                config[key] = int(value)
            elif key == "quiet":
                config[key] = value.lower() == "true"
            elif key in {"model", "llm_provider", "output_dir"}:
                config[key] = value
            i += 1
            continue
        i += 1

    return config


def _infer_task_tom_level(task_data: dict) -> Optional[int]:
    stored = task_data.get("tom_level")
    if isinstance(stored, int) and 0 <= stored <= 3:
        return stored

    try:
        from enacttom.task_gen.task_generator import GeneratedTask

        task = GeneratedTask.from_dict(task_data)
        level = task.compute_tom_level(scene_data=None)
        if isinstance(level, int):
            return min(max(level, 0), 3)
    except Exception:
        return None
    return None


def build_workspace_id(task_gen_agent: str, now: Optional[datetime] = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-{task_gen_agent}-{uuid.uuid4().hex[:8]}"


def build_generation_run_id(now: Optional[datetime] = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-generation-{uuid.uuid4().hex[:8]}"


def _build_incomplete_exit_reason(
    final_state: Dict[str, Any],
    worker_snapshot: Dict[str, Any],
    return_code: int,
) -> str:
    last_command = worker_snapshot.get("last_command")
    last_command_success = worker_snapshot.get("last_command_success")
    last_command_error = worker_snapshot.get("last_command_error")

    if last_command:
        if last_command_success is False:
            detail = f" after `{last_command}` failed"
            if last_command_error:
                detail += f": {last_command_error}"
            return (
                f"Agent exited without calling taskgen finish or taskgen fail{detail}. "
                f"Agent exit code: {return_code}."
            )
        if last_command_success is True:
            return (
                "Agent exited without calling taskgen finish or taskgen fail "
                f"after `{last_command}` succeeded. Agent exit code: {return_code}."
            )
        return (
            "Agent exited without calling taskgen finish or taskgen fail "
            f"after last observed command `{last_command}`. Agent exit code: {return_code}."
        )

    if final_state.get("submitted_tasks"):
        return (
            "Agent exited without calling taskgen finish or taskgen fail, "
            "but tasks were submitted."
        )

    return (
        "Agent exited without calling taskgen finish or taskgen fail and without "
        "submitting any tasks."
    )


def _attempt_verified_task_submit(
    *,
    working_dir: Path,
    generation_worker_dir: Path,
    generation_run_id: str,
    generation_worker_id: str,
) -> Optional[str]:
    session = TaskGenSession(str(working_dir))
    state = session.state

    if state.get("submitted_tasks"):
        return None
    if not state.get("last_submission_verification_passed"):
        return None

    verification_spec_hash = state.get("last_submission_verification_spec_hash")
    if not verification_spec_hash:
        return "verify_task passed but did not record a verified spec hash; refusing auto-submit."

    try:
        with open(working_dir / "working_task.json") as f:
            task_data = json.load(f)
    except Exception as exc:
        return f"verify_task passed but working_task.json could not be reloaded for submit: {exc}"

    from enacttom.pddl.planner import compute_task_spec_hash

    current_spec_hash = compute_task_spec_hash(task_data)
    if current_spec_hash != verification_spec_hash:
        return (
            "verify_task passed, but working_task.json changed afterward; "
            "refusing auto-submit because the verified spec is no longer current."
        )

    append_event(
        str(generation_worker_dir),
        "auto_submit_started",
        run_id=generation_run_id,
        worker_id=generation_worker_id,
        reason="verified_task_unsubmitted",
        spec_hash=current_spec_hash,
    )
    submit_result = session.submit_task()
    append_event(
        str(generation_worker_dir),
        "auto_submit_finished",
        run_id=generation_run_id,
        worker_id=generation_worker_id,
        success=submit_result["success"],
        error=submit_result.get("error"),
        data=submit_result.get("data"),
    )
    if not submit_result["success"]:
        return (
            "verify_task passed, but automatic submit_task failed: "
            f"{submit_result.get('error') or 'unknown error'}"
        )

    if len(session.state.get("submitted_tasks", [])) >= session.state.get("num_tasks_target", 0):
        finish_result = session.finish()
        append_event(
            str(generation_worker_dir),
            "auto_finish_finished",
            run_id=generation_run_id,
            worker_id=generation_worker_id,
            success=finish_result["success"],
            error=finish_result.get("error"),
            data=finish_result.get("data"),
        )
        if not finish_result["success"]:
            return (
                "submit_task succeeded after verify_task passed, "
                f"but automatic finish failed: {finish_result.get('error') or 'unknown error'}"
            )

    return None


def _required_habitat_asset_paths(project_root: Path) -> Dict[str, Path]:
    episode_path = Path(
        os.environ.get(
            "ENACTTOM_EPISODES_PATH",
            "data/datasets/enacttom_episodes/v0_0/train_2k.json.gz",
        )
    )
    if not episode_path.is_absolute():
        episode_path = project_root / episode_path

    return {
        "HSSD scene dataset config": project_root / "data/hssd-hab/hssd-hab-partnr.scene_dataset_config.json",
        "EnactToM episode file": episode_path,
        "OVMM object configs": project_root / "data/objects_ovmm/train_val/hssd/configs/objects",
        "Spot arm robot URDF": project_root / "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
    }


def _check_required_habitat_assets(project_root: Path) -> None:
    missing = [
        f"{label}: {path}"
        for label, path in _required_habitat_asset_paths(project_root).items()
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Error: missing required Habitat assets:\n- "
            + "\n- ".join(missing)
        )


def _copy_sample(src_path: Path, sampled_tasks_dir: Path, index: int) -> None:
    shutil.copy(src_path, sampled_tasks_dir / f"task_{index}.json")


def _raw_sampled_task_paths(sampled_tasks_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sampled_tasks_dir.glob("*.json")
        if not path.stem.endswith("_fields")
    )


def _write_sampled_task_field_views(sampled_tasks_dir: Path) -> None:
    field_names = [
        "task",
        "active_mechanics",
        "mechanic_bindings",
        "agent_secrets",
        "agent_actions",
        "problem_pddl",
        "num_agents",
    ]

    for task_path in _raw_sampled_task_paths(sampled_tasks_dir):
        try:
            with open(task_path) as f:
                task_data = json.load(f)
        except Exception:
            continue

        filtered = {field: task_data.get(field) for field in field_names}
        view_path = task_path.with_name(f"{task_path.stem}_fields.json")
        with open(view_path, "w") as f:
            json.dump(filtered, f, indent=2)


def populate_sampled_tasks_dir(
    sampled_tasks_dir: Path,
    selection_config: SeedSelectionConfig,
    sample_count: int = DEFAULT_SAMPLED_TASK_COUNT,
) -> tuple[Optional[Path], int]:
    selected = select_seed_tasks(selection_config, count=sample_count)
    for i, candidate in enumerate(selected, 1):
        _copy_sample(candidate.path, sampled_tasks_dir, i)
    _write_sampled_task_field_views(sampled_tasks_dir)
    return selection_config.tasks_dir if selected else None, len(selected)


def _sample_bucket_counts(
    sample_count: int,
    *,
    fail_ratio: float,
    pass_ratio: float,
) -> tuple[int, int]:
    if sample_count <= 0:
        return 0, 0
    if fail_ratio <= 0:
        return 0, sample_count
    if pass_ratio <= 0:
        return sample_count, 0

    total_ratio = fail_ratio + pass_ratio
    pass_count = int(round(sample_count * (pass_ratio / total_ratio)))
    pass_count = max(0, min(sample_count, pass_count))
    fail_count = sample_count - pass_count
    return fail_count, pass_count


def compute_calibration_stats(tasks_dir: Union[str, Iterable[str]], model: str) -> dict:
    from enacttom.benchmark_results import cal_passed, find_calibration_entry

    _empty_cat = lambda: {"passed": 0, "failed": 0, "total": 0, "rate": None}
    stats = {
        "passed": 0,
        "failed": 0,
        "untested": 0,
        "excluded_tom_level_zero": 0,
        "model": model,
        "tom_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "tom_total": 0,
        "tom_unknown": 0,
        "tom_ratios": {0: None, 1: None, 2: None, 3: None},
        "by_category": {
            "cooperative": _empty_cat(),
            "mixed": _empty_cat(),
        },
    }
    task_dirs = [tasks_dir] if isinstance(tasks_dir, str) else list(tasks_dir)
    task_paths = [Path(path) for path in task_dirs if path]
    if not task_paths:
        stats["total"] = 0
        stats["rate"] = None
        return stats

    seen_keys = set()
    task_files = []
    for tasks_path in task_paths:
        if not tasks_path.exists():
            continue
        for task_file in tasks_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
            except Exception:
                continue
            if not isinstance(task, dict):
                continue
            task_key = task.get("task_id") or str(task_file.resolve())
            if task_key in seen_keys:
                continue
            seen_keys.add(task_key)
            task_files.append((task_file, task))

    if not task_files:
        stats["total"] = 0
        stats["rate"] = None
        return stats

    for task_file, task in task_files:
        try:
            tom_level = _infer_task_tom_level(task)
            if tom_level in (0, 1, 2, 3):
                stats["tom_counts"][tom_level] += 1
                stats["tom_total"] += 1
            else:
                stats["tom_unknown"] += 1

            if isinstance(tom_level, int) and tom_level < 1:
                stats["excluded_tom_level_zero"] += 1
                continue

            cal = find_calibration_entry(task.get("calibration", []), model=model)
            category = task.get("category", "cooperative")
            cat_bucket = stats["by_category"].get(category)
            if cal is None:
                stats["untested"] += 1
            elif cal_passed(cal):
                stats["passed"] += 1
                if cat_bucket is not None:
                    cat_bucket["passed"] += 1
            else:
                stats["failed"] += 1
                if cat_bucket is not None:
                    cat_bucket["failed"] += 1
        except Exception:
            continue

    stats["total"] = stats["passed"] + stats["failed"]
    stats["rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else None
    for cat_stats in stats["by_category"].values():
        cat_stats["total"] = cat_stats["passed"] + cat_stats["failed"]
        cat_stats["rate"] = cat_stats["passed"] / cat_stats["total"] if cat_stats["total"] > 0 else None
    if stats["tom_total"] > 0:
        for level in (0, 1, 2, 3):
            stats["tom_ratios"][level] = stats["tom_counts"][level] / stats["tom_total"]
    return stats


def _load_verification_feedback(path_str: Optional[str]) -> Optional[dict]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            verification_data = json.load(f)
    except Exception:
        return None
    if verification_data.get("is_valid_tom", True):
        return None
    return {
        "required_fixes": verification_data.get("required_fixes", []),
        "criteria": verification_data.get("criteria", {}),
        "overall_reasoning": verification_data.get("overall_reasoning", ""),
    }


def _write_template_file(template_file: Path, agents_max: int) -> None:
    source_template = Path(__file__).parent / "template" / "template.json"
    with open(source_template) as f:
        template = json.load(f)
    template["num_agents"] = agents_max
    default_actions = get_authoring_default_actions(include_find_tools=False)
    template["agent_secrets"] = {
        f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"] for i in range(agents_max)
    }
    template["agent_actions"] = {
        f"agent_{i}": default_actions.copy() for i in range(agents_max)
    }
    with open(template_file, "w") as f:
        json.dump(template, f, indent=2)


def _write_taskgen_shim(working_dir: Path) -> None:
    bin_dir = working_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim_path = bin_dir / "taskgen"
    project_root = Path(__file__).resolve().parent.parent.parent
    shim_contents = f"""#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="{project_root}"
export PYTHONPATH="$PROJECT_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
cd "$PROJECT_ROOT"
exec "{sys.executable}" -m enacttom.cli.taskgen --working-dir "{working_dir}" "$@"
"""
    shim_path.write_text(shim_contents)
    shim_path.chmod(0o755)


def _write_bootstrap_files(
    *,
    working_dir: Path,
    prompt_text: str,
    authoring_constraints: str,
    available_mechanics: str,
    available_predicates: str,
    action_descriptions: str,
) -> None:
    (working_dir / "taskgen_prompt.md").write_text(prompt_text)
    (working_dir / "authoring_constraints.md").write_text(authoring_constraints)
    (working_dir / "available_mechanics.md").write_text(available_mechanics)
    (working_dir / "available_predicates.md").write_text(available_predicates)
    (working_dir / "available_actions.md").write_text(action_descriptions)
    (working_dir / "bootstrap_prompt.txt").write_text(prompt_text)


def main() -> None:
    extra_args = getattr(main, "_extra_args", None)
    runner_args = parse_runner_args(sys.argv[1:])

    num_tasks = runner_args["num_tasks"]
    model = runner_args["model"]
    llm_provider = runner_args["llm_provider"]
    output_dir = str(Path(runner_args["output_dir"]).resolve())
    subtasks_min = runner_args["subtasks_min"]
    subtasks_max = runner_args["subtasks_max"]
    agents_min = runner_args["agents_min"]
    agents_max = runner_args["agents_max"]
    quiet = runner_args["quiet"]

    if not model:
        raise SystemExit("Error: model is required.")

    query = extra_args.query if extra_args else None
    retry_verification = extra_args.retry_verification if extra_args else None
    target_model = None
    if extra_args:
        target_model = extra_args.target_model or extra_args.calibration_model
    if not target_model:
        target_model = "gpt-5.4-mini"
    target_pass_rate = extra_args.target_pass_rate if extra_args else 0.10
    category = extra_args.category if extra_args else None
    seed_tasks_dir_arg = extra_args.seed_tasks_dir if extra_args else None
    seed_pass_ratio = extra_args.seed_pass_ratio if extra_args else 0.20
    seed_fail_ratio = extra_args.seed_fail_ratio if extra_args else 0.80
    judge_threshold = extra_args.judge_threshold if extra_args else None
    generation_mode = extra_args.difficulty if extra_args else "standard"
    difficulty = extra_args.judge_difficulty if extra_args else None
    test_model = extra_args.test_model if extra_args else None
    k_levels = extra_args.k_level if extra_args else None
    task_gen_agent = extra_args.task_gen_agent if extra_args else "mini"
    skip_steps = extra_args.remove if extra_args else None
    if skip_steps:
        raise SystemExit("Error: --remove is no longer supported; generation runs the full pipeline.")

    if k_levels is not None:
        invalid = [k for k in k_levels if k not in (1, 2, 3)]
        if invalid:
            raise SystemExit(f"Error: --k-level values must be 1, 2, or 3 (got {invalid})")
        k_levels = sorted(set(k_levels))
    if seed_pass_ratio < 0 or seed_fail_ratio < 0:
        raise SystemExit("Error: --seed-pass-ratio and --seed-fail-ratio must be non-negative.")
    if seed_pass_ratio == 0 and seed_fail_ratio == 0:
        raise SystemExit("Error: at least one of --seed-pass-ratio or --seed-fail-ratio must be positive.")

    verification_feedback = _load_verification_feedback(retry_verification)
    project_root = Path(__file__).resolve().parent.parent.parent
    _check_required_habitat_assets(project_root)
    workspace_root = project_root / "tmp" / "task_gen"
    workspace_root.mkdir(parents=True, exist_ok=True)
    instance_id = build_workspace_id(task_gen_agent)
    working_dir = workspace_root / instance_id
    working_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_generation_path(path_value: Any) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()

    generation_run_id = os.environ.get("ENACTTOM_GENERATION_RUN_ID") or build_generation_run_id()
    generation_run_dir = _resolve_generation_path(
        os.environ.get("ENACTTOM_GENERATION_RUN_DIR")
        or (project_root / "outputs" / "generations" / generation_run_id)
    )
    generation_worker_id = os.environ.get("ENACTTOM_GENERATION_WORKER_ID") or "worker-0"
    generation_worker_dir = _resolve_generation_path(
        os.environ.get("ENACTTOM_GENERATION_WORKER_DIR")
        or (generation_run_dir / "workers" / generation_worker_id)
    )
    generation_mode = os.environ.get("ENACTTOM_GENERATION_MODE") or "single"
    generation_gpu = maybe_int(os.environ.get("ENACTTOM_GENERATION_GPU"))
    generation_slot = maybe_int(os.environ.get("ENACTTOM_GENERATION_SLOT"))
    generation_total_workers = maybe_int(os.environ.get("ENACTTOM_GENERATION_TOTAL_WORKERS"), 1)
    generation_requested_tasks = maybe_int(os.environ.get("ENACTTOM_GENERATION_REQUESTED_TASKS"), num_tasks)
    generation_stdout_log = os.environ.get("ENACTTOM_GENERATION_STDOUT_LOG") or ""
    if generation_stdout_log:
        generation_stdout_log = str(_resolve_generation_path(generation_stdout_log))
    generation_run_dir.mkdir(parents=True, exist_ok=True)
    generation_worker_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ENACTTOM_API_USAGE_LOG"] = str(generation_worker_dir / "api_usage.jsonl")

    sampled_tasks_dir = working_dir / "sampled_tasks"
    sampled_tasks_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "agent_trajectories").mkdir(parents=True, exist_ok=True)
    (working_dir / "submitted_tasks").mkdir(parents=True, exist_ok=True)
    if k_levels:
        current_k_level = random.choice(k_levels)
    else:
        current_k_level = random.choice([1, 2, 3])

    if not test_model:
        test_model = target_model

    seed_tasks_dir = resolve_seed_tasks_dir(seed_tasks_dir_arg, output_dir)
    calibration_task_dirs = []
    if seed_tasks_dir is not None:
        calibration_task_dirs.append(str(seed_tasks_dir))
    calibration_task_dirs.append(output_dir)
    calibration_stats = compute_calibration_stats(calibration_task_dirs, target_model)
    calibration_stats["target_rate"] = target_pass_rate

    _skip_seed_sampling = skip_steps and "seed-sampling" in skip_steps
    _no_icl = extra_args.no_icl if extra_args else False
    sampled_tasks_override = extra_args.sampled_tasks_dir if extra_args else None
    if _skip_seed_sampling:
        pass  # No seed tasks — agent generates from scratch
    elif sampled_tasks_override:
        # Explicit override — copy files as-is
        override_path = Path(sampled_tasks_override)
        override_files = [p for p in override_path.glob("*.json") if is_task_like_json(p)]
        selected = sorted(override_files)[:DEFAULT_SAMPLED_TASK_COUNT]
        for task_path in selected:
            shutil.copy(task_path, sampled_tasks_dir / task_path.name)
        _write_sampled_task_field_views(sampled_tasks_dir)
    elif _no_icl:
        # --no-icl: use basic seed selection (no calibration trajectories)
        if seed_tasks_dir is not None:
            selection_config = SeedSelectionConfig(
                tasks_dir=seed_tasks_dir,
                target_model=target_model,
                target_pass_rate=target_pass_rate,
                current_pass_rate=calibration_stats["rate"],
                category=category,
                tom_level=current_k_level,
                pass_seed_ratio=seed_pass_ratio,
                fail_seed_ratio=seed_fail_ratio,
            )
            populate_sampled_tasks_dir(
                sampled_tasks_dir,
                selection_config,
                sample_count=DEFAULT_SAMPLED_TASK_COUNT,
            )
    elif seed_tasks_dir is not None:
        # Default: ICL with calibration trajectories
        from enacttom.task_gen.icl_sampler import (
            prepare_sampled_tasks_dir_from_calibration,
            compute_pass_rate_from_calibration,
            build_seed_sampling_query,
        )
        fail_count, pass_count = _sample_bucket_counts(
            DEFAULT_SAMPLED_TASK_COUNT,
            fail_ratio=seed_fail_ratio,
            pass_ratio=seed_pass_ratio,
        )
        prepare_sampled_tasks_dir_from_calibration(
            tasks_dir=str(seed_tasks_dir),
            model=target_model,
            output_dir=str(sampled_tasks_dir),
            fail_count=fail_count,
            pass_count=pass_count,
        )
        _write_sampled_task_field_views(sampled_tasks_dir)
        # Build and inject calibrated seed-task guidance if not already set.
        if not query:
            stats = compute_pass_rate_from_calibration(str(seed_tasks_dir), target_model)
            query = build_seed_sampling_query(stats["pass_rate"], target_model, generation_idx=1)
    else:
        pass  # No seed tasks dir — generate from scratch

    _write_template_file(working_dir / "template.json", agents_max)
    _write_taskgen_shim(working_dir)

    authoring_constraints = AUTHORING_CONSTRAINTS_NOTICE
    available_mechanics = get_authoring_mechanics()
    available_predicates = get_authoring_predicates()
    action_descriptions = get_authoring_action_descriptions()

    prompt_text = build_external_taskgen_prompt(
        working_dir=str(working_dir),
        task_file=str(working_dir / "working_task.json"),
        category=category or "random",
        num_tasks=num_tasks,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        query=query,
        verification_feedback=verification_feedback,
        calibration_stats=calibration_stats if not _skip_seed_sampling else {},
        difficulty=difficulty if not _skip_seed_sampling else None,
        current_k_level=current_k_level,
        seed_tasks_dir=(str(seed_tasks_dir) if seed_tasks_dir is not None else None) if not _skip_seed_sampling else None,
        seed_pass_ratio=seed_pass_ratio,
        seed_fail_ratio=seed_fail_ratio,
        skip_steps=skip_steps,
    )
    _write_bootstrap_files(
        working_dir=working_dir,
        prompt_text=prompt_text,
        authoring_constraints=authoring_constraints,
        available_mechanics=available_mechanics,
        available_predicates=available_predicates,
        action_descriptions=action_descriptions,
    )

    state = default_state(
        working_dir=str(working_dir),
        output_dir=output_dir,
        num_tasks_target=num_tasks,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        category=category,
        seed_tasks_dir=str(seed_tasks_dir) if seed_tasks_dir is not None else None,
        seed_pass_ratio=seed_pass_ratio,
        seed_fail_ratio=seed_fail_ratio,
        judge_threshold=judge_threshold,
        difficulty=difficulty,
        test_model=test_model,
        calibration_stats=calibration_stats,
        calibration_tasks_dirs=calibration_task_dirs,
        task_gen_agent=task_gen_agent,
        allowed_k_levels=k_levels,
        skip_steps=skip_steps,
        generation_run_id=generation_run_id,
        generation_run_dir=str(generation_run_dir),
        generation_worker_id=generation_worker_id,
        generation_worker_dir=str(generation_worker_dir),
    )
    state["current_k_level"] = current_k_level
    state["task_gen_model"] = model
    state["task_gen_llm_provider"] = llm_provider
    with open(working_dir / "taskgen_state.json", "w") as f:
        json.dump(state, f, indent=2)
    existing_manifest = load_run_manifest(str(generation_run_dir))
    write_run_manifest(
        generation_run_dir,
        **_build_run_manifest_update(
            existing_manifest,
            run_id=generation_run_id,
            generation_mode=generation_mode,
            generation_total_workers=generation_total_workers,
            generation_requested_tasks=generation_requested_tasks,
            output_dir=output_dir,
            task_gen_agent=task_gen_agent,
            model=model,
        ),
    )
    write_worker_snapshot(
        generation_worker_dir,
        worker_id=generation_worker_id,
        run_id=generation_run_id,
        mode=generation_mode,
        gpu=generation_gpu,
        slot=generation_slot,
        category=category or "random",
        workspace_id=instance_id,
        workspace_path=str(working_dir),
        output_dir=output_dir,
        task_gen_agent=task_gen_agent,
        task_gen_model=model,
        target_tasks=num_tasks,
        submitted_count=0,
        current_task_index=1,
        current_k_level=current_k_level,
        scene_id=None,
        episode_id=None,
        finished=False,
        failed=False,
        fail_reason="",
        status="running",
        agent_trace_path=str(generation_worker_dir / "agent_trace.json"),
        api_usage_log_path=str(generation_worker_dir / "api_usage.jsonl"),
        stdout_log_path=generation_stdout_log,
    )
    if generation_stdout_log:
        write_worker_snapshot(
            generation_worker_dir,
            stdout_log_path=generation_stdout_log,
        )
    append_event(
        generation_worker_dir,
        "workspace_initialized",
        run_id=generation_run_id,
        worker_id=generation_worker_id,
        workspace=str(working_dir),
        task_gen_agent=task_gen_agent,
        model=model,
        llm_provider=llm_provider,
        category=category,
        num_tasks_target=num_tasks,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        current_k_level=current_k_level,
        output_dir=output_dir,
    )

    bootstrap_prompt = (working_dir / "bootstrap_prompt.txt").read_text()

    print("=" * 60)
    print("EnactToM External Task Generator")
    print("=" * 60)
    print(f"Task-gen agent: {task_gen_agent}")
    print(f"Model: {model}")
    print(f"Workspace: {working_dir}")
    print(f"Output: {output_dir}")
    print(f"Target tasks: {num_tasks}")
    if query:
        print(f"Query: {query}")
    if not quiet:
        print("Prompt file:", working_dir / "taskgen_prompt.md")
    print("=" * 60)

    launcher = ExternalAgentLauncher(project_root)
    try:
        append_event(
            generation_worker_dir,
            "agent_launch_started",
            run_id=generation_run_id,
            worker_id=generation_worker_id,
            agent_name=task_gen_agent,
            model=model,
        )
        return_code = launcher.run(
            agent_name=task_gen_agent,
            workspace_dir=working_dir,
            bootstrap_prompt=bootstrap_prompt,
            model=model,
            trace_output_path=generation_worker_dir / "agent_trace.json",
        )
    except ExternalAgentError as exc:
        append_event(
            generation_worker_dir,
            "agent_launch_failed",
            run_id=generation_run_id,
            worker_id=generation_worker_id,
            agent_name=task_gen_agent,
            model=model,
            error=str(exc),
        )
        write_worker_snapshot(
            generation_worker_dir,
            status="failed",
            failed=True,
            fail_reason=str(exc),
        )
        raise SystemExit(str(exc)) from exc

    auto_submit_error = _attempt_verified_task_submit(
        working_dir=working_dir,
        generation_worker_dir=generation_worker_dir,
        generation_run_id=generation_run_id,
        generation_worker_id=generation_worker_id,
    )

    with open(working_dir / "taskgen_state.json") as f:
        final_state = json.load(f)
    worker_snapshot = load_worker_snapshot(str(generation_worker_dir))
    final_submitted_tasks = final_state.get("submitted_tasks", [])
    incomplete_exit_reason = ""
    incomplete_exit = not final_state.get("failed", False) and not final_state.get("finished", False)
    if incomplete_exit:
        incomplete_exit_reason = _build_incomplete_exit_reason(
            final_state,
            worker_snapshot,
            return_code,
        )
    if auto_submit_error:
        incomplete_exit_reason = (
            f"{auto_submit_error} "
            + (incomplete_exit_reason or "")
        ).strip()
    effective_failed = bool(final_state.get("failed", False) or incomplete_exit)
    final_status = (
        "failed"
        if effective_failed
        else "finished"
        if final_state.get("finished", False)
        else "stopped"
    )
    write_worker_snapshot(
        generation_worker_dir,
        status=final_status,
        submitted_count=len(final_submitted_tasks),
        current_task_index=final_state.get("current_task_index"),
        current_k_level=final_state.get("current_k_level"),
        scene_id=final_state.get("scene_id"),
        episode_id=final_state.get("episode_id"),
        finished=final_state.get("finished", False),
        failed=effective_failed,
        fail_reason=final_state.get("fail_reason", "") or incomplete_exit_reason,
        submitted_tasks=final_submitted_tasks,
        workspace_path=str(working_dir),
    )
    cost_summary = summarize_worker_costs(generation_worker_dir)
    write_worker_snapshot(
        generation_worker_dir,
        api_cost_summary=cost_summary,
    )
    append_event(
        generation_worker_dir,
        "generation_finished",
        run_id=generation_run_id,
        worker_id=generation_worker_id,
        agent_name=task_gen_agent,
        model=model,
        return_code=return_code,
        finished=final_state.get("finished", False),
        failed=effective_failed,
        fail_reason=final_state.get("fail_reason", "") or incomplete_exit_reason,
        submitted_tasks=final_state.get("submitted_tasks", []),
        submitted_count=len(final_state.get("submitted_tasks", [])),
        api_cost_summary=cost_summary,
    )

    print()
    print("=" * 60)
    print("Generation Result")
    print("=" * 60)
    print(f"Agent exit code: {return_code}")
    print(f"Finished: {final_state.get('finished', False)}")
    print(f"Failed: {effective_failed}")
    if incomplete_exit_reason:
        print(f"Stop reason: {incomplete_exit_reason}")
    print(f"Workspace retained at: {working_dir}")
    for task_path in final_state.get("submitted_tasks", []):
        print(f"  - {task_path}")
    print()
    for line in format_cost_summary(cost_summary):
        print(line)

    if effective_failed:
        raise SystemExit(
            final_state.get("fail_reason")
            or incomplete_exit_reason
            or "Task generation failed."
        )
    if not final_state.get("finished"):
        # If tasks were submitted despite not finishing, treat as partial success
        if final_submitted_tasks:
            print(
                "Warning: "
                + (
                    incomplete_exit_reason
                    or f"Agent exited without calling taskgen finish, but submitted {len(final_submitted_tasks)} task(s)."
                )
            )
        else:
            raise SystemExit(incomplete_exit_reason or "Task generation did not call taskgen finish.")


if __name__ == "__main__":
    extra_args = parse_extra_args()
    main._extra_args = extra_args
    main()
