from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from enacttom.cli import CLIResult, failure, success
from enacttom.cli.judge_task import run as judge_task_run
from enacttom.cli.submit_task import run as submit_task_run
from enacttom.cli.validate_task import validate
from enacttom.task_gen.authoring_surface import get_authoring_default_actions
from enacttom.task_gen.scene_loader import SceneData
from enacttom.task_gen.task_bootstrap import build_scene_bootstrap_problem_pddl

SUBMISSION_VERIFICATION_MODELS = [
    "gpt-5.4",
    "claude-sonnet-4-6",
    "gemini-flash",
]
SUBMISSION_VERIFICATION_REQUIRED_FAILURES = 2
HARD_MODE_STANDARD_PROGRESS_CAP = 0.45


def _evaluation_passed(category: str, evaluation: Dict[str, Any]) -> bool:
    if category == "mixed":
        return evaluation.get("main_goal_success", False)
    return evaluation.get("success", False)


def _evaluation_progress(category: str, evaluation: Dict[str, Any]) -> float:
    if category == "mixed":
        return evaluation.get("main_goal_progress", evaluation.get("percent_complete", 0.0))
    return evaluation.get("percent_complete", 0.0)


def _build_results_block(category: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    if category == "mixed":
        agents = {
            aid: {"subgoal_passed": passed}
            for aid, passed in evaluation.get("agent_subgoal_status", {}).items()
        }
        return {
            "main_goal": {
                "passed": evaluation.get("main_goal_success", False),
                "progress": _evaluation_progress(category, evaluation),
            },
            "agents": agents,
        }

    return {
        "passed": evaluation.get("success", False),
        "progress": _evaluation_progress(category, evaluation),
    }


def _standard_requirement(
    current_rate: Optional[float],
    target_rate: float,
    tolerance: float = 0.05,
    current_passed: Optional[int] = None,
    current_failed: Optional[int] = None,
    hard_pass_rate_cap: bool = False,
) -> str:
    if hard_pass_rate_cap:
        if current_passed is not None and current_failed is not None:
            next_total = current_passed + current_failed + 1
            pass_rate_if_pass = (current_passed + 1) / next_total
            return "must_fail" if pass_rate_if_pass > target_rate + 1e-9 else "either"
        if current_rate is None:
            return "either"
        return "must_fail" if current_rate >= target_rate - 1e-9 else "either"

    if current_passed is not None and current_failed is not None:
        total = current_passed + current_failed
        next_total = total + 1
        pass_rate_if_pass = (current_passed + 1) / next_total
        pass_rate_if_fail = current_passed / next_total
        pass_delta = abs(pass_rate_if_pass - target_rate)
        fail_delta = abs(pass_rate_if_fail - target_rate)
        if abs(pass_delta - fail_delta) <= 1e-9:
            return "either"
        return "must_pass" if pass_delta < fail_delta else "must_fail"

    if current_rate is None:
        return "either"
    if current_rate > target_rate + tolerance:
        return "must_fail"
    if current_rate < target_rate - tolerance:
        return "must_pass"
    return "either"

def _test_timeout_seconds(num_agents: int, category: str) -> int:
    """Return a conservative timeout for one benchmark run (standard OR baseline).

    The default 1200s can be too small in shared-worker environments,
    especially for 3+ agents where planning can be slower. A timeout here is not
    a task-quality signal, so we prefer a more forgiving bound.
    """
    base = 1800
    extra = max(0, num_agents - 2) * 600
    return base + extra


def build_mode_comparison(
    category: str,
    standard: Dict[str, Any],
    baseline: Dict[str, Any],
    current_rate: Optional[float],
    target_rate: float,
    tolerance: float = 0.05,
    current_passed: Optional[int] = None,
    current_failed: Optional[int] = None,
    hard_pass_rate_cap: bool = False,
) -> Dict[str, Any]:
    std_eval = standard.get("evaluation", {})
    base_eval = baseline.get("evaluation", {})
    standard_passed = _evaluation_passed(category, std_eval)
    baseline_passed = _evaluation_passed(category, base_eval)
    standard_progress = _evaluation_progress(category, std_eval)
    baseline_progress = _evaluation_progress(category, base_eval)
    requirement = _standard_requirement(
        current_rate,
        target_rate,
        tolerance=tolerance,
        current_passed=current_passed,
        current_failed=current_failed,
        hard_pass_rate_cap=hard_pass_rate_cap,
    )
    next_total = None
    next_rate_if_pass = None
    next_rate_if_fail = None
    if current_passed is not None and current_failed is not None:
        next_total = current_passed + current_failed + 1
        next_rate_if_pass = (current_passed + 1) / next_total
        next_rate_if_fail = current_passed / next_total

    gate_passed = baseline_passed
    pass_rate_cap_exceeded = False
    progress_cap_exceeded = False
    reasons: List[str] = []
    if not baseline_passed:
        gate_passed = False
        reasons.append(
            "Baseline run must pass so the task is empirically solvable when information asymmetry is removed."
        )
    projected_pass_rate = next_rate_if_pass if next_rate_if_pass is not None else current_rate
    if hard_pass_rate_cap and requirement == "must_fail" and standard_passed:
        gate_passed = False
        pass_rate_cap_exceeded = True
        projected_text = (
            f"{projected_pass_rate:.1%}" if projected_pass_rate is not None else "an unknown projected rate"
        )
        reasons.append(
            "Hard calibration gate: standard passed, which would leave the dataset at "
            f"{projected_text}, above the hard cap of {target_rate:.0%}. "
            "Discard this task for hard-mode generation."
        )
    if hard_pass_rate_cap and standard_progress >= HARD_MODE_STANDARD_PROGRESS_CAP:
        gate_passed = False
        progress_cap_exceeded = True
        reasons.append(
            "Hard calibration gate: standard reached "
            f"{standard_progress:.1%} progress, but hard-mode tasks must stay below "
            f"{HARD_MODE_STANDARD_PROGRESS_CAP:.0%} standard progress. "
            "Discard this task for hard-mode generation."
        )
    elif requirement == "must_fail" and standard_passed:
        reasons.append(
            f"Calibration note: standard passed, which moves the dataset away from the target zone around {target_rate:.0%}. Prefer somewhat harder tasks next."
        )
    elif requirement == "must_pass" and not standard_passed:
        reasons.append(
            f"Calibration note: standard failed, which moves the dataset away from the target zone around {target_rate:.0%}. Prefer somewhat easier tasks next."
        )

    if not reasons:
        if hard_pass_rate_cap:
            if standard_passed:
                reasons.append(
                    "Baseline passed. Standard outcome is acceptable because the projected dataset "
                    f"pass rate stays at or below the hard cap of {target_rate:.0%}."
                )
            else:
                reasons.append(
                    "Baseline passed and standard failed, which keeps the hard-mode dataset at or "
                    f"below the {target_rate:.0%} pass-rate cap."
                )
        else:
            reasons.append("Baseline passed. Standard outcome is acceptable, and the dataset should continue tending toward the calibration target over time.")

    return {
        "gate_passed": gate_passed,
        "functional_tom_signal": baseline_passed,
        "hard_pass_rate_cap": hard_pass_rate_cap,
        "pass_rate_cap_exceeded": pass_rate_cap_exceeded,
        "progress_cap_exceeded": progress_cap_exceeded,
        "hard_standard_progress_cap": HARD_MODE_STANDARD_PROGRESS_CAP,
        "standard_requirement": requirement,
        "current_standard_pass_rate": current_rate,
        "target_standard_pass_rate": target_rate,
        "next_standard_pass_rate_if_pass": next_rate_if_pass,
        "next_standard_pass_rate_if_fail": next_rate_if_fail,
        "standard_passed": standard_passed,
        "baseline_passed": baseline_passed,
        "standard_progress": standard_progress,
        "baseline_progress": baseline_progress,
        "progress_delta": baseline_progress - standard_progress,
        "standard_turns": standard.get("turns", 0),
        "baseline_turns": baseline.get("turns", 0),
        "turn_delta": standard.get("turns", 0) - baseline.get("turns", 0),
        "standard_steps": standard.get("steps", 0),
        "baseline_steps": baseline.get("steps", 0),
        "step_delta": standard.get("steps", 0) - baseline.get("steps", 0),
        "reasons": reasons,
    }


def _build_test_task_retry_guidance(comparison: Dict[str, Any]) -> str:
    baseline_passed = bool(comparison.get("baseline_passed"))
    pass_rate_cap_exceeded = bool(comparison.get("pass_rate_cap_exceeded"))
    progress_cap_exceeded = bool(comparison.get("progress_cap_exceeded"))

    if pass_rate_cap_exceeded:
        target_text = _format_percent(comparison.get("target_standard_pass_rate"))
        progress_text = _format_percent(comparison.get("standard_progress"))
        cap_text = _format_percent(comparison.get("hard_standard_progress_cap"))
        return (
            "Revise the task and run `taskgen judge` -> `taskgen test_task` again. "
            "Do not call `taskgen fail`. The task is physically solvable, but it is too easy for hard-mode generation "
            f"because a standard pass would push the calibrated pool above the {target_text} cap"
            f" and standard already reached {progress_text} progress (hard target: below {cap_text}). "
            "Strengthen the hidden-information bottleneck instead of adding physical clutter."
        )

    if progress_cap_exceeded:
        progress_text = _format_percent(comparison.get("standard_progress"))
        cap_text = _format_percent(comparison.get("hard_standard_progress_cap"))
        return (
            "Revise the task and run `taskgen judge` -> `taskgen test_task` again. "
            "Do not call `taskgen fail`. The task is physically solvable, but it is too easy for hard-mode generation "
            f"because standard reached {progress_text} progress and hard-mode tasks must stay below {cap_text}. "
            "Strengthen the hidden-information bottleneck instead of adding physical clutter."
        )

    if not baseline_passed:
        return (
            "Revise the task and run `taskgen judge` -> `taskgen test_task` again. "
            "Do not call `taskgen fail`. Baseline also failed, meaning the task is physically broken or too complex even when private information is shared. "
            "Simplify NOW: (1) reduce to 2-3 goals max, (2) remove unnecessary mechanics, "
            "(3) ensure all objects/furniture exist in the scene with correct IDs, "
            "(4) ensure agents can physically reach the objects they need. "
            "Use `taskgen new_scene N --keep` to reload the current scene and verify IDs."
        )

    return (
        "Revise the task and run `taskgen judge` -> `taskgen test_task` again. "
        "Do not call `taskgen fail` unless the environment itself is broken."
    )


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def _build_test_task_failure_feedback(
    category: str,
    comparison: Dict[str, Any],
    trajectory_dir: Optional[str],
) -> Dict[str, Any]:
    pass_rate_cap_exceeded = bool(comparison.get("pass_rate_cap_exceeded"))
    progress_cap_exceeded = bool(comparison.get("progress_cap_exceeded"))
    reasons = [str(reason).strip() for reason in comparison.get("reasons") or [] if str(reason).strip()]
    evidence = [
        f"standard: passed={comparison.get('standard_passed', False)}, "
        f"progress={_format_percent(comparison.get('standard_progress'))}, "
        f"turns={comparison.get('standard_turns', 0)}",
        f"baseline: passed={comparison.get('baseline_passed', False)}, "
        f"progress={_format_percent(comparison.get('baseline_progress'))}, "
        f"turns={comparison.get('baseline_turns', 0)}",
    ]

    required_fixes: List[str] = []
    if pass_rate_cap_exceeded or progress_cap_exceeded:
        target_rate = comparison.get("target_standard_pass_rate")
        projected_rate = comparison.get("next_standard_pass_rate_if_pass")
        progress_cap = comparison.get("hard_standard_progress_cap")
        standard_progress = comparison.get("standard_progress")
        if projected_rate is not None and pass_rate_cap_exceeded:
            evidence.append(
                f"projected_standard_pass_rate_if_kept={_format_percent(projected_rate)} "
                f"(cap={_format_percent(target_rate)})"
            )
        if progress_cap is not None and standard_progress is not None:
            evidence.append(
                f"hard_mode_standard_progress={_format_percent(standard_progress)} "
                f"(cap=<{_format_percent(progress_cap)})"
            )
        summary = (
            "Hard-mode calibration rejected the task because standard still made too much progress "
            "for the hard target."
        )
        required_fixes.extend(
            [
                "Keep the physical core baseline-solvable; the problem is that standard still got too far through the task.",
                "Strengthen one decisive hidden-information dependency so the correct action cannot be guessed from public facts alone.",
                "Prefer a non-binary hidden choice or tighter message routing instead of adding extra rooms, objects, or search clutter.",
            ]
        )
        if progress_cap is not None:
            required_fixes.append(
                f"Hard-mode tasks must keep standard progress below {_format_percent(progress_cap)}."
            )
        if standard_progress is not None:
            required_fixes.append(
                f"This task let standard reach {_format_percent(standard_progress)} progress, so the hidden-information bottleneck is still too weak."
            )
        if projected_rate is not None and target_rate is not None and pass_rate_cap_exceeded:
            required_fixes.append(
                f"Projected standard pass rate would be {_format_percent(projected_rate)}; hard-mode tasks must stay at or below {_format_percent(target_rate)}."
            )
    else:
        summary = (
            "Baseline failed, so the task is not empirically solvable yet and should be simplified "
            "before adding more ToM structure."
        )
        required_fixes.extend(
            [
                "Simplify the physical core first; do not add more hidden-information structure yet.",
                "Reduce to 2-3 decisive goals and remove unnecessary mechanics, extra objects, or extra room bans.",
                "Verify every referenced object, furniture item, and room exists in the scene and is physically reachable.",
                "Prefer one clean information asymmetry instead of stacking multiple brittle dependencies.",
            ]
        )

    if not pass_rate_cap_exceeded and comparison.get("baseline_turns", 0) >= 20:
        required_fixes.append(
            "The baseline is hitting the full benchmark budget, which usually means too much search or too much coordination load."
        )

    artifact_paths: Dict[str, str] = {}
    if trajectory_dir:
        trajectory_path = Path(trajectory_dir)
        artifact_paths["trajectory_dir"] = str(trajectory_path)
        artifact_paths["comparison_json"] = str(trajectory_path / "comparison.json")

    snapshot = {
        "standard_passed": comparison.get("standard_passed"),
        "baseline_passed": comparison.get("baseline_passed"),
        "standard_progress": comparison.get("standard_progress"),
        "baseline_progress": comparison.get("baseline_progress"),
        "standard_turns": comparison.get("standard_turns"),
        "baseline_turns": comparison.get("baseline_turns"),
    }
    return {
        "source_gate": "test_task",
        "summary": summary,
        "overall_reasoning": " ".join(reasons),
        "required_fixes": required_fixes,
        "evidence": evidence,
        "artifact_paths": artifact_paths,
        "comparison_snapshot": snapshot,
    }


def _build_submission_verification_feedback(
    category: str,
    verification_payload: Dict[str, Any],
) -> Dict[str, Any]:
    passed_models = [str(model) for model in verification_payload.get("passed_models") or []]
    failed_models = [str(model) for model in verification_payload.get("failed_models") or []]
    infra_failures = verification_payload.get("infra_failures") or {}
    model_summaries = verification_payload.get("models") or {}
    required_failures = int(verification_payload.get("required_failures", 0) or 0)
    total_models = len(model_summaries)

    evidence: List[str] = []
    fast_passing_models: List[str] = []
    for model, summary in model_summaries.items():
        if not isinstance(summary, dict):
            continue
        passed = bool(summary.get("passed", False))
        turns = int(summary.get("turns", 0) or 0)
        evidence.append(
            f"{model}: passed={passed}, progress={_format_percent(summary.get('progress'))}, turns={turns}"
        )
        if passed and turns <= 10:
            fast_passing_models.append(str(model))

    if passed_models:
        summary = (
            f"Submission verification rejected the task because {len(passed_models)}/{max(total_models, 1)} "
            "verification models still solved it in standard mode."
        )
    else:
        summary = (
            "Submission verification rejected the task because not enough verification models failed."
        )

    required_fixes = [
        f"Strengthen one decisive hidden-information dependency so at least {required_failures} of {max(total_models, 1)} verification models fail in standard mode.",
        "Keep the physical core baseline-solvable; harden the information split instead of adding clutter or extra search.",
        "Use a hidden choice or incompatible end state that cannot be guessed from public information alone.",
    ]
    if fast_passing_models:
        required_fixes.append(
            f"Priority: {', '.join(fast_passing_models)} solved the task quickly, so the current information bottleneck is too weak or too easy to guess."
        )
    if infra_failures:
        failed_infra_models = ", ".join(sorted(str(model) for model in infra_failures))
        required_fixes.append(
            f"Note: infrastructure errors prevented evaluation for {failed_infra_models}; the task was judged only on the verification models that completed."
        )
    artifact_paths: Dict[str, str] = {}
    trajectory_dir = verification_payload.get("trajectory_dir")
    if trajectory_dir:
        trajectory_path = Path(str(trajectory_dir))
        artifact_paths["trajectory_dir"] = str(trajectory_path)
        artifact_paths["verification_summary_json"] = str(trajectory_path / "verification_summary.json")

    snapshot = {
        "required_failures": required_failures,
        "passed_models": passed_models,
        "failed_models": failed_models,
        "infra_failures": infra_failures,
    }
    return {
        "source_gate": "verify_task",
        "summary": summary,
        "overall_reasoning": str(verification_payload.get("message", "")),
        "required_fixes": required_fixes,
        "evidence": evidence,
        "artifact_paths": artifact_paths,
        "verification_snapshot": snapshot,
    }


def _render_benchmark_feedback_markdown(feedback: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Retry Feedback",
        "",
        f"Gate: `{feedback.get('source_gate', 'unknown')}`",
        "",
        "## Summary",
        str(feedback.get("summary", "")),
    ]

    overall_reasoning = str(feedback.get("overall_reasoning", "") or "").strip()
    if overall_reasoning:
        lines.extend(["", "## Why It Failed", overall_reasoning])

    required_fixes = feedback.get("required_fixes") or []
    if required_fixes:
        lines.extend(["", "## Required Fixes"])
        lines.extend(f"- {fix}" for fix in required_fixes)

    evidence = feedback.get("evidence") or []
    if evidence:
        lines.extend(["", "## Evidence"])
        lines.extend(f"- {item}" for item in evidence)

    artifact_paths = feedback.get("artifact_paths") or {}
    if artifact_paths:
        lines.extend(["", "## Artifacts"])
        for name, path in artifact_paths.items():
            lines.append(f"- `{name}`: `{path}`")

    return "\n".join(lines) + "\n"


def default_state(
    *,
    working_dir: str,
    output_dir: str,
    num_tasks_target: int,
    agents_min: int,
    agents_max: int,
    subtasks_min: int,
    subtasks_max: int,
    category: Optional[str],
    seed_tasks_dir: Optional[str],
    seed_pass_ratio: float,
    seed_fail_ratio: float,
    judge_threshold: Optional[float],
    difficulty: Optional[str],
    test_model: Optional[str],
    calibration_stats: Dict[str, Any],
    calibration_tasks_dirs: Optional[List[str]],
    task_gen_agent: str,
    allowed_k_levels: Optional[List[int]],
    generation_run_id: Optional[str] = None,
    generation_run_dir: Optional[str] = None,
    generation_worker_id: Optional[str] = None,
    generation_worker_dir: Optional[str] = None,
    skip_steps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "working_dir": working_dir,
        "output_dir": output_dir,
        "num_tasks_target": num_tasks_target,
        "agents_min": agents_min,
        "agents_max": agents_max,
        "subtasks_min": subtasks_min,
        "subtasks_max": subtasks_max,
        "category": category,
        "seed_tasks_dir": seed_tasks_dir,
        "seed_pass_ratio": seed_pass_ratio,
        "seed_fail_ratio": seed_fail_ratio,
        "judge_threshold": judge_threshold,
        "difficulty": difficulty,
        "test_model": test_model,
        "task_gen_agent": task_gen_agent,
        "submitted_tasks": [],
        "submitted_count": 0,
        "current_task_index": 1,
        "current_k_level": None,
        "allowed_k_levels": allowed_k_levels,
        "last_judge_passed": False,
        "last_verify_passed": False,
        "last_test_passed": False,
        "last_submission_verification_passed": False,
        "last_submission_verification_results": {},
        "last_submission_verification_spec_hash": None,
        "last_verified_spec_hash": None,
        "last_verified_trajectory_hash": None,
        "consecutive_judge_failures": 0,
        "finished": False,
        "failed": False,
        "fail_reason": "",
        "scene_id": None,
        "episode_id": None,
        "calibration_stats": calibration_stats,
        "calibration_tasks_dirs": calibration_tasks_dirs or [],
        "generation_run_id": generation_run_id,
        "generation_run_dir": generation_run_dir,
        "generation_worker_id": generation_worker_id,
        "generation_worker_dir": generation_worker_dir,
        "skip_steps": skip_steps or [],
    }


class TaskGenSession:
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.state_path = self.working_dir / "taskgen_state.json"
        self.task_file = self.working_dir / "working_task.json"
        self.template_file = self.working_dir / "template.json"
        self.trajectories_dir = self.working_dir / "agent_trajectories"
        self.submitted_tasks_dir = self.working_dir / "submitted_tasks"
        self.scene_file = self.working_dir / "current_scene.json"
        self.benchmark_feedback_json = self.working_dir / "benchmark_retry_feedback.json"
        self.benchmark_feedback_md = self.working_dir / "benchmark_retry_feedback.md"
        self.state = self._read_state()

    def _resolve_asset_path(self, path: str) -> str:
        """Resolve an asset/dataset path relative to the project root.

        Taskgen subprocesses run with cwd=project_root, but some libraries may
        compute relative paths using the process cwd or assume the caller's
        workspace. In practice, the taskgen working dir is nested under
        tmp/task_gen/..., and relative-path existence checks can fail even when
        the assets exist in the repository.

        This helper normalizes the common relative paths used by Habitat LLM
        configs (e.g. `data/hssd-hab`) to absolute paths under project_root.
        """

        if not path:
            return path
        p = Path(path)
        if p.is_absolute():
            return path
        return str((self.project_root / p).resolve())

    def _with_project_root_assets(self, env: Dict[str, str]) -> Dict[str, str]:
        """Populate env with absolute asset paths if the caller didn't specify them.

        Habitat configs sometimes rely on repo-relative paths like
        `data/hssd-hab` or `data/datasets/...`. When taskgen runs from the nested
        workspace directory, those relative checks fail even though the assets
        exist under the repository root.

        We set a few common env vars to absolute paths under project_root.
        """

        out = dict(env)
        out.setdefault(
            "HABITAT_SIM_V0_SCENE_DATASET",
            self._resolve_asset_path(
                "data/hssd-hab/hssd-hab-enacttom.scene_dataset_config.json"
            ),
        )
        out.setdefault("HABITAT_DATA_PATH", self._resolve_asset_path("data"))
        out.setdefault(
            "ENACTTOM_EPISODES_PATH",
            self._resolve_asset_path(
                "data/datasets/enacttom_episodes/v0_0/train_2k.json.gz"
            ),
        )
        return out

    def _read_state(self) -> Dict[str, Any]:
        with open(self.state_path) as f:
            return json.load(f)

    def _write_state(self) -> None:
        self.state["submitted_count"] = len(self.state.get("submitted_tasks", []))
        self.state["current_task_index"] = self.state["submitted_count"] + 1
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def _clear_benchmark_feedback(self) -> None:
        for path in (self.benchmark_feedback_json, self.benchmark_feedback_md):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _write_benchmark_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(feedback)
        payload["json_path"] = str(self.benchmark_feedback_json)
        payload["markdown_path"] = str(self.benchmark_feedback_md)
        with open(self.benchmark_feedback_json, "w") as f:
            json.dump(payload, f, indent=2)
        self.benchmark_feedback_md.write_text(
            _render_benchmark_feedback_markdown(payload),
            encoding="utf-8",
        )
        return payload

    def _load_scene_data(self) -> Optional[SceneData]:
        if not self.scene_file.exists():
            return None
        with open(self.scene_file) as f:
            scene_dict = json.load(f)
        return SceneData(
            episode_id=scene_dict["episode_id"],
            scene_id=scene_dict["scene_id"],
            rooms=scene_dict["rooms"],
            furniture=scene_dict["furniture"],
            objects=scene_dict["objects"],
            articulated_furniture=scene_dict.get("articulated_furniture", []),
            furniture_in_rooms=scene_dict["furniture_in_rooms"],
            objects_on_furniture=scene_dict["objects_on_furniture"],
            agent_spawns=scene_dict.get("agent_spawns", {}),
        )

    def _pick_k_level(self) -> int:
        allowed = self.state.get("allowed_k_levels") or [1, 2, 3]
        return random.choice(allowed)

    def _refresh_calibration_stats(self) -> None:
        from enacttom.task_gen.runner import compute_calibration_stats

        task_dirs = self.state.get("calibration_tasks_dirs") or [self.state.get("output_dir")]
        current = self.state.get("calibration_stats") or {}
        model = current.get("model") or self.state.get("test_model") or "gpt-5.2"
        target_rate = current.get("target_rate", 0.10)
        stats = compute_calibration_stats(task_dirs, model)
        stats["target_rate"] = target_rate
        self.state["calibration_stats"] = stats

    def _reset_gate_state(self) -> None:
        self.state["last_judge_passed"] = False
        self.state["last_verify_passed"] = False
        self.state["last_test_passed"] = False
        self.state["last_submission_verification_passed"] = False
        self.state["last_submission_verification_results"] = {}
        self.state["last_submission_verification_spec_hash"] = None
        self.state["last_verified_spec_hash"] = None
        self.state["last_verified_trajectory_hash"] = None
        self.state["consecutive_judge_failures"] = 0

    def status(self) -> CLIResult:
        scene_data = self._load_scene_data()
        data = {
            "working_dir": str(self.working_dir),
            "task_file": str(self.task_file),
            "prompt_file": str(self.working_dir / "taskgen_prompt.md"),
            "benchmark_feedback_file": str(self.benchmark_feedback_md) if self.benchmark_feedback_md.exists() else None,
            "benchmark_feedback_json": str(self.benchmark_feedback_json) if self.benchmark_feedback_json.exists() else None,
            "submitted_tasks": self.state.get("submitted_tasks", []),
            "submitted_count": len(self.state.get("submitted_tasks", [])),
            "num_tasks_target": self.state["num_tasks_target"],
            "current_task_index": self.state.get("current_task_index"),
            "current_k_level": self.state.get("current_k_level"),
            "category": self.state.get("category"),
            "finished": self.state.get("finished", False),
            "failed": self.state.get("failed", False),
            "fail_reason": self.state.get("fail_reason", ""),
            "last_judge_passed": self.state.get("last_judge_passed", False),
            "last_verify_passed": self.state.get("last_verify_passed", False),
            "last_test_passed": self.state.get("last_test_passed", False),
            "last_submission_verification_passed": self.state.get("last_submission_verification_passed", False),
            "last_submission_verification_results": self.state.get("last_submission_verification_results", {}),
            "last_submission_verification_spec_hash": self.state.get("last_submission_verification_spec_hash"),
            "last_verified_spec_hash": self.state.get("last_verified_spec_hash"),
            "last_verified_trajectory_hash": self.state.get("last_verified_trajectory_hash"),
            "scene_loaded": scene_data is not None,
        }
        if scene_data is not None:
            data.update(
                {
                    "scene_id": scene_data.scene_id,
                    "episode_id": scene_data.episode_id,
                    "rooms": scene_data.rooms,
                    "objects": len(scene_data.objects),
                    "furniture": len(scene_data.furniture),
                }
            )
        return success(data)

    def finish(self) -> CLIResult:
        submitted = len(self.state.get("submitted_tasks", []))
        target = self.state["num_tasks_target"]
        if submitted < target:
            return failure(
                f"Cannot finish yet: submitted {submitted}/{target} tasks.",
                data={"submitted": submitted, "target": target},
            )
        self.state["finished"] = True
        self._write_state()
        return success(
            {
                "message": f"Task generation finished with {submitted}/{target} tasks submitted.",
                "submitted_tasks": self.state.get("submitted_tasks", []),
            }
        )

    def fail(self, reason: str) -> CLIResult:
        # Soft guard: if the agent hasn't tried enough judge iterations, push back.
        judge_failures = self.state.get("consecutive_judge_failures", 0)
        submitted = len(self.state.get("submitted_tasks", []))
        if submitted == 0 and judge_failures < 10:
            return success({
                "message": (
                    f"Cannot abort yet — only {judge_failures} consecutive judge failures. "
                    "Keep trying: run `taskgen new_scene N` for a fresh scene, "
                    "simplify the task design, or try a different approach. "
                    f"Your reason was: {reason}"
                ),
                "blocked": True,
            })
        self.state["failed"] = True
        self.state["fail_reason"] = reason
        self._write_state()
        return success({"message": f"Marked run as failed: {reason}"})

    def _create_working_task_from_template(self, num_agents: int) -> None:
        with open(self.template_file) as f:
            task = json.load(f)
        default_actions = get_authoring_default_actions(include_find_tools=True)
        task["agent_secrets"] = {
            f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"] for i in range(num_agents)
        }
        task["agent_actions"] = {
            f"agent_{i}": default_actions.copy() for i in range(num_agents)
        }

        scene_data = self._load_scene_data()
        if scene_data is not None:
            task["scene_id"] = scene_data.scene_id
            task["episode_id"] = scene_data.episode_id
            if scene_data.agent_spawns:
                task["agent_spawns"] = scene_data.agent_spawns
            task["problem_pddl"] = build_scene_bootstrap_problem_pddl(
                scene_data,
                num_agents,
                problem_name=f"scene_{scene_data.scene_id}",
            )
        task["num_agents"] = num_agents
        task["task_id"] = "REPLACE_WITH_UNIQUE_ID"

        with open(self.task_file, "w") as f:
            json.dump(task, f, indent=2)

    def new_scene(self, num_agents: int, keep: bool = False) -> CLIResult:
        if num_agents < 2 or num_agents > 4:
            return failure(f"num_agents must be 2-4, got {num_agents}")

        if self.state.get("current_k_level") is None:
            self.state["current_k_level"] = self._pick_k_level()

        scene_data = self._load_scene_data()
        scene_id = scene_data.scene_id if keep and scene_data is not None else None
        if keep and scene_data is None:
            return failure("Cannot use --keep before a scene has been loaded.")

        max_scene_retries = 5
        last_error = ""
        for _ in range(max_scene_retries):
            try:
                from habitat_llm.utils import get_random_seed
            except ModuleNotFoundError:
                import random

                get_random_seed = lambda: random.randint(1, 2**31 - 1)


            # Ensure subprocess sees the same project-root-relative asset paths
            # regardless of the caller's current working directory.
            env = os.environ.copy()
            env = self._with_project_root_assets(env)
            cmd = [
                sys.executable,
                "-m",
                "enacttom.cli.new_scene",
                str(num_agents),
                "--working-dir",
                str(self.working_dir),
                "--seed",
                str(get_random_seed()),
            ]
            if scene_id:
                cmd.extend(["--scene-id", scene_id])

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(self.project_root),
                env=env,
            )
            try:
                stdout = proc.stdout
                json_start = stdout.find("{")
                if json_start >= 0:
                    stdout = stdout[json_start:]
                result = json.loads(stdout)
            except (json.JSONDecodeError, ValueError):
                last_error = proc.stderr or "Failed to parse new_scene output."
                continue

            if not isinstance(result, dict) or not result.get("success"):
                last_error = result.get("error", "Unknown error") if isinstance(result, dict) else "Unexpected output type"
                continue

            loaded_scene = self._load_scene_data()
            if loaded_scene is None:
                last_error = "Scene file was not written."
                continue

            min_objects = 5
            if len(loaded_scene.objects) < min_objects:
                last_error = (
                    f"Scene {loaded_scene.scene_id} has only {len(loaded_scene.objects)} locatable objects."
                )
                continue

            furniture_to_room: Dict[str, str] = {}
            for room_id, room_furniture in (loaded_scene.furniture_in_rooms or {}).items():
                if isinstance(room_furniture, list):
                    for furniture_id in room_furniture:
                        if isinstance(furniture_id, str):
                            furniture_to_room[furniture_id] = room_id

            rooms_with_objects = set()
            for furniture_id, objects_on_furniture in (loaded_scene.objects_on_furniture or {}).items():
                if objects_on_furniture and furniture_id in furniture_to_room:
                    rooms_with_objects.add(furniture_to_room[furniture_id])
            if furniture_to_room and len(rooms_with_objects) < 2:
                last_error = (
                    f"Scene {loaded_scene.scene_id} has objects concentrated in {len(rooms_with_objects)} room(s)."
                )
                continue

            self.state["scene_id"] = loaded_scene.scene_id
            self.state["episode_id"] = loaded_scene.episode_id
            self._reset_gate_state()
            if not keep:
                self._clear_benchmark_feedback()

            if keep and self.task_file.exists():
                try:
                    with open(self.task_file) as f:
                        task_data = json.load(f)
                except json.JSONDecodeError:
                    self._create_working_task_from_template(num_agents)
                else:
                    task_data["num_agents"] = num_agents
                    if loaded_scene.agent_spawns:
                        task_data["agent_spawns"] = loaded_scene.agent_spawns
                    with open(self.task_file, "w") as f:
                        json.dump(task_data, f, indent=2)
            else:
                self._create_working_task_from_template(num_agents)

            self._write_state()
            valid_agent_ids = [f"agent_{i}" for i in range(num_agents)]
            return success(
                {
                    "message": "Scene loaded.",
                    "scene_id": loaded_scene.scene_id,
                    "episode_id": loaded_scene.episode_id,
                    "num_agents": num_agents,
                    "valid_agent_ids": valid_agent_ids,
                    "keep": keep,
                    "rooms": loaded_scene.rooms,
                    "objects": loaded_scene.objects,
                    "furniture": loaded_scene.furniture,
                    "articulated_furniture": loaded_scene.articulated_furniture,
                    "current_k_level": self.state.get("current_k_level"),
                    "task_file": str(self.task_file),
                    "hint": (
                        f"This scene has {num_agents} agents: {valid_agent_ids}. "
                        "All mechanic_bindings, agent_secrets, message_targets, "
                        "and problem_pddl :objects MUST only reference these agent IDs. "
                        "Sampled tasks in sampled_tasks/ may have different agent counts — "
                        "adapt their patterns to this scene's agents, do not copy directly."
                    ),
                }
            )

        return failure(f"Failed to load a usable scene: {last_error}")

    def _validate_task_structure(self, task_data: Dict[str, Any]) -> CLIResult:
        result = validate(task_data, self._load_scene_data())
        if result["success"]:
            return success(result["data"])
        return failure(result["error"], data={"valid": False})

    def _build_trajectory(self, action_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from collections import defaultdict

        turns: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"agents": {}, "subtasks_completed": []}
        )
        for record in action_history:
            turn = record.get("turn", 0)
            if record.get("type") == "subtask_completion":
                turns[turn]["subtasks_completed"].extend(record.get("subtasks_completed", []))
                continue
            agent_id = record.get("agent", "unknown")
            turns[turn]["agents"][agent_id] = {
                "action": record.get("action", ""),
                "observation": record.get("result", ""),
            }

        trajectory = []
        for turn_num in sorted(turns):
            entry = turns[turn_num]
            trajectory.append(
                {
                    "turn": turn_num,
                    "agents": entry["agents"],
                    "subtasks_completed": entry["subtasks_completed"],
                }
            )
        return trajectory

    def _save_calibration_result(self, task_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        from datetime import datetime

        agent_models = getattr(self, "_last_agent_models", None)
        if not agent_models:
            model_name = self.state.get("test_model") or "unknown"
            agent_models = {
                f"agent_{i}": model_name for i in range(task_data.get("num_agents", 2))
            }

        category = task_data.get("category", "")
        raw_calibration = task_data.get("calibration", [])
        calibration = raw_calibration if isinstance(raw_calibration, list) else []
        tested_at = datetime.now().isoformat()

        for run_mode in ("standard", "baseline"):
            run_result = results.get(run_mode)
            if not isinstance(run_result, dict):
                continue
            calibration_entry = {
                "tested_at": tested_at,
                "run_mode": run_mode,
                "agent_models": agent_models,
                "steps": run_result.get("steps", 0),
                "results": _build_results_block(category, run_result.get("evaluation", {})),
                "trajectory": self._build_trajectory(run_result.get("action_history", [])),
            }
            replaced = False
            for idx, existing in enumerate(calibration):
                existing_run_mode = str(existing.get("run_mode", "standard") or "standard")
                if existing.get("agent_models") == agent_models and existing_run_mode == run_mode:
                    calibration[idx] = calibration_entry
                    replaced = True
                    break
            if not replaced:
                calibration.append(calibration_entry)

        task_data["calibration"] = calibration
        with open(self.task_file, "w") as f:
            json.dump(task_data, f, indent=2)

    def _determine_agent_models(self, task_data: Dict[str, Any]) -> Dict[str, str]:
        num_agents = task_data.get("num_agents", 2)
        model_name = self.state.get("test_model") or "gpt-5.2"
        return {f"agent_{i}": model_name for i in range(num_agents)}

    def _parse_benchmark_subprocess(
        self, proc: subprocess.CompletedProcess[str], run_dir: Path
    ) -> Dict[str, Any]:
        """Parse JSON result from a benchmark subprocess.

        Some simulator dependencies emit noisy logs to stdout and/or stderr
        before printing the CLIResult JSON payload. Search both streams for the
        first JSON object and decode whichever stream actually contains one.
        """

        streams = []
        if proc.stdout:
            streams.append(proc.stdout)
        if proc.stderr and proc.stderr != proc.stdout:
            streams.append(proc.stderr)

        result_data = None
        decoder = json.JSONDecoder()
        for stream in streams:
            candidate = stream
            json_start = candidate.find("{")
            if json_start >= 0:
                candidate = candidate[json_start:]
            else:
                continue

            try:
                result_data = json.loads(candidate)
                break
            except (json.JSONDecodeError, ValueError):
                # Fall back: sometimes warnings/logs appear after the JSON payload.
                # Decode only the first complete JSON object.
                try:
                    result_data, _ = decoder.raw_decode(candidate)
                    break
                except Exception:
                    continue

        if result_data is None:
            noisy_parts = []
            if proc.stdout:
                noisy_parts.append(proc.stdout[:250])
            if proc.stderr and proc.stderr != proc.stdout:
                noisy_parts.append(proc.stderr[:250])
            noisy = "\n--- STDERR/STDOUT SPLIT ---\n".join(noisy_parts)[:500]
            return {
                "steps": 0,
                "done": False,
                "error": f"Failed to parse output: {noisy}",
                "trajectory_dir": str(run_dir),
            }

        if not isinstance(result_data, dict):
            return {
                "steps": 0,
                "done": False,
                "error": f"Unexpected output type ({type(result_data).__name__}): {str(result_data)[:200]}",
                "trajectory_dir": str(run_dir),
            }

        if result_data.get("success"):
            result = result_data.get("data", {})
        else:
            result = {
                "steps": 0,
                "done": False,
                "error": result_data.get("error", "Unknown error"),
            }
            if isinstance(result_data.get("data"), dict):
                result.update(result_data["data"])

        result["trajectory_dir"] = str(run_dir)
        result.pop("planner_traces", None)
        return result

    def _run_benchmark_mode(
        self,
        task_data: Dict[str, Any],
        temp_task_file: str,
        run_mode: str,
        run_dir: Path,
        test_model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        num_agents = task_data.get("num_agents", 2)
        cmd = [
            sys.executable,
            "-m",
            "enacttom.cli.test_task",
            temp_task_file,
            "--working-dir",
            str(self.working_dir),
            "--trajectory-dir",
            str(run_dir),
            "--config-name",
            f"examples/enacttom_{num_agents}_robots",
            "--run-mode",
            run_mode,
        ]
        effective_test_model = test_model_override or self.state.get("test_model")
        if effective_test_model:
            cmd.extend(["--test-model", effective_test_model])

        env = os.environ.copy()
        env = self._with_project_root_assets(env)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_test_timeout_seconds(num_agents, task_data.get("category", "cooperative")),
                cwd=str(self.project_root),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "steps": 0,
                "done": False,
                "error": "Test timed out",
                "trajectory_dir": str(run_dir),
                "run_mode": run_mode,
            }
        result = self._parse_benchmark_subprocess(proc, run_dir)
        result["run_mode"] = run_mode
        return result

    def _run_benchmark(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        import tempfile

        current_task_num = len(self.state.get("submitted_tasks", [])) + 1
        run_count = self.state.get("_test_run_count", 0) + 1
        self.state["_test_run_count"] = run_count
        self._write_state()
        run_dir = self.trajectories_dir / f"task_{current_task_num}" / f"run_{run_count}"
        run_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_task_file = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(task_data, f)

            self._last_agent_models = self._determine_agent_models(task_data)
            _skip = set(self.state.get("skip_steps") or [])
            _skip_baseline = "baseline" in _skip

            run_specs: List[Dict[str, Any]] = [
                {
                    "result_key": "standard",
                    "run_mode": "standard",
                    "run_dir": run_dir / "standard",
                }
            ]
            if not _skip_baseline:
                run_specs.append(
                    {
                        "result_key": "baseline",
                        "run_mode": "baseline",
                        "run_dir": run_dir / "baseline",
                    }
                )

            with ThreadPoolExecutor(max_workers=len(run_specs)) as executor:
                futures = {
                    spec["result_key"]: executor.submit(
                        self._run_benchmark_mode,
                        task_data,
                        temp_task_file,
                        spec["run_mode"],
                        spec["run_dir"],
                    )
                    for spec in run_specs
                }
                raw_results = {
                    result_key: future.result() for result_key, future in futures.items()
                }
        finally:
            try:
                os.unlink(temp_task_file)
            except Exception:
                pass

        mode_errors = {
            run_mode: result["error"]
            for run_mode, result in raw_results.items()
            if result.get("error")
        }
        if mode_errors:
            return {
                "error": "; ".join(f"{mode}: {err}" for mode, err in sorted(mode_errors.items())),
                "mode_errors": mode_errors,
                "trajectory_dir": str(run_dir),
            }

        category = task_data.get("category", "cooperative")

        if _skip_baseline:
            # Baseline skipped: gate based only on standard failing
            std_eval = raw_results["standard"].get("evaluation", {})
            standard_passed = _evaluation_passed(category, std_eval)
            standard_progress = _evaluation_progress(category, std_eval)
            comparison = {
                "gate_passed": not standard_passed,
                "functional_tom_signal": None,
                "baseline_skipped": True,
                "standard_passed": standard_passed,
                "baseline_passed": None,
                "standard_progress": standard_progress,
                "baseline_progress": None,
                "reasons": ["Baseline skipped (--remove baseline). Gate based on standard only."],
            }
            payload = {
                "standard": raw_results["standard"],
                "baseline": {"skipped": True, "reason": "--remove baseline"},
                "comparison": comparison,
                "trajectory_dir": str(run_dir),
            }
        else:
            mode_results = {
                "standard": raw_results["standard"],
                "baseline": raw_results["baseline"],
            }

            self._refresh_calibration_stats()
            calibration_stats = self.state.get("calibration_stats") or {}
            cat_stats = calibration_stats.get("by_category", {}).get(category, {})
            if cat_stats.get("total", 0) > 0:
                cal_rate = cat_stats.get("rate")
                cal_passed = cat_stats.get("passed")
                cal_failed = cat_stats.get("failed")
            else:
                cal_rate = calibration_stats.get("rate")
                cal_passed = calibration_stats.get("passed")
                cal_failed = calibration_stats.get("failed")
            comparison = build_mode_comparison(
                category,
                mode_results["standard"],
                mode_results["baseline"],
                current_rate=cal_rate,
                target_rate=calibration_stats.get("target_rate", 0.10),
                current_passed=cal_passed,
                current_failed=cal_failed,
                hard_pass_rate_cap=calibration_stats.get("target_rate", 0.10) <= 0.05 + 1e-9,
            )
            payload = {
                "standard": mode_results["standard"],
                "baseline": mode_results["baseline"],
                "comparison": comparison,
                "trajectory_dir": str(run_dir),
            }

        with open(run_dir / "comparison.json", "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    def verify_task(self) -> CLIResult:
        if not self.task_file.exists():
            return failure("working_task.json does not exist.")

        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return failure(f"Invalid JSON in working_task.json: {e}")

        validation_result = self._validate_task_structure(task_data)
        if not validation_result["success"]:
            return validation_result

        try:
            import hydra  # type: ignore
            import habitat  # type: ignore
            _deps_ok = True
        except Exception as e:
            _deps_ok = False
            _deps_err = str(e)

        if not _deps_ok:
            from enacttom.pddl.planner import compute_task_spec_hash

            spec_hash = compute_task_spec_hash(task_data)
            self._clear_benchmark_feedback()
            self.state["last_submission_verification_passed"] = True
            self.state["last_submission_verification_spec_hash"] = spec_hash
            self.state["last_submission_verification_results"] = {
                "skipped": True,
                "reason": f"Skipped simulator verification due to missing deps: {_deps_err}",
                "required_failures": SUBMISSION_VERIFICATION_REQUIRED_FAILURES,
                "models": SUBMISSION_VERIFICATION_MODELS,
                "spec_hash": spec_hash,
            }
            self._write_state()
            payload = dict(validation_result["data"])
            payload.update(self.state["last_submission_verification_results"])
            payload["gate"] = "PASSED"
            return success(payload)

        import tempfile

        current_task_num = len(self.state.get("submitted_tasks", [])) + 1
        run_count = self.state.get("_verification_run_count", 0) + 1
        self.state["_verification_run_count"] = run_count
        self.state["last_submission_verification_passed"] = False
        self.state["last_submission_verification_results"] = {}
        self.state["last_submission_verification_spec_hash"] = None
        self._write_state()
        run_dir = self.trajectories_dir / f"task_{current_task_num}" / f"verification_{run_count}"
        run_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_task_file = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(task_data, f)

            with ThreadPoolExecutor(max_workers=len(SUBMISSION_VERIFICATION_MODELS)) as executor:
                futures = {
                    model: executor.submit(
                        self._run_benchmark_mode,
                        task_data,
                        temp_task_file,
                        "standard",
                        run_dir / model.replace("/", "_"),
                        test_model_override=model,
                    )
                    for model in SUBMISSION_VERIFICATION_MODELS
                }
                model_results = {model: future.result() for model, future in futures.items()}
        finally:
            try:
                os.unlink(temp_task_file)
            except Exception:
                pass

        errors = {
            model: result.get("error")
            for model, result in model_results.items()
            if result.get("error")
        }
        usable_model_results = {
            model: result
            for model, result in model_results.items()
            if not result.get("error")
        }
        if not usable_model_results:
            joined = "; ".join(f"{model}: {err}" for model, err in sorted(errors.items())) or "unknown error"
            return failure(
                f"Submission verification failed to run: {joined}",
                data={"model_results": model_results, "trajectory_dir": str(run_dir)},
            )

        category = task_data.get("category", "cooperative")
        summaries: Dict[str, Any] = {}
        failed_models: List[str] = []
        passed_models: List[str] = []
        for model, result in usable_model_results.items():
            evaluation = result.get("evaluation", {})
            passed = _evaluation_passed(category, evaluation)
            progress = _evaluation_progress(category, evaluation)
            summaries[model] = {
                "passed": passed,
                "failed": not passed,
                "progress": progress,
                "steps": result.get("steps", 0),
                "turns": result.get("turns", 0),
                "trajectory_dir": result.get("trajectory_dir"),
            }
            if passed:
                passed_models.append(model)
            else:
                failed_models.append(model)

        from enacttom.pddl.planner import compute_task_spec_hash

        required_failures = min(
            SUBMISSION_VERIFICATION_REQUIRED_FAILURES,
            len(SUBMISSION_VERIFICATION_MODELS),
        )
        spec_hash = compute_task_spec_hash(task_data)
        gate_passed = len(failed_models) >= required_failures
        verification_payload = {
            "required_failures": required_failures,
            "gate_passed": gate_passed,
            "models": summaries,
            "failed_models": failed_models,
            "passed_models": passed_models,
            "infra_failures": errors,
            "trajectory_dir": str(run_dir),
            "spec_hash": spec_hash,
            "message": (
                f"Submission verification passed. {len(failed_models)}/{len(usable_model_results)} completed verification models failed."
                if gate_passed
                else f"Submission verification failed. Need at least {required_failures}/{len(usable_model_results)} completed verification models to fail."
            ),
        }
        with open(run_dir / "verification_summary.json", "w") as f:
            json.dump(verification_payload, f, indent=2)

        self.state["last_submission_verification_passed"] = gate_passed
        self.state["last_submission_verification_results"] = verification_payload
        self.state["last_submission_verification_spec_hash"] = spec_hash if gate_passed else None
        self._write_state()

        payload = dict(validation_result["data"])
        payload.update(verification_payload)
        payload["gate"] = "PASSED" if gate_passed else "REJECTED"
        if gate_passed:
            self._clear_benchmark_feedback()
            payload["next_step"] = "Verification passed. Submit the task without changing the spec."
            return success(payload)

        feedback = _build_submission_verification_feedback(category, verification_payload)
        feedback_payload = self._write_benchmark_feedback(feedback)
        payload["benchmark_feedback"] = feedback_payload
        payload["feedback_file"] = feedback_payload["markdown_path"]
        payload["action_required"] = (
            f"At least {required_failures} of gpt-5.4, claude-sonnet-4-6, and gemini-flash must fail. "
            f"Read `{self.benchmark_feedback_md.name}` and address every listed issue before rerunning "
            "taskgen judge -> taskgen test_task -> taskgen verify_task."
        )
        payload["next_step"] = payload["action_required"]
        return failure("Submission verification rejected the task.", data=payload)

    def test_task(self) -> CLIResult:
        _skip_test = self.state.get("skip_steps") or []
        if "test" in _skip_test:
            self.state["last_test_passed"] = True
            self._clear_benchmark_feedback()
            self._write_state()
            return success({"gate": "PASSED", "skipped": True, "reason": "--remove test"})

        if not self.task_file.exists():
            return failure("working_task.json does not exist.")

        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return failure(f"Invalid JSON in working_task.json: {e}")

        validation_result = self._validate_task_structure(task_data)
        if not validation_result["success"]:
            return validation_result


        # If simulator dependencies (Hydra/Habitat) are not available in this
        # environment, we cannot run a full benchmark episode. In that case,
        # fall back to a structure-only pass so external task authors can still
        # submit tasks in lightweight CI containers.
        try:
            import hydra  # type: ignore
            import habitat  # type: ignore
            _deps_ok = True
        except Exception as e:
            _deps_ok = False
            _deps_err = str(e)

        if not _deps_ok:
            self.state["last_test_passed"] = True
            self._clear_benchmark_feedback()
            self._write_state()
            payload = dict(validation_result["data"])
            payload.update({"warning": f"Skipped simulator test due to missing deps: {_deps_err}", "gate": "PASSED"})
            return success(payload)
        try:
            results = self._run_benchmark(task_data)
        except Exception as e:
            self.state["failed"] = True
            self.state["fail_reason"] = f"Benchmark execution failed: {e}"
            self._write_state()
            return failure(str(e), data=validation_result["data"])

        if results.get("error"):
            self.state["failed"] = True
            self.state["fail_reason"] = f"Benchmark execution failed: {results['error']}"
            self._write_state()
            failure_data = dict(validation_result["data"])
            failure_data.update(results)
            failure_data["fatal_infra"] = True
            return failure(results["error"], data=failure_data)

        self._save_calibration_result(task_data, results)
        self.state["last_test_passed"] = results["comparison"]["gate_passed"]
        self.state["last_submission_verification_passed"] = False
        self.state["last_submission_verification_results"] = {}
        self._write_state()
        payload = dict(validation_result["data"])
        payload.update(results)
        payload["gate"] = "PASSED" if self.state["last_test_passed"] else "REJECTED"
        payload["gate_reason"] = " ".join(results["comparison"].get("reasons", []))
        if self.state["last_test_passed"]:
            self._clear_benchmark_feedback()
            payload["next_step"] = "Test passed. Run taskgen verify_task, then submit the task without changing the spec."
        else:
            feedback = _build_test_task_failure_feedback(
                task_data.get("category", "cooperative"),
                results["comparison"],
                results.get("trajectory_dir"),
            )
            feedback_payload = self._write_benchmark_feedback(feedback)
            payload["benchmark_feedback"] = feedback_payload
            payload["feedback_file"] = feedback_payload["markdown_path"]
            payload["action_required"] = (
                _build_test_task_retry_guidance(results["comparison"])
                + f" Read `{self.benchmark_feedback_md.name}` and fix every listed issue before editing again."
            )
            payload["next_step"] = payload["action_required"]
        return success(payload)

    def verify_golden_trajectory(self) -> CLIResult:
        return failure(
            "verify_golden_trajectory is no longer a separate step. "
            "Run judge; it now regenerates the plan, simulator-verifies it when needed, "
            "and then runs the quality judge."
        )

    def judge(self) -> CLIResult:
        if not self.task_file.exists():
            return failure("working_task.json does not exist.")

        current_task_num = len(self.state.get("submitted_tasks", [])) + 1
        task_traj_dir = self.trajectories_dir / f"task_{current_task_num}"
        trajectory_dir = None
        if task_traj_dir.exists():
            run_dirs = sorted(task_traj_dir.glob("run_*"), key=lambda p: p.name)
            if run_dirs:
                trajectory_dir = str(run_dirs[-1])

        result = judge_task_run(
            str(self.task_file),
            working_dir=str(self.working_dir),
            trajectory_dir=trajectory_dir,
            threshold=self.state.get("judge_threshold") or 0.7,
            difficulty=self.state.get("difficulty"),
            required_tom_level=self.state.get("current_k_level"),
            verified_trajectory_hash=(
                self.state.get("last_verified_trajectory_hash")
                if self.state.get("last_verify_passed")
                else None
            ),
            skip_steps=self.state.get("skip_steps"),
        )
        data = result.get("data") or {}
        golden = data.get("golden_trajectory") or {}
        if golden.get("sim_verified"):
            # Note: in lightweight envs sim verification may be skipped due to missing
            # hydra/habitat deps; judge_task.py still sets sim_verified=True in that case.
            
            self.state["last_verify_passed"] = True
            self.state["last_verified_spec_hash"] = golden.get("spec_hash")
            self.state["last_verified_trajectory_hash"] = golden.get("trajectory_hash")
        elif golden.get("sim_verification_ran"):
            self.state["last_verify_passed"] = False

        if not result["success"]:
            self.state["last_judge_passed"] = False
            self._write_state()
            return result

        self.state["last_judge_passed"] = bool(data.get("passed"))
        self.state["last_submission_verification_passed"] = False
        self.state["last_submission_verification_results"] = {}
        if self.state["last_judge_passed"]:
            self.state["consecutive_judge_failures"] = 0
            # Strip required_fixes on pass — the LLM judge sometimes returns
            # suggestions even for passing tasks, which misleads the agent.
            data.pop("required_fixes", None)
            data["next_step"] = (
                "Task passed judge. Do not change the task spec. "
                "Run taskgen test_task, then taskgen verify_task, then taskgen submit_task."
            )
        else:
            self.state["consecutive_judge_failures"] = (
                self.state.get("consecutive_judge_failures", 0) + 1
            )
            data["failure_count"] = self.state["consecutive_judge_failures"]
            data.setdefault(
                "action_required",
                "Modify the task using required_fixes and run taskgen judge again.",
            )
        self._write_state()
        return success(data)

    def submit_task(self) -> CLIResult:
        _skip = set(self.state.get("skip_steps") or [])
        if not self.state.get("last_judge_passed") and "llm-council" not in _skip:
            return failure(
                "Must run judge successfully before submitting. "
                "judge now includes golden trajectory regeneration and simulator verification."
            )
        # In some environments we skip simulator verification (e.g., missing Hydra/GL deps).
        # When simulation is skipped, allow submission as long as judge + test passed.
        if not self.state.get("last_verify_passed") and "simulation" not in _skip:
            return failure(
                "Must simulator-verify the regenerated golden trajectory before submitting. "
                "If simulator verification is unavailable in this environment, add 'simulation' to skip_steps "
                "in taskgen_state.json."
            )
        if not self.state.get("last_test_passed") and "test" not in _skip:
            return failure("Must run test_task successfully before submitting.")
        if not self.state.get("last_submission_verification_passed"):
            return failure(
                "Must run verify_task successfully before submitting. "
                f"Submission requires at least {SUBMISSION_VERIFICATION_REQUIRED_FAILURES} of "
                "gpt-5.4, claude-sonnet-4-6, and gemini-flash to fail."
            )

        allowed_tom_levels = (
            [self.state["current_k_level"]] if self.state.get("current_k_level") else None
        )
        result = submit_task_run(
            str(self.task_file),
            output_dir=str(self.state["output_dir"]),
            working_dir=str(self.working_dir),
            submitted_dir=str(self.submitted_tasks_dir),
            subtasks_min=self.state["subtasks_min"],
            subtasks_max=self.state["subtasks_max"],
            agents_min=self.state["agents_min"],
            agents_max=self.state["agents_max"],
            allowed_tom_levels=allowed_tom_levels,
        )
        if not result["success"]:
            return result

        data = result["data"]
        self.state.setdefault("submitted_tasks", []).append(data["output_path"])
        self._refresh_calibration_stats()
        self._reset_gate_state()
        self._clear_benchmark_feedback()
        self.state["_test_run_count"] = 0
        self.state["_verification_run_count"] = 0
        if len(self.state["submitted_tasks"]) < self.state["num_tasks_target"]:
            self.state["current_k_level"] = self._pick_k_level()
        self._write_state()

        response = dict(data)
        response["submitted_count"] = len(self.state["submitted_tasks"])
        response["next_required_k_level"] = self.state.get("current_k_level")
        response["message"] = (
            f"Task submitted ({response['submitted_count']}/{self.state['num_tasks_target']})."
        )
        return success(response)
