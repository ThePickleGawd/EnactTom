"""
Evaluate task quality using the configured judge model(s).

Scores task on category-specific criteria and returns pass/fail with blockers.

Usage:
    # CLI
    python -m enacttom.cli.judge_task task.json [--working-dir DIR] [--threshold 0.65]

    # Programmatic
    from enacttom.cli.judge_task import run
    result = run("task.json", working_dir="/tmp/work")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from enacttom.cli import CLIResult, failure, success
from enacttom.cli.task_metadata import compute_strict_tom_metadata
from enacttom.cli.validate_task import static_validate_trajectory
from enacttom.task_gen.task_bootstrap import canonicalize_task_problem_pddl

def _dedupe_strings(items: List[str]) -> List[str]:
    deduped: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _stage_failure(
    *,
    stage: str,
    error: str,
    blocking_failures: List[Dict[str, Any]],
    required_fixes: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> CLIResult:
    payload: Dict[str, Any] = {
        "stage": stage,
        "blocking_failures": blocking_failures,
        "required_fixes": _dedupe_strings(required_fixes),
        "summary": f"FAIL at {stage}: {error}",
    }
    if extra:
        payload.update(extra)
    return failure(error, data=payload)


def _build_model_result(judgment: Any, min_threshold: float, overall_threshold: float) -> Dict[str, Any]:
    failed_criteria: List[Dict[str, Any]] = []
    for criterion_name, criterion in judgment.criteria_scores.items():
        if criterion.score < min_threshold:
            failed_criteria.append(
                {
                    "criterion": criterion_name,
                    "score": criterion.score,
                    "reasoning": criterion.reasoning,
                }
            )

    if not failed_criteria and not judgment.is_valid and judgment.overall_score < overall_threshold:
        failed_criteria.append(
            {
                "criterion": "overall_score",
                "score": judgment.overall_score,
                "reasoning": f"Average score below threshold {overall_threshold:.2f}.",
            }
        )

    return {
        "passed": judgment.is_valid,
        "score": judgment.overall_score,
        "failed_criteria": failed_criteria,
    }


def _build_blocking_failures(
    model_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    failed_models = {
        model for model, result in model_results.items() if not result.get("passed")
    }
    aggregated: Dict[str, Dict[str, Any]] = {}
    for model, result in model_results.items():
        for criterion in result.get("failed_criteria", []):
            name = criterion["criterion"]
            entry = aggregated.setdefault(
                name,
                {"criterion": name, "models": {}, "evidence": []},
            )
            entry["models"][model] = criterion["score"]
            evidence = criterion.get("reasoning", "").strip()
            if evidence:
                entry["evidence"].append(f"{model}: {evidence}")

    def _sort_key(item: Dict[str, Any]) -> Any:
        scores = list(item["models"].values())
        avg = sum(scores) / len(scores) if scores else 1.0
        return (-len(scores), avg, item["criterion"])

    blockers: List[Dict[str, Any]] = []
    for entry in sorted(aggregated.values(), key=_sort_key):
        models = entry["models"]
        blockers.append(
            {
                "criterion": entry["criterion"],
                "consensus": set(models.keys()) == failed_models if failed_models else False,
                "models": models,
                "evidence": entry["evidence"][:3],
            }
        )
    return blockers


def run(
    task_file: str,
    working_dir: str = None,
    scene_file: str = None,
    trajectory_dir: str = None,
    models: Optional[List[str]] = None,
    threshold: float = 0.65,
    difficulty: Optional[str] = None,
    user_query: Optional[str] = None,
    required_tom_level: Optional[int] = None,
    verified_trajectory_hash: Optional[str] = None,
    skip_regenerate_golden: bool = False,
    skip_steps: Optional[List[str]] = None,
) -> CLIResult:
    """
    Evaluate task quality using the configured judge model(s).

    Args:
        task_file: Path to task JSON file.
        working_dir: Optional working directory (for scene data, trajectory lookup).
        scene_file: Optional explicit scene data JSON file.
        trajectory_dir: Optional path to benchmark rollout data.
        models: Judge model names (default: ["gpt-5.4-mini"]).
        threshold: Overall score threshold for passing.
        difficulty: Difficulty level context (easy/medium/hard).
        user_query: Optional user query the task should align with.
        required_tom_level: Optional strict tom_level the task must satisfy.
        verified_trajectory_hash: Previously simulator-verified trajectory hash.
            When the newly regenerated plan matches this hash, the prior
            simulator verification is reused.
        skip_regenerate_golden: Assume golden trajectory has already been
            regenerated by the caller and skip the regeneration step here.

    Returns:
        CLIResult with a simplified blocker-first payload.
    """
    _skip = set(skip_steps or [])
    hard_gates = {"pddl", "tom", "simulation", "llm-council", "structure"}
    skipped_hard_gates = sorted(_skip & hard_gates)
    if skipped_hard_gates:
        return failure(
            "Task judging requires the full pipeline; cannot skip: "
            + ", ".join(skipped_hard_gates)
        )

    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")

    scene_data = _load_scene_data(working_dir, scene_file)
    changed = canonicalize_task_problem_pddl(task_data, scene_data)
    if changed:
        with open(task_path, "w") as f:
            json.dump(task_data, f, indent=2)
            f.write("\n")

    # Deterministic strict PDDL verification is the first judge gate.
    from enacttom.cli.verify_pddl import run as verify_pddl_run

    verify_result = verify_pddl_run(task_file, working_dir=working_dir)
    if not verify_result["success"]:
        err = verify_result.get("error") or ""
        return _stage_failure(
            stage="pddl",
            error=err,
            blocking_failures=[
                {
                    "criterion": "pddl_verification",
                    "consensus": True,
                    "models": {},
                    "evidence": [err],
                }
            ],
            required_fixes=[err],
            extra=verify_result.get("data"),
        )

    try:
        strict_tom = compute_strict_tom_metadata(task_data, scene_data=scene_data)
    except Exception as e:
        return _stage_failure(
            stage="pddl",
            error=f"Strict ToM verification failed: {e}",
            blocking_failures=[
                {
                    "criterion": "strict_tom_verification",
                    "consensus": True,
                    "models": {},
                    "evidence": [str(e)],
                }
            ],
            required_fixes=[f"Fix strict ToM verification: {e}"],
        )

    strict_tom_level = strict_tom.get("tom_level")
    if False:
        pass
    if "tom" not in _skip and isinstance(strict_tom_level, int) and strict_tom_level < 1:
        return _stage_failure(
            stage="pddl",
            error="Strict tom_level is 0. EnactToM tasks must require at least one grounded K() dependency.",
            blocking_failures=[
                {
                    "criterion": "strict_tom_level",
                    "consensus": True,
                    "models": {},
                    "evidence": [
                        "Computed strict tom_level is 0, which means the task has no benchmark-valid ToM requirement.",
                    ],
                }
            ],
            required_fixes=[
                "Add a grounded K() dependency so strict tom_level is at least 1.",
            ],
            extra={
                "computed_tom_level": strict_tom_level,
                "strict_tom_verification": strict_tom,
            },
        )
    if "tom" not in _skip and required_tom_level is not None and strict_tom_level != required_tom_level:
        return _stage_failure(
            stage="pddl",
            error=f"Strict tom_level is {strict_tom_level} but required tom_level is {required_tom_level}.",
            blocking_failures=[
                {
                    "criterion": "strict_tom_level",
                    "consensus": True,
                    "models": {},
                    "evidence": [
                        f"Computed strict tom_level is {strict_tom_level}; required is {required_tom_level}.",
                    ],
                }
            ],
            required_fixes=[
                f"Raise strict tom_level from {strict_tom_level} to {required_tom_level} with a real epistemic dependency.",
            ],
            extra={
                "required_tom_level": required_tom_level,
                "computed_tom_level": strict_tom_level,
                "strict_tom_verification": strict_tom,
            },
        )

    golden_status: Dict[str, Any] = {
        "regenerated": False,
        "sim_verification_ran": False,
        "sim_verified": False,
        "spec_hash": None,
        "trajectory_hash": None,
    }
    if skip_regenerate_golden:
        metadata = task_data.get("golden_trajectory_metadata", {})
        if isinstance(metadata, dict):
            golden_status["spec_hash"] = metadata.get("spec_hash")
            golden_status["trajectory_hash"] = metadata.get("trajectory_hash")
        golden_status["regenerated"] = False
    else:
        try:
            from enacttom.pddl.planner import regenerate_golden_trajectory

            regen = regenerate_golden_trajectory(
                task_data,
                scene_data=scene_data,
                source="judge",
                task_file=task_file,
            )
            golden_status.update(
                {
                    "regenerated": True,
                    "spec_hash": regen.get("spec_hash"),
                    "trajectory_hash": regen.get("trajectory_hash"),
                    "num_steps": regen.get("num_steps"),
                }
            )
        except Exception as e:
            return _stage_failure(
                stage="golden_regeneration",
                error=f"Failed to regenerate golden trajectory from task spec: {e}",
                blocking_failures=[
                    {
                        "criterion": "golden_trajectory_regeneration",
                        "consensus": True,
                        "models": {},
                        "evidence": [str(e)],
                    }
                ],
                required_fixes=[f"Fix golden trajectory regeneration: {e}"],
                extra={"golden_trajectory": golden_status},
            )

        needs_sim_verification = (
            not verified_trajectory_hash
            or golden_status["trajectory_hash"] != verified_trajectory_hash
        )
        if needs_sim_verification:
            golden_status["sim_verification_ran"] = True
            if not working_dir:
                err = "Simulator verification requires working_dir."
                return _stage_failure(
                    stage="simulation",
                    error=err,
                    blocking_failures=[
                        {
                            "criterion": "golden_trajectory_execution",
                            "consensus": True,
                            "models": {},
                            "evidence": [err],
                        }
                    ],
                    required_fixes=[err],
                    extra={"golden_trajectory": golden_status},
                )
            verification = _verify_golden_trajectory_in_sim(
                task_file=task_file,
                working_dir=working_dir,
                task_data=task_data,
                scene_data=scene_data,
            )
            if not verification["success"]:
                err = verification.get("error") or ""
                payload = verification.get("data") or {}
                payload["golden_trajectory"] = golden_status
                return _stage_failure(
                    stage="simulation",
                    error=err,
                    blocking_failures=[
                        {
                            "criterion": "golden_trajectory_execution",
                            "consensus": True,
                            "models": {},
                            "evidence": [
                                err,
                                *(
                                    [f"Failed action: {payload.get('action')}"]
                                    if payload.get("action")
                                    else []
                                ),
                            ],
                        }
                    ],
                    required_fixes=[
                        err,
                        *(
                            [f"Repair or replan the failing action `{payload.get('action')}`."]
                            if payload.get("action")
                            else []
                        ),
                    ],
                    extra=payload,
                )
            golden_status["sim_verified"] = True
            golden_status["sim_verification"] = verification.get("data")
        else:
            golden_status["sim_verified"] = True
            golden_status["sim_verification_skipped_reason"] = (
                "Regenerated trajectory hash matches the last simulator-verified plan."
            )

    # Validate task structure before expensive LLM calls
    from enacttom.cli.validate_task import validate

    validation = validate(task_data, scene_data)
    if not validation["success"]:
        return _stage_failure(
            stage="validation",
            error=validation["error"],
            blocking_failures=[
                {
                    "criterion": "task_validation",
                    "consensus": True,
                    "models": {},
                    "evidence": [validation["error"]],
                }
            ],
            required_fixes=[validation["error"]],
            extra=validation.get("data"),
        )

    # Find latest trajectory dir if not explicitly provided
    traj_path = None
    if trajectory_dir:
        traj_path = Path(trajectory_dir)
    elif working_dir:
        trajectories_dir = Path(working_dir) / "agent_trajectories"
        if trajectories_dir.exists():
            task_dirs = sorted(trajectories_dir.glob("task_*"), key=lambda p: p.name)
            if task_dirs:
                run_dirs = sorted(task_dirs[-1].glob("run_*"), key=lambda p: p.name)
                if run_dirs:
                    traj_path = run_dirs[-1]

    # Create judge and evaluate
    from enacttom.task_gen.judge import Judge

    judge = Judge(
        models=models,
        overall_threshold=threshold,
        difficulty=difficulty,
        user_query=user_query if not difficulty else None,
        skip_steps=skip_steps,
    )

    try:
        verdict = judge.evaluate(
            task_data,
            scene_data=scene_data,
            trajectory_dir=traj_path,
        )
    except RuntimeError as e:
        if "infrastructure errors" in str(e) or "no model judgments" in str(e):
            raise
        return failure(f"Evaluation failed: {e}")
    except Exception as e:
        return failure(f"Evaluation failed: {e}")

    # Build result
    model_results = {
        model: _build_model_result(j, judge.min_criterion_threshold, judge.overall_threshold)
        for model, j in verdict.judgments.items()
    }
    blocking_failures = _build_blocking_failures(model_results)
    required_fixes = _dedupe_strings(verdict.required_fixes)
    if not required_fixes and blocking_failures:
        required_fixes = [
            blocker["evidence"][0]
            for blocker in blocking_failures
            if blocker.get("evidence")
        ][:3]

    result_data: Dict[str, Any] = {
        "stage": "judge",
        "passed": verdict.passed,
        "overall_score": verdict.overall_score,
        "threshold": judge.overall_threshold,
        "models": list(verdict.judgments.keys()),
        "model_results": model_results,
        "blocking_failures": blocking_failures,
        "required_fixes": required_fixes,
        "golden_trajectory": golden_status,
    }

    if verdict.disagreements:
        result_data["disagreements"] = verdict.disagreements

    if verdict.passed:
        result_data["summary"] = (
            f"PASS - Judge accepted task (score: {verdict.overall_score:.2f})"
        )
    else:
        result_data["summary"] = (
            f"FAIL - Task did not pass judge (score: {verdict.overall_score:.2f})"
        )

    # Save verdict JSON
    verdict_dict = verdict.to_dict()

    if working_dir:
        judgments_dir = Path(working_dir) / "judgments"
        judgments_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        judgment_file = judgments_dir / f"judgment_{timestamp}.json"
        with open(judgment_file, "w") as f:
            json.dump(verdict_dict, f, indent=2)
        result_data["judgment_file"] = str(judgment_file)

    # Print status to stderr (human-readable)
    from enacttom.task_gen.judge import Colors

    status = "PASS" if verdict.passed else "FAIL"
    color = Colors.GREEN if verdict.passed else Colors.RED
    print(
        f"\n{Colors.BOLD}{Colors.CYAN}=== Task Evaluation (Council) ==={Colors.RESET}",
        file=sys.stderr,
    )
    print(
        f"{Colors.BOLD}{color}{status}{Colors.RESET} - Score: {verdict.overall_score:.2f} "
        f"(threshold: {judge.overall_threshold})",
        file=sys.stderr,
    )
    print(f"Models: {', '.join(verdict.judgments.keys())}", file=sys.stderr)

    if verdict.disagreements:
        print(f"\n{Colors.YELLOW}Model disagreements:{Colors.RESET}", file=sys.stderr)
        for d in verdict.disagreements:
            print(f"  - {d}", file=sys.stderr)

    if not verdict.passed and blocking_failures:
        print(f"\n{Colors.YELLOW}Blocking failures:{Colors.RESET}", file=sys.stderr)
        for blocker in blocking_failures:
            models_str = ", ".join(f"{m}={s:.2f}" for m, s in blocker["models"].items())
            consensus = " consensus" if blocker.get("consensus") else ""
            print(
                f"  - {blocker['criterion']}{consensus}"
                + (f" [{models_str}]" if models_str else ""),
                file=sys.stderr,
            )
            for evidence in blocker.get("evidence", [])[:2]:
                print(f"    {evidence}", file=sys.stderr)

    if not verdict.passed and required_fixes:
        print(f"\n{Colors.YELLOW}Required fixes:{Colors.RESET}", file=sys.stderr)
        for i, fix in enumerate(required_fixes, 1):
            print(f"  {i}. {fix}", file=sys.stderr)

    return success(result_data)


def _verify_golden_trajectory_in_sim(
    *,
    task_file: str,
    working_dir: Optional[str],
    task_data: Dict[str, Any],
    scene_data: Any,
) -> CLIResult:
    golden = task_data.get("golden_trajectory", [])
    if not golden:
        return failure("Deterministic planner produced empty trajectory.")

    static_errors = static_validate_trajectory(task_data, golden, scene_data)
    if static_errors:
        return failure(
            f"Golden trajectory static validation failed: {static_errors[0]}",
            data={"all_errors": static_errors},
        )

    if not working_dir:
        return failure(
            "Cannot simulator-verify golden trajectory without a working_dir."
        )

    num_agents = task_data.get("num_agents", 2)
    cmd = [
        sys.executable,
        "-m",
        "enacttom.cli.verify_trajectory",
        str(task_file),
        "--working-dir",
        str(working_dir),
        "--config-name",
        f"examples/enacttom_{num_agents}_robots",
    ]
    try:
        project_root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired:
        return failure("Golden trajectory verification timed out (20 min). Possible navmesh issue.")
    except Exception as e:
        return failure(f"Golden trajectory verification subprocess error: {e}")

    try:
        stdout = proc.stdout
        json_start = stdout.find("{")
        if json_start >= 0:
            stdout = stdout[json_start:]
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return failure(
            f"Failed to parse golden trajectory verification output: {proc.stderr[:500]}"
        )

    if result.get("success") and result.get("data", {}).get("valid"):
        return success(result["data"])

    return failure(
        result.get("error", "Golden trajectory verification failed."),
        data=result.get("data", {}),
    )


def _load_scene_data(working_dir: Optional[str], scene_file: Optional[str]):
    """Load SceneData from file if available."""
    scene_path = Path(scene_file) if scene_file else None
    if scene_path is None and working_dir:
        scene_path = Path(working_dir) / "current_scene.json"

    if scene_path and scene_path.exists():
        try:
            from enacttom.task_gen.scene_loader import SceneData

            with open(scene_path) as sf:
                sd = json.load(sf)
            return SceneData(
                episode_id=sd["episode_id"],
                scene_id=sd["scene_id"],
                rooms=sd.get("rooms", []),
                furniture=sd.get("furniture", []),
                objects=sd.get("objects", []),
                articulated_furniture=sd.get("articulated_furniture", []),
                furniture_in_rooms=sd.get("furniture_in_rooms", {}),
                objects_on_furniture=sd.get("objects_on_furniture", {}),
                agent_spawns=sd.get("agent_spawns", {}),
            )
        except Exception:
            pass
    return None


if __name__ == "__main__":
    import argparse

    from enacttom.cli import print_result

    parser = argparse.ArgumentParser(description="Evaluate task quality with judge model(s)")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--working-dir", default=None, help="Working directory")
    parser.add_argument("--scene-file", default=None, help="Scene data JSON file")
    parser.add_argument("--trajectory-dir", default=None, help="Benchmark rollout data directory")
    parser.add_argument("--models", default=None, help="Comma-separated judge models")
    parser.add_argument("--threshold", type=float, default=0.65, help="Score threshold (default: 0.65)")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--required-tom-level", type=int, default=None)
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")] if args.models else None
    result = run(
        args.task_file,
        working_dir=args.working_dir,
        scene_file=args.scene_file,
        trajectory_dir=args.trajectory_dir,
        models=model_list,
        threshold=args.threshold,
        difficulty=args.difficulty,
        required_tom_level=args.required_tom_level,
    )
    print_result(result)
