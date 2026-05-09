"""
Submit a validated task to the output directory.

Validates structure, checks goal/agent counts, generates filename,
and copies to output directories.

Gate checks (verify/judge/test passed) are NOT enforced here — they are
agent-level state managed by agent.py. This module only handles the
file operations and final validation.

Usage:
    # CLI
    python -m enacttom.cli.submit_task task.json --output-dir DIR [--working-dir DIR]

    # Programmatic
    from enacttom.cli.submit_task import run
    result = run("task.json", output_dir="data/enacttom/tasks")
"""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from enacttom.cli import CLIResult, failure, success
from enacttom.cli.task_metadata import compute_strict_tom_metadata
from enacttom.task_gen.task_bootstrap import canonicalize_task_problem_pddl


def _load_scene_data(working_dir: Optional[str], scene_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load raw scene JSON for object typing/init synthesis."""
    scene_path = Path(scene_file) if scene_file else None
    if scene_path is None and working_dir:
        scene_path = Path(working_dir) / "current_scene.json"
    if not scene_path or not scene_path.exists():
        return None
    try:
        with open(scene_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _ensure_domain_pddl_file(domain_name: str) -> Path:
    """
    Ensure a concrete shared domain PDDL file exists on disk.

    Writes `data/enacttom/pddl/domains/<domain_name>/domain.pddl` when missing.
    """
    project_root = Path(__file__).resolve().parents[2]
    domain_dir = project_root / "enacttom" / "pddl" / "domains" / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)
    domain_path = domain_dir / "domain.pddl"

    if not domain_path.exists():
        from enacttom.pddl.domain import ENACTTOM_DOMAIN

        # For now we support a single baked-in EnactToM domain.
        if domain_name != ENACTTOM_DOMAIN.name:
            raise ValueError(
                f"Unknown pddl_domain '{domain_name}'. Expected '{ENACTTOM_DOMAIN.name}'."
            )
        domain_path.write_text(ENACTTOM_DOMAIN.to_pddl() + "\n", encoding="utf-8")

    return domain_path


def _compute_tom_metadata(
    task_data: Dict[str, Any],
    scene_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute authoritative ToM metadata from canonical problem_pddl.

    Returns a dict with at least:
      - tom_level (int, may be 0)
      - tom_reasoning (optional str)
    """
    return compute_strict_tom_metadata(task_data, scene_data)


def _apply_runtime_metadata(task_data: Dict[str, Any]) -> None:
    """Backfill functional runtime metadata derived from problem_pddl."""
    from enacttom.pddl.runtime_projection import build_runtime_metadata, project_runtime_from_problem

    projection = project_runtime_from_problem(task_data["problem_pddl"])
    if not projection.is_valid:
        reasons = "; ".join(projection.invalid_reasons) or "unknown runtime projection error"
        raise ValueError(f"Invalid runtime functional projection: {reasons}")

    task_data.update(build_runtime_metadata(task_data))


def run(
    task_file: str,
    output_dir: str,
    working_dir: str = None,
    scene_file: str = None,
    submitted_dir: str = None,
    subtasks_min: int = 3,
    subtasks_max: int = 20,
    agents_min: int = 2,
    agents_max: int = 4,
    allowed_tom_levels: Optional[list] = None,
) -> CLIResult:
    """
    Submit a task to the output directory.

    Args:
        task_file: Path to task JSON file.
        output_dir: Permanent output directory.
        working_dir: Optional working directory (for scene data).
        scene_file: Optional explicit scene data JSON file.
        submitted_dir: Optional session-scoped submitted_tasks directory.
        subtasks_min: Minimum PDDL conjuncts or subtasks.
        subtasks_max: Maximum PDDL conjuncts or subtasks.
        agents_min: Minimum agent count.
        agents_max: Maximum agent count.
        allowed_tom_levels: If set, only tasks with a computed tom_level
            in this list are accepted. E.g. [2, 3] rejects k=1 tasks.

    Returns:
        CLIResult with data keys: output_path, filename, submitted_path.
    """
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")
    scene_data = _load_scene_data(working_dir=working_dir, scene_file=scene_file)
    canonicalize_task_problem_pddl(task_data, scene_data)

    # Validate task structure
    from enacttom.cli.validate_task import run as validate_run

    validation = validate_run(task_file, working_dir=working_dir, scene_file=scene_file)
    if not validation["success"]:
        return validation

    if not isinstance(task_data.get("problem_pddl"), str) or not task_data.get("problem_pddl", "").strip():
        return failure("Task must define non-empty problem_pddl before submit.")
    legacy_goal_fields = [k for k in ("goals", "pddl_goal", "pddl_ordering", "pddl_owners") if k in task_data]
    if legacy_goal_fields:
        return failure(
            "Legacy goal fields are not supported. "
            f"Remove {legacy_goal_fields} and encode goals in problem_pddl only."
        )

    # Compute and persist ToM metadata from canonical PDDL at submit time.
    try:
        tom_meta = _compute_tom_metadata(task_data, scene_data=scene_data)
        task_data["tom_level"] = tom_meta["tom_level"]
        if "tom_reasoning" in tom_meta:
            task_data["tom_reasoning"] = tom_meta["tom_reasoning"]
        else:
            task_data.pop("tom_reasoning", None)
        _apply_runtime_metadata(task_data)
    except Exception as e:
        return failure(f"Failed to compute tom_level: {e}")

    task_data["tom_level_method"] = "strict_fd"

    if task_data["tom_level"] < 1:
        return failure(
            "Task tom_level is 0. EnactToM tasks must require at least one K() goal in problem_pddl.",
            data={
                "computed_tom_level": task_data["tom_level"],
                "required_min": 1,
                "method": "strict_fd",
            },
        )

    # Enforce allowed ToM levels when specified.
    if allowed_tom_levels and task_data["tom_level"] not in allowed_tom_levels:
        allowed_str = ", ".join(str(l) for l in sorted(allowed_tom_levels))
        return failure(
            f"Task tom_level is {task_data['tom_level']} but only levels [{allowed_str}] "
            f"are allowed. Redesign the task to require deeper Theory-of-Mind reasoning.",
            data={"computed_tom_level": task_data["tom_level"], "allowed": allowed_tom_levels},
        )

    # Always regenerate golden trajectory from authoritative task spec.
    try:
        from enacttom.pddl.planner import regenerate_golden_trajectory

        regenerate_golden_trajectory(
            task_data,
            scene_data=scene_data,
            source="submit",
            task_file=str(task_path),
        )
    except Exception as e:
        return failure(f"Failed to regenerate golden trajectory from task spec: {e}")

    # Validate goal count from canonical problem_pddl.
    if task_data.get("problem_pddl"):
        from enacttom.pddl.problem_pddl import parse_problem_pddl
        from enacttom.pddl.dsl import collect_leaf_literals

        try:
            parsed_problem = parse_problem_pddl(task_data["problem_pddl"])
            num_goals = len(collect_leaf_literals(parsed_problem.goal_formula))
        except Exception:
            num_goals = 0
        if num_goals < subtasks_min:
            return failure(
                f"Task has {num_goals} PDDL goal conjuncts, minimum is {subtasks_min}.",
                data={"current": num_goals, "required_min": subtasks_min},
            )
        if num_goals > subtasks_max:
            return failure(
                f"Task has {num_goals} PDDL goal conjuncts, maximum is {subtasks_max}.",
                data={"current": num_goals, "required_max": subtasks_max},
            )

    # Validate agent count
    num_agents = task_data.get("num_agents", 2)
    if num_agents < agents_min:
        return failure(
            f"Task has {num_agents} agents, minimum is {agents_min}.",
            data={"current": num_agents, "required_min": agents_min},
        )
    if num_agents > agents_max:
        return failure(
            f"Task has {num_agents} agents, maximum is {agents_max}.",
            data={"current": num_agents, "required_max": agents_max},
        )

    # Canonical task_id naming (stable hash over problem + key metadata).
    title = task_data.get("title", "untitled")
    title_slug = re.sub(r"[^a-z0-9]+", "-", str(title).lower()).strip("-")[:40] or "untitled"
    category = str(task_data.get("category", "cooperative")).lower()
    scene_id = str(task_data.get("scene_id", "scene"))
    episode_id = str(task_data.get("episode_id", "episode"))
    problem_hash = hashlib.sha256(
        (task_data.get("problem_pddl", "") + "|" + category + "|" + scene_id + "|" + episode_id).encode("utf-8")
    ).hexdigest()[:8]
    canonical_task_id = f"enacttom-{scene_id}-{episode_id}-{category}-{title_slug}-{problem_hash}"
    task_data["task_id"] = canonical_task_id
    with open(task_path, "w") as f:
        json.dump(task_data, f, indent=2)
        f.write("\n")

    # Generate filename: {datetime}_{title_slug}.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"{timestamp}_{canonical_task_id}.json"

    # Copy to main output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / output_filename
    with open(output_path, "w") as f:
        json.dump(task_data, f, indent=2)
        f.write("\n")

    # Ensure shared domain PDDL exists.
    pddl_domain = task_data.get("pddl_domain", "enacttom")
    try:
        domain_path = _ensure_domain_pddl_file(pddl_domain)
    except ValueError as e:
        return failure(str(e))

    result_data = {
        "output_path": str(output_path),
        "filename": output_filename,
        "task_id": canonical_task_id,
        "title": task_data.get("title", "untitled"),
        "domain_path": str(domain_path),
        "pddl_domain": pddl_domain,
    }

    # Also copy to submitted_tasks/ for session tracking
    if submitted_dir:
        sub_dir = Path(submitted_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)
        submitted_path = sub_dir / output_filename
        with open(submitted_path, "w") as f:
            json.dump(task_data, f, indent=2)
            f.write("\n")
        result_data["submitted_path"] = str(submitted_path)

    return success(result_data)


if __name__ == "__main__":
    import argparse

    from enacttom.cli import print_result

    parser = argparse.ArgumentParser(description="Submit a validated task")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--output-dir", required=True, help="Permanent output directory")
    parser.add_argument("--working-dir", default=None, help="Working directory (for scene data)")
    parser.add_argument("--scene-file", default=None, help="Scene data JSON file")
    parser.add_argument("--submitted-dir", default=None, help="Session submitted_tasks directory")
    parser.add_argument("--subtasks-min", type=int, default=3)
    parser.add_argument("--subtasks-max", type=int, default=20)
    parser.add_argument("--agents-min", type=int, default=2)
    parser.add_argument("--agents-max", type=int, default=10)
    args = parser.parse_args()

    result = run(
        args.task_file,
        output_dir=args.output_dir,
        working_dir=args.working_dir,
        scene_file=args.scene_file,
        submitted_dir=args.submitted_dir,
        subtasks_min=args.subtasks_min,
        subtasks_max=args.subtasks_max,
        agents_min=args.agents_min,
        agents_max=args.agents_max,
    )
    print_result(result)
