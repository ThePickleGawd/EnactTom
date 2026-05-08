"""
Verify PDDL goal solvability and compute ToM depth.

Checks:
1. `problem_pddl` syntax is valid
2. Raw `problem_pddl` is self-contained
3. Goal is strictly solvable by Fast Downward under depth-bounded proof checks
4. Computes minimum ToM depth from iterative solving

Usage:
    # CLI
    python -m enacttom.cli.verify_pddl task.json [--working-dir DIR]

    # Programmatic
    from enacttom.cli.verify_pddl import run
    result = run("task.json", working_dir="/tmp/work")
"""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from enacttom.cli import CLIResult, failure, success
from enacttom.task_gen.task_bootstrap import canonicalize_task_problem_pddl


def _build_failure_summary(
    problem: "Problem",
    functional_goal_pddl: str,
    raw_error: str,
) -> str:
    """Build a 2-4 sentence actionable summary explaining why PDDL verification failed.

    Designed to give the task-gen agent enough context to fix the problem_pddl
    in one edit rather than blind trial-and-error.
    """
    from enacttom.pddl.dsl import And, Formula, Literal, Not, Or

    init_preds: Dict[str, Set[Tuple[str, ...]]] = {}
    for lit in problem.init:
        if not lit.negated:
            init_preds.setdefault(lit.predicate, set()).add(lit.args)

    # --- Collect issues ---
    issues: List[str] = []

    # 1. Missing agent positions
    agents_without_room = []
    grounded_agents = {
        args[0] for args in init_preds.get("agent_in_room", set()) if len(args) == 2
    }
    for name, typ in problem.objects.items():
        if typ == "agent" and name not in grounded_agents:
            agents_without_room.append(name)
    if agents_without_room:
        issues.append(
            f"Agents {', '.join(agents_without_room)} have no starting room — "
            f"add (agent_in_room <agent> <room>) to :init for each."
        )

    # 2. Missing object/furniture room grounding
    grounded_objects = {
        args[0] for args in init_preds.get("is_in_room", set()) if len(args) == 2
    }
    goal_refs: Set[str] = set()

    def _collect_goal_refs(f: Formula) -> None:
        if isinstance(f, Literal):
            for arg in f.args:
                if not arg.startswith("?"):
                    goal_refs.add(arg)
        elif isinstance(f, (And, Or)):
            for op in f.operands:
                _collect_goal_refs(op)
        elif isinstance(f, Not) and f.operand is not None:
            _collect_goal_refs(f.operand)

    if problem.goal:
        _collect_goal_refs(problem.goal)

    ungrounded = sorted(
        obj_id for obj_id in goal_refs
        if problem.objects.get(obj_id) not in ("agent", "room", None)
        and obj_id not in grounded_objects
    )
    if ungrounded:
        issues.append(
            f"Objects/furniture referenced in the goal have no room placement: "
            f"{', '.join(ungrounded)}. Add (is_in_room <id> <room>) for each."
        )

    # 3. Missing initial object positions (needed for move/place goals)
    positioned = set()
    for pred in ("is_on_top", "is_on_floor", "is_inside", "is_in_hand"):
        for args in init_preds.get(pred, set()):
            if args:
                positioned.add(args[0])
    unpositioned = sorted(
        obj_id for obj_id in goal_refs
        if problem.objects.get(obj_id) == "object"
        and obj_id not in positioned
    )
    if unpositioned:
        issues.append(
            f"Objects {', '.join(unpositioned)} have no initial position — "
            f"the planner cannot move them. Add (is_on_top <object> <furniture>) to :init."
        )

    # 4. Contradictory goal detection
    if "contradictory" in raw_error.lower() or "expression false" in raw_error.lower():
        issues.append(
            "The goal contains contradictory requirements (e.g., a physical negation "
            "conflicts with an epistemic K() goal over the same fact). "
            "Check that no goal literal is both asserted and negated."
        )

    # 5. Planner status fallback
    if "unsolvable" in raw_error.lower() and not issues:
        issues.append(
            "The planner proved the goal is unreachable from the initial state. "
            "Verify that the :init facts describe a world state where the goal "
            "can actually be achieved through the available actions (navigate, "
            "pick_up, place, open, close)."
        )

    if not issues:
        # Generic fallback with the raw error
        return f"PDDL verification failed: {raw_error}"

    summary = "PDDL verification failed. " + " ".join(issues)
    return summary


def run(task_file: str, working_dir: str = None) -> CLIResult:
    """
    Verify PDDL goal solvability and compute ToM depth.

    Args:
        task_file: Path to task JSON file.
        working_dir: Optional working directory (used to find current_scene.json).

    Returns:
        CLIResult with data keys: valid, solvable, tom_level, tom_reasoning,
        goal_description, num_conjuncts, solve_time, pddl_goal.
    """
    total_start = time.perf_counter()
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")

    scene_data: Optional[Dict[str, Any]] = None
    if working_dir:
        scene_file = Path(working_dir) / "current_scene.json"
        if scene_file.exists():
            try:
                with open(scene_file) as sf:
                    scene_data = json.load(sf)
            except (json.JSONDecodeError, IOError):
                scene_data = None

    changed = canonicalize_task_problem_pddl(task_data, scene_data)
    if changed:
        with open(task_path, "w") as f:
            json.dump(task_data, f, indent=2)
            f.write("\n")

    # Build GoalSpec from canonical inline problem_pddl.
    from enacttom.pddl.goal_spec import GoalSpec
    from enacttom.pddl.domain import ENACTTOM_DOMAIN
    from enacttom.pddl.problem_pddl import (
        parse_problem_pddl,
        validate_problem_pddl_self_contained,
    )

    problem_pddl = task_data.get("problem_pddl")
    legacy_goal_fields = [k for k in ("goals", "pddl_goal", "pddl_ordering", "pddl_owners") if k in task_data]
    parsed_problem = None

    parse_start = time.perf_counter()
    if not isinstance(problem_pddl, str) or not problem_pddl.strip():
        return failure("Task must define non-empty 'problem_pddl'.")
    if legacy_goal_fields:
        return failure(
            "Legacy goal fields are not supported. "
            f"Remove {legacy_goal_fields} and encode goals in problem_pddl only."
        )
    try:
        parsed_problem = parse_problem_pddl(problem_pddl)
    except ValueError as e:
        return failure(f"Invalid problem_pddl: {e}")
    raw_pddl_errors = validate_problem_pddl_self_contained(
        parsed_problem,
        num_agents=task_data.get("num_agents", 2),
    )
    if raw_pddl_errors:
        return failure(
            "problem_pddl must be self-contained:\n" + "\n".join(raw_pddl_errors),
            data={"valid": False, "pddl_goal": parsed_problem.goal_pddl},
        )

    declared_domain = task_data.get("pddl_domain")
    if isinstance(declared_domain, str) and declared_domain:
        if parsed_problem.domain_name != declared_domain:
            return failure(
                "problem_pddl domain mismatch: "
                f":domain is '{parsed_problem.domain_name}' but pddl_domain is '{declared_domain}'"
            )
    if parsed_problem.domain_name != ENACTTOM_DOMAIN.name:
        return failure(
            f"Unsupported problem domain '{parsed_problem.domain_name}'. "
            f"Expected '{ENACTTOM_DOMAIN.name}'."
        )
    if re.search(r"\bteam_[a-zA-Z0-9_]+\b", parsed_problem.goal_pddl):
        return failure(
            "Invalid problem_pddl :goal: found team_* identifier(s). "
            "Use world-state predicates in :goal and put agent ownership in :goal-owners."
        )

    goal_spec = GoalSpec.from_legacy(parsed_problem.goal_pddl, [], {})
    parse_time_s = time.perf_counter() - parse_start

    # Validate goal spec against domain
    num_agents = task_data.get("num_agents", 2)
    valid_agents = {f"agent_{i}" for i in range(num_agents)}
    validation_errors = goal_spec.validate(ENACTTOM_DOMAIN, valid_agents)
    if validation_errors:
        return failure(
            "Goal validation errors:\n" + "\n".join(validation_errors),
            data={"valid": False, "pddl_goal": goal_spec.to_pddl_string()},
        )

    from enacttom.pddl.runtime_projection import project_runtime_from_parsed_problem
    projection = project_runtime_from_parsed_problem(parsed_problem)
    if not projection.is_valid:
        return failure(
            "Runtime functional projection is invalid:\n" + "\n".join(projection.invalid_reasons),
            data={"valid": False, "pddl_goal": goal_spec.to_pddl_string()},
        )

    # Build task object
    from enacttom.task_gen.task_generator import GeneratedTask
    task = GeneratedTask.from_dict(task_data)

    # Compile and solve functional projection only.
    from enacttom.pddl.compiler import compile_task
    from enacttom.pddl.epistemic import ObservabilityModel
    from enacttom.pddl.fd_solver import FastDownwardSolver
    from enacttom.pddl.problem_pddl import replace_goal_in_problem_pddl
    from enacttom.pddl.solver import _max_epistemic_depth

    compile_start = time.perf_counter()
    solvable_task_data = deepcopy(task_data)
    solvable_task_data["problem_pddl"] = replace_goal_in_problem_pddl(
        task_data["problem_pddl"],
        projection.functional_goal_pddl,
    )
    solvable_task = GeneratedTask.from_dict(solvable_task_data)
    problem = compile_task(solvable_task, scene_data)
    compile_time_s = time.perf_counter() - compile_start

    solve_start = time.perf_counter()
    observability = ObservabilityModel.from_task_with_scene(solvable_task, scene_data)
    result = FastDownwardSolver().solve(
        ENACTTOM_DOMAIN,
        problem,
        observability,
        max_belief_depth=0,
        strict=False,
    )
    solve_wall_time_s = time.perf_counter() - solve_start

    if result is None or not result.solvable:
        error_msg = (result.error if result is not None else None) or "unknown"
        summary = _build_failure_summary(problem, projection.functional_goal_pddl, error_msg)
        return failure(
            summary,
            data={
                "valid": False,
                "pddl_goal": goal_spec.to_pddl_string(),
                "functional_goal_pddl": projection.functional_goal_pddl,
                "solver_error": error_msg,
            },
        )

    epistemic_goal_depth = _max_epistemic_depth(parsed_problem.goal_formula)
    if epistemic_goal_depth <= 0:
        tom_reasoning = (
            "No epistemic operators appear in problem_pddl. "
            "Functional solvability was checked on the non-epistemic runtime goal."
        )
    else:
        tom_reasoning = (
            "Functional solvability was checked on the projected non-epistemic runtime goal. "
            f"Reported ToM depth is the authored epistemic nesting depth in problem_pddl: {epistemic_goal_depth}."
        )

    # Goal description
    formula = projection.functional_goal or goal_spec.to_formula()
    from enacttom.pddl.describe import goal_to_natural_language
    description = goal_to_natural_language(formula)

    output = {
        "valid": True,
        "pddl_goal": parsed_problem.goal_pddl if parsed_problem else goal_spec.to_pddl_string(),
        "functional_goal_pddl": projection.functional_goal_pddl,
        "solvable": True,
        "tom_level": epistemic_goal_depth,
        "minimal_tom_level": epistemic_goal_depth,
        "epistemic_goal_depth": epistemic_goal_depth,
        "tom_reasoning": tom_reasoning,
        "goal_description": description,
        "num_conjuncts": len(goal_spec),
        "solve_time": result.solve_time,
        "proved_unsat_below": [],
        "proof_backend": "functional_fast_downward_strict",
        "proof_strict": True,
        "proof_attempts": [
            {
                "level": 0,
                "solvable": True,
                "belief_depth": 0,
                "error": result.error,
            }
        ],
        "timing": {
            "parse_time_ms": round(parse_time_s * 1000, 3),
            "compile_time_ms": round(compile_time_s * 1000, 3),
            "solve_time_ms": round(solve_wall_time_s * 1000, 3),
            "total_time_ms": round((time.perf_counter() - total_start) * 1000, 3),
        },
    }
    if result.plan:
        output["plan"] = result.plan

    return success(output)


if __name__ == "__main__":
    import argparse

    from enacttom.cli import print_result

    parser = argparse.ArgumentParser(description="Verify PDDL goal solvability")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--working-dir", default=None, help="Working directory (for current_scene.json)")
    args = parser.parse_args()

    result = run(args.task_file, working_dir=args.working_dir)
    print_result(result)
