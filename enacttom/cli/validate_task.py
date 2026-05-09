"""
Validate task JSON structure without running the benchmark simulator.

Checks required fields, episode ID consistency, mechanic bindings,
object IDs, PDDL goal syntax, subtask DAG, agent IDs, and more.

Usage:
    # CLI
    python -m enacttom.cli.validate_task task.json [--working-dir DIR] [--scene-file FILE]

    # Programmatic
    from enacttom.cli.validate_task import validate
    result = validate(task_data, scene_data)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from enacttom.cli import CLIResult, failure, success
from enacttom.task_gen.spec_validator import (
    validate_blocking_spec,
    validate_room_restriction_trajectory,
)
from enacttom.task_gen.task_bootstrap import canonicalize_task_problem_pddl

if TYPE_CHECKING:
    from enacttom.task_gen.scene_loader import SceneData


ID_PATTERN = re.compile(r"\b[a-z_]+_\d+\b")
PLACEHOLDER_SCENE_IDS = {"", "unknown", "synthetic_scene"}
PLACEHOLDER_EPISODE_IDS = {"", "unknown", "synthetic_episode"}


def validate_runtime_grounding(
    task_data: Dict[str, Any],
    scene_data: Optional["SceneData"] = None,
) -> Optional[str]:
    """Validate that a task is grounded in a real benchmark scene."""
    scene_id = str(task_data.get("scene_id", "")).strip()
    episode_id = str(task_data.get("episode_id", "")).strip()

    if scene_id in PLACEHOLDER_SCENE_IDS or episode_id in PLACEHOLDER_EPISODE_IDS:
        return (
            "Task is not benchmarkable: scene_id/episode_id must come from a real "
            "dataset scene, not placeholder scene values."
        )

    if scene_data:
        expected_episode = scene_data.episode_id
        if episode_id != expected_episode:
            return (
                f"episode_id must be '{expected_episode}' (from loaded scene), "
                f"got '{episode_id}'"
            )

    return None


def _extract_known_task_ids(task_data: Dict[str, Any]) -> set[str]:
    """
    Best-effort fallback for IDs referenced by the task spec itself.

    This mirrors the static task-gen verifier so trajectory validation does not
    reject planner-generated IDs when the loaded scene inventory is incomplete.
    """
    ids: set[str] = set()

    def collect_from_value(value: Any) -> None:
        if isinstance(value, str):
            ids.update(ID_PATTERN.findall(value))
        elif isinstance(value, dict):
            for nested in value.values():
                collect_from_value(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_from_value(nested)

    for key in (
        "task",
        "title",
        "problem_pddl",
        "tom_reasoning",
        "initial_states",
        "subtasks",
        "mechanic_bindings",
        "agent_secrets",
    ):
        collect_from_value(task_data.get(key))

    return ids


def validate(
    task_data: Dict[str, Any],
    scene_data: Optional["SceneData"] = None,
) -> CLIResult:
    """
    Validate task JSON structure.

    Args:
        task_data: Parsed task dict.
        scene_data: Optional SceneData object (for object ID validation).

    Returns:
        CLIResult with data keys: valid, task_id, title, mechanics, tom_required, summary.
    """
    canonicalize_task_problem_pddl(task_data, scene_data)

    # Core required fields
    required_fields = [
        "task_id", "title", "task", "episode_id",
        "mechanic_bindings", "agent_secrets",
        "agent_actions"
    ]

    missing = [f for f in required_fields if f not in task_data]
    if missing:
        return failure(f"Missing required fields: {missing}")

    grounding_error = validate_runtime_grounding(task_data, scene_data=scene_data)
    if grounding_error:
        return failure(grounding_error)

    # Shared deterministic spec checks
    spec_errors = validate_blocking_spec(task_data, scene_data)
    if spec_errors:
        return failure(
            spec_errors[0],
            data={"errors": spec_errors, "summary": f"Task has {len(spec_errors)} validation error(s)"},
        )

    # golden_trajectory is a derived artifact and may be stale/missing before
    # verify_golden_trajectory. Static trajectory checks run separately.

    # Validate problem_pddl (canonical goal format, required).
    has_problem_pddl = isinstance(task_data.get("problem_pddl"), str) and bool(task_data.get("problem_pddl").strip())
    legacy_goal_fields = [k for k in ("goals", "pddl_goal", "pddl_ordering", "pddl_owners") if k in task_data]

    if not has_problem_pddl:
        return failure("Task must have non-empty 'problem_pddl'.")
    if legacy_goal_fields:
        return failure(
            "Legacy goal fields are not supported. "
            f"Remove {legacy_goal_fields} and encode goals in problem_pddl only."
        )

    try:
        from enacttom.pddl.problem_pddl import parse_problem_pddl
        from enacttom.pddl.domain import ENACTTOM_DOMAIN
        from enacttom.pddl.dsl import validate_goal_predicates
        from enacttom.pddl.problem_pddl import validate_problem_pddl_self_contained
        from enacttom.pddl.runtime_projection import project_runtime_from_parsed_problem

        parsed = parse_problem_pddl(task_data["problem_pddl"])
        declared_domain = task_data.get("pddl_domain", "")
        if declared_domain and parsed.domain_name != declared_domain:
            return failure(
                "problem_pddl domain mismatch: "
                f":domain is '{parsed.domain_name}' but pddl_domain is '{declared_domain}'"
            )
        if parsed.domain_name != ENACTTOM_DOMAIN.name:
            return failure(
                f"Unsupported problem domain '{parsed.domain_name}'. "
                f"Expected '{ENACTTOM_DOMAIN.name}'."
            )

        pred_errors = validate_goal_predicates(parsed.goal_formula, ENACTTOM_DOMAIN)
        if pred_errors:
            return failure(
                "problem_pddl goal predicate validation failed: "
                + "; ".join(pred_errors)
            )

        self_contained_errors = validate_problem_pddl_self_contained(
            parsed, num_agents=task_data.get("num_agents", 2)
        )
        if self_contained_errors:
            return failure(
                "problem_pddl must be self-contained: "
                + "; ".join(self_contained_errors)
            )

        projection = project_runtime_from_parsed_problem(parsed)
        if not projection.is_valid:
            return failure(
                "Runtime functional projection is invalid: "
                + "; ".join(projection.invalid_reasons)
            )
        unsupported_probes = [probe.source_pddl for probe in projection.probes if not probe.supported]
        if unsupported_probes:
            return failure(
                "Unsupported runtime literal-ToM probe formulas: "
                + "; ".join(unsupported_probes)
            )
    except Exception as e:
        return failure(f"Invalid problem_pddl: {e}")

    # Validate agent IDs are consistent with num_agents
    num_agents = task_data.get("num_agents", 2)
    valid_agent_ids = {f"agent_{i}" for i in range(num_agents)}

    for agent_id in task_data.get("agent_actions", {}).keys():
        if agent_id not in valid_agent_ids:
            return failure(
                f"agent_actions contains invalid agent ID '{agent_id}'. "
                f"Valid: {sorted(valid_agent_ids)} (num_agents={num_agents})"
            )

    for agent_id in task_data.get("agent_secrets", {}).keys():
        if agent_id not in valid_agent_ids:
            return failure(
                f"agent_secrets contains invalid agent ID '{agent_id}'. "
                f"Valid: {sorted(valid_agent_ids)} (num_agents={num_agents})"
            )

    # Check task description is not empty
    if not task_data.get("task") or len(task_data.get("task", "")) < 20:
        return failure("task field must be at least 20 characters")

    # Validate object IDs in task description exist in scene
    task_desc = task_data.get("task", "")
    object_pattern = r'\b[a-z_]+_\d+\b'
    object_refs = re.findall(object_pattern, task_desc)

    if scene_data:
        valid_scene_ids = set(
            scene_data.rooms + scene_data.furniture + scene_data.objects
        )
        invalid_task_refs = [
            ref for ref in object_refs
                        if ref not in valid_scene_ids and not ref.startswith(("agent_",))
        ]
        if invalid_task_refs:
            return failure(
                f"task references objects that don't exist in scene: {invalid_task_refs}. "
                f"Use only: {list(scene_data.objects)[:10]}..."
            )

        # Check agent_secrets for invented object IDs
        for agent_id, secrets in task_data.get("agent_secrets", {}).items():
            for secret in secrets:
                secret_refs = re.findall(object_pattern, secret)
                invalid_secret_refs = [
                    ref for ref in secret_refs
                    if ref not in valid_scene_ids and not ref.startswith(("agent_",))
                ]
                if invalid_secret_refs:
                    return failure(
                        f"agent_secrets[{agent_id}] references objects that don't exist in scene: "
                        f"{invalid_secret_refs}"
                    )

    # Check mechanic_bindings structure
    TRIGGER_OBJECT_MECHANICS = {
        "inverse_state", "remote_control", "state_mirroring"
    }
    for i, binding in enumerate(task_data.get("mechanic_bindings", [])):
        if "mechanic_type" not in binding:
            return failure(f"mechanic_bindings[{i}] missing mechanic_type")
        mechanic_type = binding.get("mechanic_type", "")
        if mechanic_type in TRIGGER_OBJECT_MECHANICS and "trigger_object" not in binding:
            return failure(f"mechanic_bindings[{i}] ({mechanic_type}) missing trigger_object")
        if mechanic_type == "limited_bandwidth" and not isinstance(binding.get("message_limits"), dict):
            return failure(f"mechanic_bindings[{i}] (limited_bandwidth) missing message_limits dict")

    # Check agent_secrets has proper structure
    if not isinstance(task_data.get("agent_secrets"), dict):
        return failure("agent_secrets must be a dict")

    # Validate message_targets if present
    raw_mt = task_data.get("message_targets")
    if raw_mt is not None:
        if not isinstance(raw_mt, dict):
            return failure(
                "message_targets must be a dict mapping agent_id to list of allowed recipient agent_ids"
            )
        for mt_agent, mt_targets in raw_mt.items():
            if mt_agent not in valid_agent_ids:
                return failure(
                    f"message_targets key '{mt_agent}' is not a valid agent ID. "
                    f"Valid: {sorted(valid_agent_ids)}"
                )
            if not isinstance(mt_targets, list):
                return failure(f"message_targets['{mt_agent}'] must be a list of agent IDs")
            for target_id in mt_targets:
                if target_id not in valid_agent_ids:
                    return failure(
                        f"message_targets['{mt_agent}'] contains invalid agent ID '{target_id}'. "
                        f"Valid: {sorted(valid_agent_ids)}"
                    )
                if target_id == mt_agent:
                    return failure(
                        f"message_targets['{mt_agent}'] contains self-reference. "
                        "Agents cannot target themselves."
                    )

    # Try to parse as GeneratedTask
    try:
        from enacttom.task_gen import GeneratedTask

        GeneratedTask.from_dict(task_data)
    except Exception as e:
        return failure(f"Failed to parse as GeneratedTask: {e}")

    return success({
        "valid": True,
        "task_id": task_data.get("task_id"),
        "title": task_data.get("title"),
        "mechanics": [b.get("mechanic_type") for b in task_data.get("mechanic_bindings", [])],
        "tom_required": task_data.get("theory_of_mind_required", False),
        "summary": "Task structure is valid",
    })


def static_validate_trajectory(
    task_data: Dict[str, Any],
    golden: List[Dict[str, Any]],
    scene_data: Optional["SceneData"] = None,
) -> List[str]:
    """
    Fast static validation of golden trajectory (no simulator required).

    Catches invalid object IDs, action names, missing agents, malformed syntax.

    Returns:
        List of error messages (empty if valid).
    """
    errors: List[str] = []
    num_agents = task_data.get("num_agents", 2)
    valid_agents = {f"agent_{i}" for i in range(num_agents)}

    valid_actions = {
        "Navigate", "Open", "Close", "Pick", "Place",
        "Communicate", "Wait", "FindObjectTool", "FindReceptacleTool", "FindRoomTool",
    }

    valid_ids: set = set()
    if scene_data:
        valid_ids.update(scene_data.rooms)
        valid_ids.update(scene_data.furniture)
        valid_ids.update(scene_data.objects)

    valid_ids.update(_extract_known_task_ids(task_data))

    for step_idx, step in enumerate(golden):
        actions = step.get("actions", [])
        if not actions:
            errors.append(f"Step {step_idx}: No actions array")
            continue

        agents_in_step: set = set()
        for action_entry in actions:
            agent = action_entry.get("agent", "")
            action_str = action_entry.get("action", "")

            if agent not in valid_agents:
                errors.append(
                    f"Step {step_idx}: Invalid agent '{agent}' (valid: {sorted(valid_agents)})"
                )
            agents_in_step.add(agent)

            match = re.match(r'(\w+)(?:\[(.*)\])?$', action_str)
            if not match:
                errors.append(f"Step {step_idx}: Malformed action '{action_str}'")
                continue

            action_name, args = match.group(1), match.group(2)

            if action_name not in valid_actions:
                errors.append(f"Step {step_idx}: Unknown action '{action_name}'")

            if action_name in (
                "Wait", "Communicate", "FindObjectTool", "FindReceptacleTool", "FindRoomTool"
            ) or not args:
                continue

            skip_words = {"on", "within", "next_to", "None", ""}
            parts = [p.strip() for p in args.split(",")]
            for part in parts:
                if part in skip_words:
                    continue
                if valid_ids and part not in valid_ids:
                    errors.append(f"Step {step_idx}: Unknown object '{part}' in {action_str}")

        missing_agents = valid_agents - agents_in_step
        if missing_agents:
            errors.append(
                f"Step {step_idx}: Missing actions for {sorted(missing_agents)} (add Wait if idle)"
            )

    # Check room-restriction consistency
    errors.extend(validate_room_restriction_trajectory(task_data, scene_data, golden))

    return errors[:10]


def run(
    task_file: str,
    working_dir: str = None,
    scene_file: str = None,
) -> CLIResult:
    """
    Validate task JSON structure from file paths.

    Args:
        task_file: Path to task JSON file.
        working_dir: Optional working directory (for current_scene.json).
        scene_file: Optional explicit scene data JSON file.

    Returns:
        CLIResult.
    """
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")

    # Load scene data
    scene_data = None
    scene_path = Path(scene_file) if scene_file else None
    if scene_path is None and working_dir:
        scene_path = Path(working_dir) / "current_scene.json"

    if scene_path and scene_path.exists():
        try:
            from enacttom.task_gen.scene_loader import SceneData

            with open(scene_path) as sf:
                sd = json.load(sf)
            scene_data = SceneData(
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
            pass  # Proceed without scene data

    changed = canonicalize_task_problem_pddl(task_data, scene_data)
    if changed:
        with open(task_path, "w") as f:
            json.dump(task_data, f, indent=2)
            f.write("\n")

    return validate(task_data, scene_data)


if __name__ == "__main__":
    import argparse

    from enacttom.cli import print_result

    parser = argparse.ArgumentParser(description="Validate task JSON structure")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--working-dir", default=None, help="Working directory (for current_scene.json)")
    parser.add_argument("--scene-file", default=None, help="Explicit scene data JSON file")
    args = parser.parse_args()

    result = run(args.task_file, working_dir=args.working_dir, scene_file=args.scene_file)
    print_result(result)
