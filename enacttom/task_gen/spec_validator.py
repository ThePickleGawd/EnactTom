"""
Shared deterministic task-spec validation used by generation and CLI verification.

Phase 1 scope:
- Template placeholder detection
- Mechanic field/schema checks
- Mechanic binding completeness checks
- Mechanic binding scene-reference checks
- Category/schema consistency checks
- Basic golden-trajectory structural checks
- Room-restriction vs Navigate trajectory checks
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from enacttom.mechanics.handlers import MECHANIC_INFO
from enacttom.task_gen.task_generator import normalize_mechanic_bindings


_ACTION_PATTERN = re.compile(r"(\w+)(?:\[(.*)\])?$")
_PLACEHOLDER_PATTERNS = (
    "REPLACE_WITH_",
    "REPLACE_CONTAINER",
    "REPLACE_ITEM",
    "EXAMPLE_",
    "ACTION_NAME[TARGET]",
)

_VALID_ACTIONS = {
    "Navigate", "Open", "Close", "Pick", "Place",
    "Communicate", "Wait", "FindObjectTool", "FindReceptacleTool", "FindRoomTool",
}


def _get_scene_list(scene_data: Optional[Any], key: str) -> List[str]:
    if not scene_data:
        return []
    if isinstance(scene_data, dict):
        value = scene_data.get(key, [])
    else:
        value = getattr(scene_data, key, [])
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


def _get_scene_dict(scene_data: Optional[Any], key: str) -> Dict[str, List[str]]:
    if not scene_data:
        return {}
    if isinstance(scene_data, dict):
        value = scene_data.get(key, {})
    else:
        value = getattr(scene_data, key, {})
    if not isinstance(value, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in value.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [x for x in v if isinstance(x, str)]
    return out


def _collect_placeholder_hits(obj: Any, path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(obj, str):
        if any(p in obj for p in _PLACEHOLDER_PATTERNS):
            hits.append(path or "$")
        return hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if any(p in k for p in _PLACEHOLDER_PATTERNS):
                key_path = f"{path}.{k}" if path else k
                hits.append(f"{key_path} (key)")
            child = f"{path}.{k}" if path else k
            hits.extend(_collect_placeholder_hits(v, child))
        return hits
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            child = f"{path}[{i}]" if path else f"[{i}]"
            hits.extend(_collect_placeholder_hits(v, child))
    return hits


def _parse_action(action_str: str) -> Tuple[Optional[str], Optional[str]]:
    m = _ACTION_PATTERN.match(action_str or "")
    if not m:
        return None, None
    action_name, args = m.group(1), m.group(2)
    if args == "":
        args = None
    return action_name, args


def _parse_communicate_recipient(args: Optional[str]) -> Optional[str]:
    if not isinstance(args, str) or not args:
        return None
    match = re.search(r",\s*(agent_\d+)\s*$", args)
    if not match:
        return None
    return match.group(1)


def _has_ordering_cycle(ordering: List[Dict[str, Any]]) -> bool:
    """
    Check if pddl_ordering constraints form a cycle.

    Uses DFS with recursion stack (same pattern as dag._has_cycle).
    """
    # Build adjacency list
    graph: Dict[str, Set[str]] = {}
    nodes: Set[str] = set()
    for constraint in ordering:
        if not isinstance(constraint, dict):
            continue
        before = constraint.get("before", "")
        after = constraint.get("after", "")
        if before and after:
            graph.setdefault(before, set()).add(after)
            nodes.add(before)
            nodes.add(after)

    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True  # Back edge = cycle
        rec_stack.discard(node)
        return False

    for node in nodes:
        if node not in visited:
            if dfs(node):
                return True
    return False


def _extract_room_restrictions(
    task_data: Dict[str, Any],
) -> Dict[str, Set[str]]:
    restrictions: Dict[str, Set[str]] = {}

    def add_binding(binding: Dict[str, Any]) -> None:
        if not isinstance(binding, dict):
            return
        if binding.get("mechanic_type") != "room_restriction":
            return
        rooms = binding.get("restricted_rooms", [])
        agents = binding.get("for_agents", [])
        if not isinstance(rooms, list) or not isinstance(agents, list):
            return
        room_set = {r for r in rooms if isinstance(r, str) and r}
        for agent in agents:
            if isinstance(agent, str) and agent:
                restrictions.setdefault(agent, set()).update(room_set)

    for binding in normalize_mechanic_bindings(
        task_data.get("mechanic_bindings", []),
        problem_pddl=task_data.get("problem_pddl"),
    ):
        if isinstance(binding, dict):
            add_binding(binding)

    # Be tolerant of swapped-fields legacy tasks: dicts in active_mechanics.
    for mech in task_data.get("active_mechanics", []):
        if isinstance(mech, dict):
            add_binding(mech)

    return restrictions


def _build_target_to_room(scene_data: Optional[Any]) -> Dict[str, str]:
    from enacttom.pddl.planner import build_target_to_room_map
    return build_target_to_room_map(scene_data)


def _iter_formula_nodes(formula: Any):
    """Depth-first walk of DSL formulas."""
    if formula is None:
        return
    yield formula
    inner = getattr(formula, "inner", None)
    if inner is not None:
        yield from _iter_formula_nodes(inner)
    operand = getattr(formula, "operand", None)
    if operand is not None:
        yield from _iter_formula_nodes(operand)
    operands = getattr(formula, "operands", None)
    if isinstance(operands, tuple):
        for op in operands:
            yield from _iter_formula_nodes(op)


def _collect_literals(formula: Any) -> List[Any]:
    literals: List[Any] = []
    for node in _iter_formula_nodes(formula):
        if node.__class__.__name__ == "Literal":
            literals.append(node)
    return literals


def _looks_container_like_furniture(furniture_id: str) -> bool:
    prefixes = (
        "cabinet_",
        "drawer_",
        "chest_of_drawers_",
        "wardrobe_",
        "fridge_",
        "microwave_",
        "safe_",
        "dishwasher_",
        "washer_dryer_",
        "locker_",
        "cupboard_",
    )
    return isinstance(furniture_id, str) and furniture_id.startswith(prefixes)


def _validate_goal_support_surfaces(
    goal_formula: Any,
    articulated: Set[str],
) -> List[str]:
    errors: List[str] = []
    if not articulated:
        return errors

    for literal in _collect_literals(goal_formula):
        if getattr(literal, "negated", False):
            continue
        if getattr(literal, "predicate", None) != "is_on_top":
            continue
        args = getattr(literal, "args", ())
        if len(args) != 2:
            continue
        obj_id, furniture_id = args
        if furniture_id not in articulated:
            continue
        if not _looks_container_like_furniture(furniture_id):
            continue
        errors.append(
            "problem_pddl goal uses "
            f"(is_on_top {obj_id} {furniture_id}) on articulated/container furniture. "
            "Use is_inside for cabinets, drawers, fridges, safes, and similar containers, "
            "or choose a non-articulated support surface like a table, shelf, stand, or floor."
        )

    return errors


def _collect_epistemic_goals(formula: Any) -> List[Any]:
    goals: List[Any] = []
    for node in _iter_formula_nodes(formula):
        if node.__class__.__name__ in {"Knows", "Believes"}:
            goals.append(node)
    return goals


def validate_room_restriction_trajectory(
    task_data: Dict[str, Any],
    scene_data: Optional[Any],
    golden: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    errors: List[str] = []
    restrictions = _extract_room_restrictions(task_data)
    if not restrictions:
        return errors
    if not scene_data:
        return errors

    if golden is None:
        golden = task_data.get("golden_trajectory", [])
    if not isinstance(golden, list):
        return errors

    target_to_room = _build_target_to_room(scene_data)
    if not target_to_room:
        return errors

    for step_idx, step in enumerate(golden):
        if not isinstance(step, dict):
            continue
        actions = step.get("actions", [])
        if not isinstance(actions, list):
            continue
        for action_entry in actions:
            if not isinstance(action_entry, dict):
                continue
            agent = action_entry.get("agent", "")
            action_str = action_entry.get("action", "")
            if not isinstance(agent, str) or not isinstance(action_str, str):
                continue
            action_name, args = _parse_action(action_str)
            if action_name != "Navigate" or not args:
                continue
            target = args.split(",")[0].strip()
            target_room = target_to_room.get(target)
            if not target_room:
                continue
            if target_room in restrictions.get(agent, set()):
                errors.append(
                    f"Step {step_idx}: {agent} navigates to '{target}' "
                    f"(in {target_room}) but is restricted from {target_room}."
                )
    return errors


def validate_blocking_spec(
    task_data: Dict[str, Any],
    scene_data: Optional[Any] = None,
) -> List[str]:
    errors: List[str] = []

    # ------------------------------------------------------------------
    # Basic schema/category sanity
    # ------------------------------------------------------------------
    required_fields = [
        "task_id", "title", "task", "episode_id",
        "mechanic_bindings", "agent_secrets", "agent_actions",
    ]
    missing = [f for f in required_fields if f not in task_data]
    if missing:
        errors.append(f"Missing required fields: {missing}")
        # Continue so callers can still get other deterministic errors when possible.

    category = task_data.get("category")
    if isinstance(category, str) and category:
        if category not in {"cooperative", "mixed"}:
            errors.append(
                "category must be one of ['cooperative', 'mixed']"
            )

    # ------------------------------------------------------------------
    # Template placeholder artifacts
    # ------------------------------------------------------------------
    # task_id is auto-generated at submit time, so exclude it from the check.
    placeholder_hits = _collect_placeholder_hits(
        {k: v for k, v in task_data.items() if k != "task_id"}
    )
    if placeholder_hits:
        shown = placeholder_hits[:12]
        more = "" if len(placeholder_hits) <= 12 else f" (+{len(placeholder_hits)-12} more)"
        errors.append(
            f"Unfilled template placeholders in: {shown}{more}. "
            "Replace all REPLACE_WITH_* and template placeholders."
        )

    # ------------------------------------------------------------------
    # Agent ID consistency
    # ------------------------------------------------------------------
    num_agents = task_data.get("num_agents", 2)
    if not isinstance(num_agents, int) or num_agents <= 0:
        errors.append("num_agents must be a positive integer")
        num_agents = 2
    valid_agent_ids = {f"agent_{i}" for i in range(num_agents)}

    for field_name in ("agent_actions", "agent_secrets"):
        field_val = task_data.get(field_name, {})
        if not isinstance(field_val, dict):
            errors.append(f"{field_name} must be a dict")
            continue
        for agent_id in field_val.keys():
            if agent_id not in valid_agent_ids:
                errors.append(
                    f"{field_name} contains invalid agent ID '{agent_id}'. "
                    f"Valid IDs: {sorted(valid_agent_ids)}"
                )

    # ------------------------------------------------------------------
    # Message targets validation
    # ------------------------------------------------------------------
    raw_mt = task_data.get("message_targets")
    if raw_mt is not None:
        if not isinstance(raw_mt, dict):
            errors.append("message_targets must be a dict")
        else:
            for mt_agent, mt_targets in raw_mt.items():
                if mt_agent not in valid_agent_ids:
                    errors.append(
                        f"message_targets key '{mt_agent}' is not a valid agent ID. "
                        f"Valid: {sorted(valid_agent_ids)}"
                    )
                if not isinstance(mt_targets, list):
                    errors.append(f"message_targets['{mt_agent}'] must be a list of agent IDs")
                    continue
                for target_id in mt_targets:
                    if not isinstance(target_id, str) or target_id not in valid_agent_ids:
                        errors.append(
                            f"message_targets['{mt_agent}'] contains invalid agent ID '{target_id}'. "
                            f"Valid: {sorted(valid_agent_ids)}"
                        )
                    elif target_id == mt_agent:
                        errors.append(
                            f"message_targets['{mt_agent}'] contains self-reference"
                        )

    # ------------------------------------------------------------------
    # Scene inventory
    # ------------------------------------------------------------------
    rooms = set(_get_scene_list(scene_data, "rooms"))
    furniture = set(_get_scene_list(scene_data, "furniture"))
    objects = set(_get_scene_list(scene_data, "objects"))
    articulated = set(_get_scene_list(scene_data, "articulated_furniture"))
    scene_known_ids = rooms | furniture | objects

    # ------------------------------------------------------------------
    # Mechanics schema and binding checks
    # ------------------------------------------------------------------
    active_mechanics = task_data.get("active_mechanics", [])
    mechanic_bindings = normalize_mechanic_bindings(
        task_data.get("mechanic_bindings", []),
        problem_pddl=task_data.get("problem_pddl"),
    )

    if active_mechanics is not None and not isinstance(active_mechanics, list):
        errors.append("active_mechanics must be a list")
        active_mechanics = []
    if mechanic_bindings is not None and not isinstance(mechanic_bindings, list):
        errors.append("mechanic_bindings must be a list")
        mechanic_bindings = []

    if isinstance(active_mechanics, list):
        dict_like = sum(1 for x in active_mechanics if isinstance(x, dict))
        if dict_like > 0:
            if not mechanic_bindings:
                errors.append(
                    "Detected swapped mechanic fields: binding dicts found in active_mechanics while mechanic_bindings is empty."
                )

    for i, binding in enumerate(mechanic_bindings if isinstance(mechanic_bindings, list) else []):
        if not isinstance(binding, dict):
            errors.append(f"mechanic_bindings[{i}] must be an object")
            continue

        mechanic_type = binding.get("mechanic_type")
        if not isinstance(mechanic_type, str) or not mechanic_type:
            errors.append(f"mechanic_bindings[{i}] missing mechanic_type")
            continue

        info = MECHANIC_INFO.get(mechanic_type)
        if not info:
            errors.append(f"mechanic_bindings[{i}] has unknown mechanic_type '{mechanic_type}'")
            continue

        # Required keys by mechanic.
        missing_required: List[str] = []
        required_keys = info.get("setup_keys", [])
        for key in required_keys:
            val = binding.get(key)
            if val is None or (isinstance(val, str) and not val):
                missing_required.append(key)

        if missing_required:
            errors.append(
                f"mechanic_bindings[{i}] ({mechanic_type}) missing required fields: {missing_required}"
            )

        # Validate room_restriction structure.
        if mechanic_type == "room_restriction":
            rr = binding.get("restricted_rooms")
            fa = binding.get("for_agents")
            if not isinstance(rr, list) or not rr:
                errors.append(
                    f"mechanic_bindings[{i}] (room_restriction) requires non-empty restricted_rooms list"
                )
            if not isinstance(fa, list) or not fa:
                errors.append(
                    f"mechanic_bindings[{i}] (room_restriction) requires non-empty for_agents list"
                )
            if isinstance(rr, list) and rooms:
                unknown_rooms = [r for r in rr if isinstance(r, str) and r not in rooms]
                if unknown_rooms:
                    errors.append(
                        f"mechanic_bindings[{i}] (room_restriction) unknown restricted_rooms: {unknown_rooms}"
                    )
            if isinstance(fa, list):
                unknown_agents = [a for a in fa if isinstance(a, str) and a not in valid_agent_ids]
                if unknown_agents:
                    errors.append(
                        f"mechanic_bindings[{i}] (room_restriction) unknown for_agents: {unknown_agents}"
                    )

        # Validate limited_bandwidth structure.
        if mechanic_type == "limited_bandwidth":
            ml = binding.get("message_limits")
            if not isinstance(ml, dict) or not ml:
                errors.append(
                    f"mechanic_bindings[{i}] (limited_bandwidth) requires non-empty message_limits dict"
                )
            elif isinstance(ml, dict):
                for agent_id, limit in ml.items():
                    if agent_id not in valid_agent_ids:
                        errors.append(
                            f"mechanic_bindings[{i}] (limited_bandwidth) unknown agent '{agent_id}' in message_limits"
                        )
                    if not isinstance(limit, (int, float)) or limit < 1:
                        errors.append(
                            f"mechanic_bindings[{i}] (limited_bandwidth) message_limits[{agent_id}] must be a positive integer, got {limit}"
                        )

        if mechanic_type == "remote_control":
            target_state = binding.get("target_state", "is_open")
            if target_state not in {"is_open", "is_closed", "is_unlocked", "is_locked"}:
                errors.append(
                    f"mechanic_bindings[{i}] (remote_control) unsupported target_state '{target_state}'. "
                    "Supported: is_open, is_closed, is_unlocked, is_locked."
                )

        if mechanic_type == "state_mirroring":
            target_state = binding.get("target_state", "is_open")
            if target_state not in {"is_open", "is_closed"}:
                errors.append(
                    f"mechanic_bindings[{i}] (state_mirroring) unsupported target_state '{target_state}'. "
                    "Supported: is_open, is_closed."
                )

        # Validate binding object references against scene (when available).
        if scene_known_ids:
            for key in ("trigger_object", "target_object"):
                val = binding.get(key)
                if isinstance(val, str) and val and val not in scene_known_ids:
                    errors.append(
                        f"mechanic_bindings[{i}] ({mechanic_type}): {key}='{val}' not found in scene."
                    )

    mechanic_types = {
        b.get("mechanic_type")
        for b in (mechanic_bindings or [])
        if isinstance(b, dict) and isinstance(b.get("mechanic_type"), str)
    }
    if category in {"cooperative", "mixed"} and num_agents > 1:
        asymmetry_mechanics = {
            "room_restriction",
            "restricted_communication",
            "remote_control",
        }
        if mechanic_types and not (mechanic_types & asymmetry_mechanics):
            errors.append(
                "cooperative/mixed tasks need at least one asymmetry mechanic "
                "(room_restriction, restricted_communication, or remote_control)."
            )

    # ------------------------------------------------------------------
    # PDDL validation (problem_pddl only)
    # ------------------------------------------------------------------
    problem_pddl = task_data.get("problem_pddl")
    has_problem_pddl = isinstance(problem_pddl, str) and bool(problem_pddl.strip())
    legacy_goal_fields = [
        k for k in ("goals", "pddl_goal", "pddl_ordering", "pddl_owners")
        if k in task_data
    ]

    if not has_problem_pddl:
        errors.append("Task must define non-empty problem_pddl.")
    if legacy_goal_fields:
        errors.append(
            "Legacy goal fields are not supported. "
            f"Remove {legacy_goal_fields} and encode goals in problem_pddl only."
        )

    if has_problem_pddl:
        try:
            from enacttom.pddl.domain import ENACTTOM_DOMAIN
            from enacttom.pddl.goal_spec import GoalSpec
            from enacttom.pddl.problem_pddl import parse_problem_pddl

            parsed_problem = parse_problem_pddl(problem_pddl)
            declared_domain = task_data.get("pddl_domain")
            if isinstance(declared_domain, str) and declared_domain:
                if parsed_problem.domain_name != declared_domain:
                    errors.append(
                        "problem_pddl domain mismatch: "
                        f":domain is '{parsed_problem.domain_name}' but pddl_domain is '{declared_domain}'"
                    )
            if parsed_problem.domain_name != ENACTTOM_DOMAIN.name:
                errors.append(
                    f"Unsupported problem domain '{parsed_problem.domain_name}'. "
                    f"Expected '{ENACTTOM_DOMAIN.name}'."
                )

            spec = GoalSpec.from_legacy(parsed_problem.goal_pddl, [], {})
            errors.extend(spec.validate(ENACTTOM_DOMAIN, valid_agent_ids))
            errors.extend(
                _validate_goal_support_surfaces(parsed_problem.goal_formula, articulated)
            )

            goal_lower = parsed_problem.goal_pddl.lower()
            if re.search(r"\bteam_[a-z0-9_]+\b", goal_lower):
                errors.append(
                    "problem_pddl :goal contains team_* identifiers. "
                    "Use world-state literals in :goal and encode agent ownership in :goal-owners."
                )

            if "has_most" in goal_lower or "has_at_least" in goal_lower or "has_item" in goal_lower:
                errors.append(
                    "problem_pddl goal uses item/inventory predicates, which are outside the "
                    "minimal EnactToM paper task surface."
                )

            if category == "mixed" and not parsed_problem.owners:
                errors.append(
                    "mixed task problem_pddl is missing :goal-owners section. "
                    "Each agent needs a personal objective for credit assignment."
                )
        except Exception as e:
            errors.append(f"problem_pddl validation error: {e}")

    # ------------------------------------------------------------------
    # Golden trajectory structural checks (optional derived artifact)
    # ------------------------------------------------------------------
    golden = task_data.get("golden_trajectory")
    if golden is None:
        return errors
    if not isinstance(golden, list):
        errors.append("golden_trajectory must be a list when provided.")
        return errors
    if not golden:
        return errors

    for step_idx, step in enumerate(golden):
        if not isinstance(step, dict):
            errors.append(f"golden_trajectory[{step_idx}] must be an object")
            continue
        actions = step.get("actions", [])
        if not isinstance(actions, list) or not actions:
            errors.append(f"golden_trajectory[{step_idx}] missing non-empty actions list")
            continue

        seen_agents: Set[str] = set()
        for action_idx, action_entry in enumerate(actions):
            if not isinstance(action_entry, dict):
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] must be an object"
                )
                continue
            agent = action_entry.get("agent")
            action_str = action_entry.get("action")
            if not isinstance(agent, str) or agent not in valid_agent_ids:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] has invalid agent '{agent}'"
                )
            if isinstance(agent, str):
                if agent in seen_agents:
                    errors.append(f"golden_trajectory[{step_idx}] has duplicate action for {agent}")
                seen_agents.add(agent)

            if not isinstance(action_str, str) or not action_str:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] missing action string"
                )
                continue
            action_name, args = _parse_action(action_str)
            if not action_name:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] has malformed action '{action_str}'"
                )
                continue
            if action_name not in _VALID_ACTIONS:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] unknown action '{action_name}'"
                )
                continue
            allowed_by_agent = task_data.get("agent_actions", {}).get(agent)
            if isinstance(allowed_by_agent, list) and action_name not in allowed_by_agent:
                errors.append(
                    f"golden_trajectory[{step_idx}].actions[{action_idx}] uses '{action_name}' "
                    f"for {agent}, but it is not in agent_actions[{agent}]"
                )

            # Open/Close on non-articulated furniture is always invalid.
            if action_name in {"Open", "Close"} and args and furniture and articulated:
                target = args.split(",")[0].strip()
                if target in furniture and target not in articulated:
                    errors.append(
                        f"golden_trajectory[{step_idx}] uses {action_name}[{target}] "
                        f"but {target} is not articulated/openable."
                    )

    # Multi-agent quality guard: avoid trajectories where one agent does all work.
    #
    # NOTE:
    # In some infra-light environments the golden trajectory may be missing or
    # degenerate (e.g., regeneration falls back to a placeholder with only Wait[]).
    # In that case, failing the entire task here prevents authors from iterating.
    # We therefore only enforce this guard when the golden trajectory contains
    # at least one non-Wait action.
    if len(valid_agent_ids) > 1:
        non_wait_counts: Dict[str, int] = {agent_id: 0 for agent_id in valid_agent_ids}
        any_non_wait = False
        for step in golden:
            actions = step.get("actions", []) if isinstance(step, dict) else []
            if not isinstance(actions, list):
                continue
            for entry in actions:
                if not isinstance(entry, dict):
                    continue
                agent = entry.get("agent")
                action_str = entry.get("action")
                if isinstance(agent, str) and agent in non_wait_counts and isinstance(action_str, str):
                    if action_str != "Wait[]":
                        any_non_wait = True
                        non_wait_counts[agent] += 1
                        action_name, args = _parse_action(action_str)
                        if action_name == "Communicate":
                            recipient = _parse_communicate_recipient(args)
                            if recipient in non_wait_counts:
                                non_wait_counts[recipient] += 1
        if any_non_wait:
            active_agents = [a for a, c in non_wait_counts.items() if c > 0]
            if len(active_agents) <= 1:
                errors.append(
                    "golden_trajectory has only one active agent (others only Wait[]). "
                    "Distribute required actions across multiple agents."
                )
    # Room restriction consistency against trajectory Navigate actions.
    errors.extend(validate_room_restriction_trajectory(task_data, scene_data, golden))

    return errors
