"""
Task → PDDL compiler.

Converts a GeneratedTask + scene data into a PDDL Problem instance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from enacttom.pddl.dsl import (
    Domain, Formula, Literal, Knows, Believes, Not, And, Or,
    Problem,
)
from enacttom.pddl.problem_pddl import parse_problem_pddl, validate_problem_pddl_self_contained

if TYPE_CHECKING:
    from enacttom.task_gen.task_generator import GeneratedTask

# Valid PDDL identifiers: letters, digits, underscores, hyphens
_VALID_PDDL_ID = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')


def _infer_object_types(formula: Formula, domain: Domain) -> Dict[str, str]:
    """Infer PDDL types for object IDs referenced in a formula.

    Uses domain predicate signatures to determine the most specific type
    for each object. Falls back to naming convention if no predicate
    constraint narrows the type beyond ``object``.
    """
    # Build predicate-name -> list of param types
    pred_types: Dict[str, List[str]] = {}
    for pred in domain.predicates:
        pred_types[pred.name] = [p.type for p in pred.params]

    # Collect (object_id -> set of required types) from goal literals
    constraints: Dict[str, Set[str]] = {}

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            ptypes = pred_types.get(node.predicate, [])
            for i, arg in enumerate(node.args):
                if arg.startswith("?"):
                    continue
                typ = ptypes[i] if i < len(ptypes) else "object"
                constraints.setdefault(arg, set()).add(typ)
        elif isinstance(node, (Knows, Believes)):
            _walk(node.inner)
        elif isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)
        elif isinstance(node, And):
            for op in node.operands:
                _walk(op)
        elif isinstance(node, Or):
            for op in node.operands:
                _walk(op)

    _walk(formula)

    # Resolve: pick most specific type (anything that isn't root "object")
    result: Dict[str, str] = {}
    for obj_id, types in constraints.items():
        specific = types - {"object"}
        if len(specific) == 1:
            result[obj_id] = specific.pop()
        elif specific:
            # Multiple non-root types — shouldn't happen in valid PDDL, pick first
            result[obj_id] = sorted(specific)[0]
        else:
            # Only "object" constraint — use naming convention
            result[obj_id] = _type_from_name(obj_id)
    return result


def _type_from_name(obj_id: str) -> str:
    """Guess PDDL type from naming convention."""
    if obj_id.startswith("agent_"):
        return "agent"
    room_prefixes = (
        "room_", "kitchen_", "bedroom_", "bathroom_", "living_room_",
        "garage_", "hallway_", "lobby_", "office_", "closet_", "laundry_",
        "dining_", "pantry_", "porch_", "utility_",
    )
    if any(obj_id.startswith(p) for p in room_prefixes):
        return "room"
    # Default: most objects in enacttom scenes are furniture
    return "furniture"


def compile_task(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
) -> Problem:
    """Compile a GeneratedTask into a PDDL Problem."""
    task_problem_pddl = getattr(task, "problem_pddl", None)
    if isinstance(task_problem_pddl, str) and task_problem_pddl.strip():
        parsed = parse_problem_pddl(task_problem_pddl)
        validation_errors = validate_problem_pddl_self_contained(
            parsed,
            num_agents=getattr(task, "num_agents", None),
        )
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        problem = parsed.to_problem()

        # Mechanics are the single authored source for runtime constraints.
        _ensure_room_restrictions(task, problem)
        _ensure_mechanic_init_facts(task, problem)
        # Add default init facts for articulated furniture only.
        _add_default_init_facts(problem, scene_data)

        # Auto-populate grounding facts from scene data where missing.
        if scene_data:
            _ensure_scene_grounding(problem, scene_data)

        # Populate can_communicate if not already in init
        _ensure_can_communicate(task, problem)
        return problem

    raise ValueError(
        "Task must define non-empty problem_pddl. "
        "Legacy goal formats are no longer supported."
    )


def _looks_articulated_furniture(obj_id: str) -> bool:
    """Conservative fallback when scene metadata is unavailable."""
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
    return obj_id.startswith(prefixes)


def _add_default_init_facts(
    problem: Problem,
    scene_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Add default init facts for the planner.

    Only articulated furniture gets default closed-state facts. Open surfaces
    like tables should not be compiled as closed containers.
    """
    existing = {(l.predicate, l.args) for l in problem.init}
    articulated_ids = set()
    if scene_data:
        raw_articulated = scene_data.get("articulated_furniture") or []
        articulated_ids = {
            obj_id for obj_id in raw_articulated if isinstance(obj_id, str) and obj_id
        }

    for obj_id, typ in problem.objects.items():
        if typ == "furniture":
            is_articulated = obj_id in articulated_ids
            if not articulated_ids:
                is_articulated = _looks_articulated_furniture(obj_id)
            if not is_articulated:
                continue
            if (("is_open", (obj_id,)) not in existing
                    and ("is_closed", (obj_id,)) not in existing):
                problem.init.append(Literal("is_closed", (obj_id,)))


def _binding_value(binding: Any, key: str) -> Any:
    if isinstance(binding, dict):
        return binding.get(key)
    return getattr(binding, key, None)


def _literal_key(literal: Literal) -> tuple[str, tuple[str, ...]]:
    return literal.predicate, literal.args


def _ensure_fact(problem: Problem, predicate: str, args: tuple[str, ...], existing: Optional[Set[tuple[str, tuple[str, ...]]]] = None) -> None:
    if existing is None:
        existing = {_literal_key(lit) for lit in problem.init}
    fact = (predicate, args)
    if fact in existing:
        return
    problem.init.append(Literal(predicate, args))
    existing.add(fact)


def _ensure_mechanic_init_facts(task: "GeneratedTask", problem: Problem) -> None:
    """Compile planner-visible mechanic facts from mechanic_bindings."""
    existing = {_literal_key(lit) for lit in problem.init}

    for binding in getattr(task, "mechanic_bindings", []) or []:
        mechanic_type = _binding_value(binding, "mechanic_type")
        trigger = _binding_value(binding, "trigger_object")
        target = _binding_value(binding, "target_object")
        target_state = _binding_value(binding, "target_state") or "is_open"

        if mechanic_type == "inverse_state":
            if isinstance(trigger, str) and trigger:
                _ensure_fact(problem, "is_inverse", (trigger,), existing)
            continue

        if mechanic_type == "state_mirroring":
            if not (isinstance(trigger, str) and isinstance(target, str) and trigger and target):
                continue
            if target_state == "is_open":
                _ensure_fact(problem, "mirrors", (trigger, target), existing)
            elif target_state == "is_closed":
                _ensure_fact(problem, "mirrors_closed", (trigger, target), existing)
            else:
                raise ValueError(
                    f"Unsupported state_mirroring target_state '{target_state}'. "
                    "Supported: is_open, is_closed."
                )
            continue

        if mechanic_type == "remote_control":
            if not (isinstance(trigger, str) and isinstance(target, str) and trigger and target):
                continue
            if target_state == "is_open":
                _ensure_fact(problem, "controls", (trigger, target), existing)
            elif target_state == "is_unlocked":
                _ensure_fact(problem, "controls_unlocked", (trigger, target), existing)
            elif target_state == "is_closed":
                _ensure_fact(problem, "controls_closed", (trigger, target), existing)
            elif target_state == "is_locked":
                _ensure_fact(problem, "controls_locks", (trigger, target), existing)
            else:
                raise ValueError(
                    f"Unsupported remote_control target_state '{target_state}'. "
                    "Supported: is_open, is_closed, is_unlocked, is_locked."
                )
            continue

        if mechanic_type in {"limited_bandwidth", "restricted_communication", "room_restriction"}:
            continue


def _extract_restrictions_from_secrets(
    task: "GeneratedTask",
    problem: Problem,
) -> Dict[str, List[str]]:
    """Extract room restrictions from agent_secrets text as a fallback.

    Parses patterns like "you cannot enter kitchen_1, bedroom_1, or bathroom_1"
    from agent secret strings.  Only returns rooms that are declared as room
    objects in the problem so we don't inject bogus predicates.
    """
    declared_rooms = {
        name for name, typ in problem.objects.items() if typ == "room"
    }
    restrictions: Dict[str, List[str]] = {}
    secrets = getattr(task, "agent_secrets", {}) or {}
    for agent_id, secret_list in secrets.items():
        if not isinstance(secret_list, list):
            continue
        text = " ".join(str(s) for s in secret_list)
        # Match "cannot enter room_1, room_2, and/or room_3"
        for m in re.finditer(
            r"cannot enter\s+([\w_]+(?:[\s,]+(?:and\s+|or\s+)?[\w_]+)*)",
            text,
        ):
            raw = m.group(1)
            # Split on commas, "and", "or", whitespace
            tokens = re.split(r"[,\s]+(?:and\s+|or\s+)?", raw)
            for tok in tokens:
                tok = tok.strip().rstrip(".")
                if tok in declared_rooms:
                    restrictions.setdefault(agent_id, []).append(tok)
    return restrictions


def _ensure_room_restrictions(task: "GeneratedTask", problem: Problem) -> None:
    """Populate is_restricted predicates from mechanic bindings and agent secrets.

    Primary source: room_restriction mechanic bindings.
    Fallback: parse "cannot enter" patterns from agent_secrets text.
    This ensures the planner respects room restrictions even when the
    task-gen agent only wrote them in natural language instructions.
    """
    existing = {(l.predicate, l.args) for l in problem.init}

    def _add(agent: str, room: str) -> None:
        fact = ("is_restricted", (agent, room))
        if fact not in existing:
            problem.init.append(Literal("is_restricted", (agent, room)))
            existing.add(fact)

    # Primary: mechanic_bindings
    for binding in getattr(task, "mechanic_bindings", []) or []:
        if _binding_value(binding, "mechanic_type") != "room_restriction":
            continue
        rooms = _binding_value(binding, "restricted_rooms") or []
        agents = _binding_value(binding, "for_agents") or []
        for agent in agents:
            if not isinstance(agent, str):
                continue
            for room in rooms:
                if not isinstance(room, str):
                    continue
                _add(agent, room)

    # Fallback: extract from agent_secrets natural language
    for agent, rooms in _extract_restrictions_from_secrets(task, problem).items():
        for room in rooms:
            _add(agent, room)


def _ensure_scene_grounding(problem: Problem, scene_data: Dict[str, Any]) -> None:
    """Auto-populate missing grounding facts from scene data.

    Injects agent_in_room, is_in_room, and is_on_top facts that exist in
    the scene but were omitted from the authored problem_pddl.  Only adds
    facts for objects already declared in problem.objects so we don't
    introduce undeclared identifiers.
    """
    existing = {(l.predicate, l.args) for l in problem.init}
    declared = set(problem.objects.keys())

    def _add(predicate: str, args: tuple) -> None:
        if (predicate, args) not in existing:
            problem.init.append(Literal(predicate, args))
            existing.add((predicate, args))

    # Build furniture -> room and object -> (room, furniture) maps from scene
    furniture_to_room: Dict[str, str] = {}
    for room, furns in (scene_data.get("furniture_in_rooms") or {}).items():
        if not isinstance(furns, list):
            continue
        for furn in furns:
            if isinstance(furn, str):
                furniture_to_room[furn] = room

    object_to_furniture: Dict[str, str] = {}
    for furn, objs in (scene_data.get("objects_on_furniture") or {}).items():
        if not isinstance(objs, list):
            continue
        for obj in objs:
            if isinstance(obj, str):
                object_to_furniture[obj] = furn

    # 1. agent_in_room from agent_spawns
    agent_spawns = scene_data.get("agent_spawns") or {}
    for agent_id, room_id in agent_spawns.items():
        if agent_id in declared and isinstance(room_id, str):
            _add("agent_in_room", (agent_id, room_id))

    # 2. is_in_room for furniture
    for furn_id, room_id in furniture_to_room.items():
        if furn_id in declared:
            _add("is_in_room", (furn_id, room_id))

    # 3. is_in_room for objects (derived from furniture location)
    for obj_id, furn_id in object_to_furniture.items():
        if obj_id not in declared:
            continue
        room_id = furniture_to_room.get(furn_id)
        if room_id:
            _add("is_in_room", (obj_id, room_id))
        # Also add is_on_top so the planner knows the starting position
        if furn_id in declared:
            _add("is_on_top", (obj_id, furn_id))


def _ensure_can_communicate(task: "GeneratedTask", problem: Problem) -> None:
    """Populate can_communicate predicates in problem init.

    - Default: all agent pairs can communicate (can_communicate a_i a_j for i!=j).
    - restricted_communication mechanic: only allowed pairs.
    - message_targets: same constraint source.

    Skips if can_communicate is already present in init (authored in problem_pddl).
    """
    existing = {(l.predicate, l.args) for l in problem.init}
    has_can_comm = any(pred == "can_communicate" for pred, _ in existing)
    if has_can_comm:
        return  # Already authored

    num_agents = task.num_agents
    all_agents = [f"agent_{i}" for i in range(num_agents)]

    # Check for restricted_communication mechanic
    restricted_targets = None
    for binding in task.mechanic_bindings:
        if _binding_value(binding, "mechanic_type") == "restricted_communication":
            allowed_targets = _binding_value(binding, "allowed_targets")
            if allowed_targets:
                restricted_targets = allowed_targets
            break

    # Check for message_targets
    message_targets = task.message_targets

    if isinstance(restricted_targets, dict):
        # Only add allowed pairs from allowed_targets dict.
        for agent, targets in restricted_targets.items():
            if not isinstance(agent, str):
                continue
            if not isinstance(targets, (list, tuple, set)):
                continue
            for target in targets:
                if isinstance(target, str):
                    problem.init.append(Literal("can_communicate", (agent, target)))
    elif message_targets:
        # message_targets: agent -> list of targets
        for agent, targets in message_targets.items():
            for target in targets:
                problem.init.append(Literal("can_communicate", (agent, target)))
    else:
        # Default: all pairs
        for a in all_agents:
            for b in all_agents:
                if a != b:
                    problem.init.append(Literal("can_communicate", (a, b)))
