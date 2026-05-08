from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _scene_to_dict(scene_data: Any) -> Dict[str, Any]:
    if isinstance(scene_data, dict):
        return scene_data
    if hasattr(scene_data, "to_dict"):
        return scene_data.to_dict()
    return {}


def _resolve_agent_room(scene: Dict[str, Any], rooms: List[str], agent_id: str, agent_idx: int) -> str | None:
    agent_spawns = scene.get("agent_spawns") or {}
    spawn = agent_spawns.get(agent_id)
    if isinstance(spawn, str) and spawn in rooms:
        return spawn
    if rooms:
        return rooms[agent_idx % len(rooms)]
    return None


def build_scene_bootstrap_problem_pddl(
    scene_data: Any,
    num_agents: int,
    *,
    problem_name: str = "task_problem",
    relevant_ids: Optional[Set[str]] = None,
) -> str:
    scene = _scene_to_dict(scene_data)
    all_rooms = _ordered_unique(scene.get("rooms") or [])
    articulated = set(_ordered_unique(scene.get("articulated_furniture") or []))

    furniture_to_room: Dict[str, str] = {}
    ordered_furniture: List[str] = []
    for room_id in all_rooms:
        room_furniture = (scene.get("furniture_in_rooms") or {}).get(room_id) or []
        for furniture_id in _ordered_unique(room_furniture):
            if furniture_id in furniture_to_room:
                continue
            furniture_to_room[furniture_id] = room_id
            ordered_furniture.append(furniture_id)

    object_to_furniture: Dict[str, str] = {}
    ordered_objects: List[str] = []
    for furniture_id in ordered_furniture:
        furniture_objects = (scene.get("objects_on_furniture") or {}).get(furniture_id) or []
        for object_id in _ordered_unique(furniture_objects):
            if object_id in object_to_furniture:
                continue
            object_to_furniture[object_id] = furniture_id
            ordered_objects.append(object_id)

    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    agent_rooms = {
        agent_id: _resolve_agent_room(scene, all_rooms, agent_id, idx)
        for idx, agent_id in enumerate(agent_ids)
    }

    rooms = list(all_rooms)
    if relevant_ids:
        selected_objects = {obj_id for obj_id in ordered_objects if obj_id in relevant_ids}
        selected_furniture = {furniture_id for furniture_id in ordered_furniture if furniture_id in relevant_ids}
        selected_rooms = {room_id for room_id in all_rooms if room_id in relevant_ids}

        for object_id in list(selected_objects):
            furniture_id = object_to_furniture.get(object_id)
            if furniture_id:
                selected_furniture.add(furniture_id)

        for furniture_id in list(selected_furniture):
            room_id = furniture_to_room.get(furniture_id)
            if room_id:
                selected_rooms.add(room_id)

        for room_id in agent_rooms.values():
            if room_id:
                selected_rooms.add(room_id)

        ordered_furniture = [furniture_id for furniture_id in ordered_furniture if furniture_id in selected_furniture]
        ordered_objects = [object_id for object_id in ordered_objects if object_id in selected_objects]
        rooms = [room_id for room_id in all_rooms if room_id in selected_rooms]

    lines = [
        f"(define (problem {problem_name})",
        "  (:domain enacttom)",
        "  (:objects",
        f"    {' '.join(agent_ids)} - agent",
    ]
    if rooms:
        lines.append(f"    {' '.join(rooms)} - room")
    if ordered_objects:
        lines.append(f"    {' '.join(ordered_objects)} - object")
    if ordered_furniture:
        lines.append(f"    {' '.join(ordered_furniture)} - furniture")
    lines.extend(
        [
            "  )",
            "  (:init",
        ]
    )

    for agent_id in agent_ids:
        room_id = agent_rooms.get(agent_id)
        if room_id:
            lines.append(f"    (agent_in_room {agent_id} {room_id})")

    for furniture_id in ordered_furniture:
        room_id = furniture_to_room[furniture_id]
        lines.append(f"    (is_in_room {furniture_id} {room_id})")

    for object_id in ordered_objects:
        furniture_id = object_to_furniture[object_id]
        room_id = furniture_to_room.get(furniture_id)
        if room_id:
            lines.append(f"    (is_in_room {object_id} {room_id})")
        lines.append(f"    (is_on_top {object_id} {furniture_id})")

    goal_literals: List[str] = []
    support_surfaces = [
        furniture_id for furniture_id in ordered_furniture if furniture_id not in articulated
    ]
    if ordered_objects and support_surfaces:
        goal_object = ordered_objects[0]
        current_parent = object_to_furniture.get(goal_object)
        target_furniture = next(
            (
                furniture_id
                for furniture_id in support_surfaces
                if furniture_id != current_parent
            ),
            support_surfaces[0],
        )
        goal_literals.append(f"(is_on_top {goal_object} {target_furniture})")

    open_target = next((furniture_id for furniture_id in ordered_furniture if furniture_id in articulated), None)
    if open_target:
        goal_literals.append(f"(is_open {open_target})")

    if not goal_literals:
        fallback_agent = agent_ids[0] if agent_ids else None
        fallback_room = agent_rooms.get(fallback_agent) if fallback_agent else None
        if fallback_agent and fallback_room:
            goal_literals.append(f"(agent_in_room {fallback_agent} {fallback_room})")
        elif rooms:
            goal_literals.append(f"(is_in_room {ordered_furniture[0]} {rooms[0]})" if ordered_furniture else f"(agent_in_room agent_0 {rooms[0]})")
        else:
            goal_literals.append("(and)")

    if len(goal_literals) == 1:
        goal_pddl = goal_literals[0]
    else:
        goal_pddl = "(and " + " ".join(goal_literals) + ")"

    lines.extend(
        [
            "  )",
            f"  (:goal {goal_pddl})",
            ")",
        ]
    )
    return "\n".join(lines)


def _append_goal_owners(problem_pddl: str, owners: Dict[str, str]) -> str:
    if not owners:
        return problem_pddl
    if not problem_pddl.rstrip().endswith(")"):
        return problem_pddl

    owner_lines = ["  (:goal-owners"]
    for formula_pddl, owner_id in owners.items():
        owner_lines.append(f"    ({owner_id} {formula_pddl})")
    owner_lines.append("  )")
    owner_block = "\n".join(owner_lines)
    stripped = problem_pddl.rstrip()
    return stripped[:-1] + owner_block + "\n)"


def canonicalize_problem_pddl_with_scene(
    task_data: Dict[str, Any],
    scene_data: Any,
) -> str:
    """Rebuild authored problem_pddl from scene defaults while preserving goals."""
    from enacttom.pddl.problem_pddl import (
        collect_object_ids_from_formula,
        parse_goal_string,
        parse_problem_pddl,
        replace_goal_in_problem_pddl,
    )

    raw_problem = task_data.get("problem_pddl")
    if not isinstance(raw_problem, str) or not raw_problem.strip():
        raise ValueError("Task must define non-empty problem_pddl.")

    parsed = parse_problem_pddl(raw_problem)
    relevant_ids: Set[str] = set(collect_object_ids_from_formula(parsed.goal_formula))
    for formula_pddl in (parsed.owners or {}).keys():
        try:
            owner_formula = parse_goal_string(formula_pddl)
        except ValueError:
            continue
        relevant_ids.update(collect_object_ids_from_formula(owner_formula))

    for binding in task_data.get("mechanic_bindings") if isinstance(task_data.get("mechanic_bindings"), list) else []:
        if not isinstance(binding, dict):
            continue
        for key in ("trigger_object", "target_object", "prerequisite_object"):
            value = binding.get(key)
            if isinstance(value, str) and value:
                relevant_ids.add(value)
        for key in ("restricted_rooms", "for_agents"):
            values = binding.get(key) or []
            if isinstance(values, list):
                relevant_ids.update(v for v in values if isinstance(v, str) and v)
        for key in ("allowed_targets", "message_limits"):
            mapping = binding.get(key) or {}
            if not isinstance(mapping, dict):
                continue
            relevant_ids.update(k for k in mapping.keys() if isinstance(k, str) and k)
            for value in mapping.values():
                if isinstance(value, list):
                    relevant_ids.update(v for v in value if isinstance(v, str) and v)

    message_targets = task_data.get("message_targets") or {}
    if isinstance(message_targets, dict):
        relevant_ids.update(k for k in message_targets.keys() if isinstance(k, str) and k)
        for value in message_targets.values():
            if isinstance(value, list):
                relevant_ids.update(v for v in value if isinstance(v, str) and v)

    num_agents = int(task_data.get("num_agents", 2) or 2)
    relevant_ids.update(f"agent_{idx}" for idx in range(num_agents))
    bootstrap = build_scene_bootstrap_problem_pddl(
        scene_data,
        num_agents,
        problem_name=parsed.problem_name,
        relevant_ids=relevant_ids,
    )
    rebuilt = replace_goal_in_problem_pddl(bootstrap, parsed.goal_pddl)
    return _append_goal_owners(rebuilt, parsed.owners or {})


def canonicalize_task_problem_pddl(
    task_data: Dict[str, Any],
    scene_data: Any,
) -> bool:
    """Normalize problem_pddl against the loaded scene. Returns True if changed."""
    if not scene_data:
        return False
    current = task_data.get("problem_pddl")
    if not isinstance(current, str) or not current.strip():
        return False
    canonical = canonicalize_problem_pddl_with_scene(task_data, scene_data)
    if canonical == current:
        return False
    task_data["problem_pddl"] = canonical
    return True
