"""
Epistemic layer for EnactToM PDDL.

Derives observability models from task structure (room restrictions,
mechanic bindings) to determine what each agent can/cannot observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from enacttom.task_gen.task_generator import normalize_mechanic_bindings

if TYPE_CHECKING:
    from enacttom.pddl.dsl import Literal
    from enacttom.task_gen.task_generator import GeneratedTask


@dataclass
class ObservabilityModel:
    """
    Models what each agent can and cannot observe.

    Derived automatically from room_restrictions + mechanic_bindings.
    Used to construct epistemic init states and compute ToM depth.
    """

    # agent -> rooms they cannot enter/see
    restricted_rooms: Dict[str, Set[str]] = field(default_factory=dict)

    # mechanic effects that are hidden from some agents
    # Maps: trigger_object -> set of agents who can't observe the effect
    hidden_effects: Dict[str, Set[str]] = field(default_factory=dict)

    # agent -> set of agents they can message (None = unrestricted)
    message_targets: Dict[str, Optional[Set[str]]] = field(default_factory=dict)

    # agent -> message limit (None = unlimited)
    message_limits: Dict[str, Optional[int]] = field(default_factory=dict)

    # object/furniture/room -> room it's in (populated from scene data)
    object_rooms: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: "GeneratedTask") -> "ObservabilityModel":
        """Build observability model from a GeneratedTask."""
        model = cls()
        num_agents = task.num_agents
        all_agents = {f"agent_{i}" for i in range(num_agents)}

        raw_bindings: List[Dict[str, Any]] = []
        for binding in task.mechanic_bindings:
            if isinstance(binding, dict):
                raw_bindings.append(binding)
                continue
            if hasattr(binding, "to_dict"):
                try:
                    candidate = binding.to_dict()
                    if isinstance(candidate, dict):
                        raw_bindings.append(candidate)
                        continue
                except Exception:
                    pass
            mechanic_type = getattr(binding, "mechanic_type", None)
            if isinstance(mechanic_type, str):
                raw_bindings.append(
                    {
                        "mechanic_type": mechanic_type,
                        "trigger_object": getattr(binding, "trigger_object", None),
                        "restricted_rooms": getattr(binding, "restricted_rooms", None),
                        "for_agents": getattr(binding, "for_agents", None),
                        "message_limits": getattr(binding, "message_limits", None),
                        "allowed_targets": getattr(binding, "allowed_targets", None),
                    }
                )

        normalized_bindings = normalize_mechanic_bindings(
            raw_bindings,
            problem_pddl=getattr(task, "problem_pddl", None),
        )
        for binding in normalized_bindings:
            mtype = binding.get("mechanic_type")

            if mtype == "room_restriction":
                rooms = set(binding.get("restricted_rooms") or [])
                for agent in (binding.get("for_agents") or []):
                    model.restricted_rooms.setdefault(agent, set()).update(rooms)

            elif mtype in ("remote_control", "state_mirroring"):
                trigger = binding.get("trigger_object")
                if trigger:
                    model.hidden_effects[trigger] = set(all_agents)

            elif mtype == "limited_bandwidth":
                ml = binding.get("message_limits") or {}
                for agent_id, limit in ml.items():
                    if isinstance(limit, (int, float)):
                        model.message_limits[agent_id] = int(limit)

            elif mtype == "restricted_communication":
                allowed_targets = binding.get("allowed_targets") or {}
                for agent_id, targets in allowed_targets.items():
                    if isinstance(targets, list):
                        model.message_targets[agent_id] = set(targets)

        # Message targets from task
        if task.message_targets:
            for agent_id, targets in task.message_targets.items():
                model.message_targets[agent_id] = set(targets)

        return model

    @classmethod
    def from_task_with_scene(
        cls,
        task: "GeneratedTask",
        scene_data: Optional[Dict[str, Any]] = None,
    ) -> "ObservabilityModel":
        """Build observability model from task PDDL, with scene fallback only."""
        model = cls.from_task(task)

        problem_pddl = getattr(task, "problem_pddl", None)
        if isinstance(problem_pddl, str) and problem_pddl.strip():
            try:
                from enacttom.pddl.problem_pddl import (
                    build_object_room_map_from_problem,
                    parse_problem_pddl,
                )

                parsed_problem = parse_problem_pddl(problem_pddl)
                model.object_rooms = build_object_room_map_from_problem(parsed_problem)
            except Exception:
                model.object_rooms = {}

        if not model.object_rooms and scene_data:
            model.object_rooms = _build_object_room_map(scene_data)

        return model

    def agent_can_observe_room(self, agent: str, room: str) -> bool:
        """Check if an agent can observe events in a room."""
        return room not in self.restricted_rooms.get(agent, set())

    def agent_can_observe_effect(self, agent: str, trigger_object: str) -> bool:
        """Check if an agent can observe the effect of a trigger."""
        hidden = self.hidden_effects.get(trigger_object, set())
        return agent not in hidden

    def get_unobservable_agents(self, room: str) -> Set[str]:
        """Get agents that cannot observe events in a room."""
        result = set()
        for agent, rooms in self.restricted_rooms.items():
            if room in rooms:
                result.add(agent)
        return result

    def has_information_asymmetry(self) -> bool:
        """Check if the task has any information asymmetry between agents."""
        return bool(self.restricted_rooms or self.hidden_effects)

    def is_fact_observable_by(
        self,
        agent: str,
        predicate: str,
        args: Tuple[str, ...],
    ) -> bool:
        """
        Check if a specific fact is directly observable by an agent.

        A fact is observable if the agent can access the rooms containing
        all entities referenced in the fact. Returns True (conservative)
        if we can't determine an entity's room.
        """
        if not self.restricted_rooms.get(agent):
            return True  # No restrictions — agent can see everything

        agent_restricted = self.restricted_rooms[agent]

        for arg in args:
            # Skip agent references (agents observe their own state)
            if arg.startswith("agent_"):
                continue
            room = self.object_rooms.get(arg)
            if room and room in agent_restricted:
                return False  # Entity is in a room agent can't access

        return True  # All entities are in accessible rooms (or room unknown)

    def is_k_goal_trivial(
        self,
        agent: str,
        inner: "Literal",
    ) -> bool:
        """
        Check if K(agent, literal) is trivially satisfied.

        Trivial means the agent can directly observe all entities
        referenced in the literal. A trivial K() goal doesn't add
        real ToM depth — the agent can just look.

        Returns True if the K() goal is trivial.
        """
        return self.is_fact_observable_by(agent, inner.predicate, inner.args)


def _build_object_room_map(scene_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Build object/furniture -> room mapping from scene data.

    Same logic as spec_validator._build_target_to_room() but takes raw dict.
    """
    obj_rooms: Dict[str, str] = {}

    rooms = scene_data.get("rooms", [])
    furniture_in_rooms = scene_data.get("furniture_in_rooms", {})
    objects_on_furniture = scene_data.get("objects_on_furniture", {})

    # Rooms map to themselves
    for room in rooms:
        if isinstance(room, str):
            obj_rooms[room] = room

    # Furniture -> room
    furniture_to_room: Dict[str, str] = {}
    for room, furns in furniture_in_rooms.items():
        if not isinstance(furns, list):
            continue
        for furn in furns:
            if isinstance(furn, str):
                furniture_to_room[furn] = room
                obj_rooms[furn] = room

    # Objects -> room (via furniture)
    for furn, objs in objects_on_furniture.items():
        room = furniture_to_room.get(furn)
        if not room or not isinstance(objs, list):
            continue
        for obj in objs:
            if isinstance(obj, str):
                obj_rooms[obj] = room

    return obj_rooms
