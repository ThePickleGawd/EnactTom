"""Shared authoring-surface configuration for task generation."""

from __future__ import annotations

from typing import Iterable, List


SUPPORTED_AUTHORING_MECHANICS = [
    "room_restriction",
    "limited_bandwidth",
    "restricted_communication",
    "remote_control",
    "state_mirroring",
    "inverse_state",
]

SUPPORTED_AUTHORING_PREDICATES = {
    "is_on_top",
    "is_inside",
    "is_in_room",
    "is_on_floor",
    "is_next_to",
    "is_open",
    "is_closed",
    "is_clean",
    "is_dirty",
    "is_filled",
    "is_empty",
    "is_powered_on",
    "is_powered_off",
    "is_unlocked",
    "is_locked",
    "is_held_by",
    "agent_in_room",
    "is_inverse",
    "mirrors",
    "controls",
    "controls_unlocked",
    "controls_closed",
    "controls_locks",
    "is_restricted",
    "can_communicate",
}

AUTHORING_CONSTRAINTS_NOTICE = (
    "The paper authoring surface uses scene objects only; do not add new "
    "runtime schemas beyond problem_pddl, agent_secrets, mechanics, and "
    "golden trajectory fields."
)


def get_authoring_default_actions(*, include_find_tools: bool) -> List[str]:
    actions = ["Navigate", "Open", "Close", "Pick", "Place"]
    if include_find_tools:
        actions.extend(["FindObjectTool", "FindReceptacleTool", "FindRoomTool"])
    actions.extend(["Communicate", "Wait"])
    return actions


def get_authoring_action_descriptions() -> str:
    from enacttom.actions import ActionRegistry

    descriptions = ActionRegistry.get_all_action_descriptions().splitlines()
    return "\n".join(descriptions)


def get_authoring_mechanics() -> str:
    from enacttom.mechanics import get_mechanics_for_task_generation

    return get_mechanics_for_task_generation(
        visible_mechanics=SUPPORTED_AUTHORING_MECHANICS
    )


def get_authoring_predicates() -> str:
    from enacttom.pddl.domain import get_predicates_for_prompt

    return get_predicates_for_prompt(
        allowed_predicates=SUPPORTED_AUTHORING_PREDICATES
    )


def format_supported_mechanics(names: Iterable[str] | None = None) -> str:
    mechanic_names = list(names or SUPPORTED_AUTHORING_MECHANICS)
    return ", ".join(mechanic_names)
