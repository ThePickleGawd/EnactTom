"""Helpers for exposing seed/example tasks safely under literal-ToM semantics."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Set


_TITLE_PLACEHOLDER = "Rewrite Title For Current Scene"
_TASK_PLACEHOLDER = (
    "Rewrite the public task from scratch for the current scene. Describe only "
    "the functional shared objective in natural language. Do not mention K() "
    "goals, probes, or knowledge as a pass/fail requirement."
)
_AGENT_SECRET_PLACEHOLDER = (
    "Rewrite from scratch for the current scene. Mention agent-specific "
    "access/communication constraints, mechanic hints, and relevant private "
    "information. Use exact scene IDs for goal-critical targets. Do not say "
    "that knowledge is required for task success."
)

_DERIVED_FIELDS_TO_DROP = (
    "functional_goal_pddl",
    "literal_tom_probes",
    "runtime_semantics_version",
    "golden_trajectory",
    "tom_level",
    "tom_reasoning",
    "calibration",
    "judge",
    "benchmark_results",
)


def _rewrite_agent_ids_in_bindings(
    bindings: List[Dict[str, Any]],
    valid_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Rewrite mechanic_bindings so they only reference valid agent IDs.

    - for_agents: filter to valid IDs only.
    - message_limits / allowed_targets: re-key to valid IDs only.
    - Bindings that become empty after filtering are dropped.
    """
    result: List[Dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        b = dict(binding)

        # for_agents (room_restriction, etc.)
        if "for_agents" in b and isinstance(b["for_agents"], list):
            b["for_agents"] = [a for a in b["for_agents"] if a in valid_ids]
            if not b["for_agents"]:
                continue  # binding has no valid agents left — drop it

        # message_limits (limited_bandwidth)
        if "message_limits" in b and isinstance(b["message_limits"], dict):
            b["message_limits"] = {
                a: v for a, v in b["message_limits"].items() if a in valid_ids
            }
            if not b["message_limits"]:
                continue

        # allowed_targets (restricted_communication)
        if "allowed_targets" in b and isinstance(b["allowed_targets"], dict):
            b["allowed_targets"] = {
                a: [t for t in targets if t in valid_ids]
                for a, targets in b["allowed_targets"].items()
                if a in valid_ids and isinstance(targets, list)
            }
            if not b["allowed_targets"]:
                continue

        result.append(b)
    return result


def _rewrite_message_targets(
    mt: Dict[str, Any],
    valid_ids: Set[str],
) -> Dict[str, List[str]]:
    """Filter message_targets to valid agent IDs only."""
    result: Dict[str, List[str]] = {}
    for agent, targets in mt.items():
        if agent not in valid_ids or not isinstance(targets, list):
            continue
        filtered = [t for t in targets if isinstance(t, str) and t in valid_ids]
        if filtered:
            result[agent] = filtered
    return result


def sanitize_task_for_seeding(
    task_data: Dict[str, Any],
    *,
    num_agents: int | None = None,
) -> Dict[str, Any]:
    """Return a seed-safe task copy with natural-language fields reset.

    The generator still gets structural fields such as category, mechanics,
    communication graph, and PDDL. Public/secret language and derived runtime
    metadata are cleared so stale semantics do not leak into new generations.

    All agent-ID-bearing structural fields (mechanic_bindings, message_targets,
    agent_actions) are rewritten to only reference agent_0..agent_{N-1}.
    problem_pddl is cleared since it references the seed's scene objects.
    """

    sanitized = deepcopy(task_data)

    for key in _DERIVED_FIELDS_TO_DROP:
        sanitized.pop(key, None)

    if num_agents is None:
        raw_num_agents = sanitized.get("num_agents")
        num_agents = raw_num_agents if isinstance(raw_num_agents, int) and raw_num_agents > 0 else 2

    valid_ids = {f"agent_{i}" for i in range(num_agents)}

    sanitized["title"] = _TITLE_PLACEHOLDER
    sanitized["task"] = _TASK_PLACEHOLDER

    sanitized["agent_secrets"] = {
        f"agent_{idx}": [_AGENT_SECRET_PLACEHOLDER]
        for idx in range(num_agents)
    }

    # Rewrite mechanic_bindings to only reference valid agents.
    bindings = sanitized.get("mechanic_bindings")
    if isinstance(bindings, list):
        sanitized["mechanic_bindings"] = _rewrite_agent_ids_in_bindings(
            bindings, valid_ids
        )

    # Rewrite message_targets.
    mt = sanitized.get("message_targets")
    if isinstance(mt, dict):
        sanitized["message_targets"] = _rewrite_message_targets(mt, valid_ids)

    # Rewrite agent_actions.
    agent_actions = sanitized.get("agent_actions")
    if isinstance(agent_actions, dict):
        # Keep action lists from lowest-numbered valid agents; fill missing.
        existing_actions = [
            v for k, v in sorted(agent_actions.items()) if isinstance(v, list)
        ]
        default_actions = existing_actions[0] if existing_actions else []
        sanitized["agent_actions"] = {
            f"agent_{i}": (
                agent_actions.get(f"agent_{i}", default_actions[:])
                if isinstance(agent_actions.get(f"agent_{i}"), list)
                else default_actions[:]
            )
            for i in range(num_agents)
        }

    # Clear problem_pddl — it references the seed scene's object/furniture IDs
    # which won't match the new scene. The agent must author it from scratch.
    sanitized.pop("problem_pddl", None)

    sanitized["num_agents"] = num_agents

    return sanitized
