"""Habitat tool filtering for EnactToM benchmark agents."""

from typing import List, Optional


def wrap_habitat_tools(
    agent,
    game_manager,
    allowed_actions: Optional[List[str]] = None,
) -> None:
    """Filter an agent's Habitat tools to the task-declared action set."""
    del game_manager

    if allowed_actions is None:
        return

    effective_allowed = set(allowed_actions)
    if "Open" in effective_allowed or "Close" in effective_allowed:
        effective_allowed.update({"Open", "Close"})

    for name in list(agent.tools.keys()):
        if name not in effective_allowed:
            del agent.tools[name]
