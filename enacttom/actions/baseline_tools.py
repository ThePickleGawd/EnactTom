"""
Baseline-only benchmark tools.

These tools are injected at runtime for EnactToM baseline benchmark mode and are not
part of the normal task action space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from habitat_llm.tools.tool import PerceptionTool


class ReadAgentTrajectoryTool(PerceptionTool):
    """
    Read another agent's completed trajectory in baseline benchmark mode.

    The tool can expose completed prior-turn traces as Thought + Action, or
    Observation + Thought + Action when observation traces are enabled.
    """

    def __init__(self, agent_uid: int = 0, include_observations: bool = False):
        super().__init__("ReadAgentTrajectoryTool", agent_uid_arg=agent_uid)
        self._trajectory_store: Optional[Sequence[Dict[str, Any]]] = None
        self._include_observations = include_observations

    def set_trajectory_store(self, trajectory_store: Sequence[Dict[str, Any]]) -> None:
        self._trajectory_store = trajectory_store

    def get_state_description(self) -> str:
        return "Reviewing teammate trajectory"

    @property
    def description(self) -> str:
        if self._include_observations:
            return (
                "ReadAgentTrajectoryTool[agent_id]: Read another agent's completed "
                "trajectory as Observation + Thought + Action entries from prior "
                "turns only. Example: ReadAgentTrajectoryTool[agent_1]"
            )
        return (
            "ReadAgentTrajectoryTool[agent_id]: Read another agent's completed "
            "trajectory as Thought + Action pairs from prior turns only. "
            "Example: ReadAgentTrajectoryTool[agent_1]"
        )

    @property
    def argument_types(self) -> List[str]:
        return ["AGENT_INSTANCE"]

    def process_high_level_action(
        self,
        input_query: str,
        observations: dict,
    ) -> Tuple[None, str]:
        target = str(input_query or "").strip()
        if not target:
            return None, "Specify an agent ID, for example ReadAgentTrajectoryTool[agent_1]."

        if not target.startswith("agent_"):
            return None, f"Invalid agent ID '{target}'. Use format agent_N."

        if self._trajectory_store is None:
            return None, "Trajectory store is unavailable."

        records = [
            entry for entry in self._trajectory_store
            if entry.get("agent_id") == target
        ]
        if not records:
            return None, f"No completed trajectory is available for {target} yet."

        lines = [f"{target} completed trajectory:"]
        for entry in records:
            turn = entry.get("turn", "?")
            observation = str(entry.get("observation") or "").strip()
            raw_thought = str(entry.get("thought") or "None")
            thought = raw_thought
            if raw_thought.startswith("Thought:"):
                thought = raw_thought.split("Thought:", 1)[1].strip() or "None"
            action = str(entry.get("action") or "")
            if self._include_observations and observation:
                lines.append(f"Turn {turn}: Observation: {observation}")
            lines.append(f"Turn {turn}: Thought: {thought}")
            lines.append(f"Turn {turn}: Action: {action}")

        return None, "\n".join(lines)
