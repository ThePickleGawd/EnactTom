#!/usr/bin/env python3

"""Partial-observation world-graph action updates for EnactToM."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from habitat_llm.world_model import Entity, Furniture, Object, WorldGraph
from habitat_llm.world_model.world_graph import flip_edge


class DynamicWorldGraph(WorldGraph):
    """World graph used by partial-observation EnactToM runs.

    The graph contents come from simulator-derived subgraphs. This subclass only
    keeps the deterministic action-result updates needed while an object is
    carried and temporarily absent from perception.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger(__name__)
        self._articulated_agents: dict = {}

    def set_articulated_agents(self, articulated_agent: dict):
        self._articulated_agents = articulated_agent

    def move_object_from_agent_to_placement_node(
        self,
        object_node: Entity,
        agent_node: Entity,
        placement_node: Furniture,
        verbose: bool = True,
    ) -> None:
        """Move an object from an agent node onto a placement node."""
        self.remove_edge(object_node, agent_node)
        self.add_edge(object_node, placement_node, "on", flip_edge("on"))
        if "translation" in placement_node.properties:
            object_node.properties["translation"] = placement_node.properties["translation"]
        if verbose:
            self._logger.info(
                "Moved %s from %s to %s",
                object_node.name,
                agent_node.name,
                placement_node.name,
            )

    def _get_node(self, name: str) -> Optional[Entity]:
        try:
            return self.get_node_from_name(name.strip())
        except ValueError:
            return None

    def _parse_place_args(self, action_args: str) -> tuple[Optional[Object], Optional[Furniture]]:
        parts = [part.strip() for part in str(action_args).split(",")]
        if len(parts) < 3:
            return None, None
        object_node = self._get_node(parts[0])
        placement_node = self._get_node(parts[2])
        if not isinstance(object_node, Object):
            return None, None
        if not isinstance(placement_node, Furniture):
            return object_node, None
        return object_node, placement_node

    def _set_object_state(self, object_name: str, state_name: str, value: bool) -> None:
        object_node = self._get_node(object_name)
        if object_node is not None:
            object_node.set_state({state_name: value})

    def _update_successful_action(
        self,
        agent_uid: int,
        high_level_action: Tuple[str, str, Optional[str]],
        verbose: bool = False,
    ) -> None:
        action_name = str(high_level_action[0]).lower()
        action_args = str(high_level_action[1] or "")
        agent_node = self._get_node(f"agent_{agent_uid}")
        if agent_node is None:
            return

        if "place" in action_name or "rearrange" in action_name:
            object_node, placement_node = self._parse_place_args(action_args)
            if object_node is not None and placement_node is not None:
                self.move_object_from_agent_to_placement_node(
                    object_node,
                    agent_node,
                    placement_node,
                    verbose=verbose,
                )
            return

        if "pour" in action_name or "fill" in action_name:
            self._set_object_state(action_args, "is_filled", True)
            return

        if "power" in action_name:
            if "on" in action_name:
                self._set_object_state(action_args, "is_powered_on", True)
            elif "off" in action_name:
                self._set_object_state(action_args, "is_powered_on", False)
            return

        if "clean" in action_name:
            self._set_object_state(action_args, "is_clean", True)

    def update_by_action(
        self,
        agent_uid: int,
        high_level_action: Tuple[str, str, Optional[str]],
        action_response: str,
        verbose: bool = False,
    ) -> None:
        """Update this agent's graph after its own successful action."""
        if isinstance(action_response, str) and "success" in action_response.lower():
            self._update_successful_action(int(agent_uid), high_level_action, verbose)

    def update_by_other_agent_action(
        self,
        other_agent_uid: int,
        high_level_action_and_args: Tuple[str, str, Optional[str]],
        action_results: str,
        use_semantic_similarity: bool = False,
        verbose: bool = False,
    ) -> None:
        """Update this graph after another agent's successful action.

        EnactToM uses shared simulator identifiers, so no semantic remapping is
        needed between agents' graph nodes.
        """
        if use_semantic_similarity:
            raise NotImplementedError("Semantic remapping is not supported.")
        if isinstance(action_results, str) and "success" in action_results.lower():
            self._update_successful_action(
                int(other_agent_uid),
                high_level_action_and_args,
                verbose,
            )
