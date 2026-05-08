"""Small Habitat world-graph adapter used by EnactToM runners."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


class HabitatWorldAdapter:
    """Expose the Habitat world graph through the fields EnactToM needs."""

    def __init__(self, env_interface: "EnvironmentInterface", agent_uid: int = 0):
        self.env = env_interface
        self.agent_uid = agent_uid

    @property
    def world_graph(self):
        return self.env.world_graph[self.agent_uid]

    @property
    def full_world_graph(self):
        return self.env.full_world_graph

    def get_all_objects(self) -> List[Any]:
        return self.full_world_graph.get_all_objects()

    def get_all_furniture(self) -> List[Any]:
        return self.full_world_graph.get_all_furnitures()

    def get_all_rooms(self) -> List[Any]:
        return self.full_world_graph.get_all_rooms()

    def get_interactable_entities(self) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []

        for furniture in self.get_all_furniture():
            entities.append(
                {
                    "id": getattr(furniture, "sim_handle", furniture.name),
                    "name": furniture.name,
                    "type": "furniture",
                    "states": self._get_entity_states(furniture),
                    "is_articulated": (
                        furniture.is_articulated()
                        if hasattr(furniture, "is_articulated")
                        else False
                    ),
                    "properties": getattr(furniture, "properties", {}),
                    "room": self._get_room_name(furniture),
                }
            )

        for obj in self.get_all_objects():
            entities.append(
                {
                    "id": getattr(obj, "sim_handle", obj.name),
                    "name": obj.name,
                    "type": "object",
                    "states": self._get_entity_states(obj),
                    "is_articulated": False,
                    "properties": getattr(obj, "properties", {}),
                    "room": self._get_room_name(obj),
                }
            )

        return entities

    def _get_room_name(self, entity: Any) -> Optional[str]:
        try:
            room = self.full_world_graph.get_room_for_entity(entity)
            return room.name if hasattr(room, "name") else str(room)
        except Exception:
            return None

    def _get_entity_states(self, entity: Any) -> Dict[str, Any]:
        states: Dict[str, Any] = {}
        props = getattr(entity, "properties", {})

        state_keys = [
            "is_open",
            "is_closed",
            "is_on",
            "is_off",
            "is_powered_on",
            "is_powered_off",
            "is_filled",
            "is_clean",
        ]

        for key in state_keys:
            if key in props:
                states[key] = props[key]

        if "states" in props:
            states.update(props["states"])

        return states

    def get_room_ids(self) -> List[str]:
        return [room.name for room in self.get_all_rooms()]

    def get_agent_location(self, agent_id: str) -> Optional[str]:
        from habitat_llm.world_model.entity import Room

        try:
            agent_name = agent_id if "agent" in agent_id else f"agent_{agent_id}"
            agent_node = self.full_world_graph.get_node_from_name(agent_name)
            if agent_node:
                neighbors = self.full_world_graph.get_neighbors_of_type(agent_node, Room)
                if neighbors:
                    return neighbors[0].name
        except Exception:
            pass
        return None

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for entity in self.get_interactable_entities():
            if entity["name"] == name or entity["id"] == name:
                return entity
        return None
