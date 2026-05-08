"""
EnactToM Game State.

Central state management for all EnactToM-specific state.
This is an overlay on top of Habitat's simulator state.

Design principles:
- Single source of truth for all EnactToM state
- Fully serializable (can checkpoint/restore)
- Mechanics are stateless transforms on this state
- Syncs relevant properties from Habitat each step
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import json
import copy


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ActionRecord:
    """Record of an action taken."""
    step: int
    agent_id: str
    action_name: str
    target: Optional[str]
    success: bool
    observation: str
    effects: List[str] = field(default_factory=list)
    mechanic_applied: Optional[str] = None


@dataclass
class Goal:
    """A task goal to be achieved."""
    goal_id: str
    description: str
    goal_type: str  # e.g., "find_item", "change_state", "reach_location"
    target: Optional[str] = None
    target_state: Optional[Dict[str, Any]] = None
    status: GoalStatus = GoalStatus.PENDING
    completed_at_step: Optional[int] = None
    completed_by: Optional[str] = None


@dataclass
class EnactToMGameState:
    """
    Central game state for EnactToM.

    All EnactToM-specific state lives here. Mechanics are stateless
    transforms that read/write this state.
    """

    # === Synced from Habitat (updated each step) ===
    agent_positions: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    agent_rooms: Dict[str, str] = field(default_factory=dict)
    # Ground truth object states from Habitat (is_open, is_on, etc.)
    object_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # All entities in the scene
    entities: List[Dict[str, Any]] = field(default_factory=list)

    # === Our overlay (custom properties) ===
    # Custom properties per object (is_open, is_locked, inverse, linked_to, etc.)
    object_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # === Mechanic state (owned here, mechanics are stateless) ===
    # state_mirroring: pairs of linked objects [(obj_a, obj_b, state), ...]
    mirror_pairs: List[Tuple[str, str, str]] = field(default_factory=list)
    # inverse_state: objects with inverted behavior
    inverse_objects: Set[str] = field(default_factory=set)
    # remote_control: mappings from trigger -> (target, state)
    remote_mappings: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # Per-agent observation history
    agent_observations: Dict[str, List[str]] = field(default_factory=dict)
    # === Timeline ===
    current_step: int = 0
    action_history: List[ActionRecord] = field(default_factory=list)

    # === Goals ===
    goals: List[Goal] = field(default_factory=list)
    completed_goals: Set[str] = field(default_factory=set)

    # === Mechanic bindings (from task definition) ===
    mechanic_bindings: List[Dict[str, Any]] = field(default_factory=list)
    active_mechanics: List[str] = field(default_factory=list)

    # === Room restriction mechanic ===
    # Maps agent_id -> set of room names they cannot enter
    # Used to force collaboration (agent with info can't access location)
    restricted_rooms: Dict[str, Set[str]] = field(default_factory=dict)

    # === Limited bandwidth mechanic ===
    # Maps agent_id -> max number of Communicate actions allowed
    message_limits: Dict[str, int] = field(default_factory=dict)
    # Maps agent_id -> number of Communicate actions used so far
    messages_sent: Dict[str, int] = field(default_factory=dict)

    # === Restricted communication mechanic ===
    # Maps agent_id -> list of agent_ids they can send messages to
    # Agents not listed have no restrictions
    allowed_targets: Dict[str, List[str]] = field(default_factory=dict)

    def get_object_property(self, obj_id: str, prop: str, default: Any = None) -> Any:
        """Get a custom property for an object."""
        return self.object_properties.get(obj_id, {}).get(prop, default)

    def set_object_property(self, obj_id: str, prop: str, value: Any) -> "EnactToMGameState":
        """Set a custom property for an object. Returns new state."""
        new_state = copy.copy(self)
        new_props = copy.copy(self.object_properties)
        if obj_id not in new_props:
            new_props[obj_id] = {}
        else:
            new_props[obj_id] = copy.copy(new_props[obj_id])
        new_props[obj_id][prop] = value
        new_state.object_properties = new_props
        return new_state

    def get_habitat_state(self, obj_id: str, prop: str, default: Any = None) -> Any:
        """Get a property from Habitat's object state."""
        return self.object_states.get(obj_id, {}).get(prop, default)

    def record_action(self, record: ActionRecord) -> "EnactToMGameState":
        """Record an action. Returns new state."""
        new_state = copy.copy(self)
        new_state.action_history = self.action_history + [record]
        return new_state

    def add_observation(self, agent_id: str, observation: str) -> "EnactToMGameState":
        """Add an observation for an agent. Returns new state."""
        new_state = copy.copy(self)
        new_obs = copy.copy(self.agent_observations)
        if agent_id not in new_obs:
            new_obs[agent_id] = []
        else:
            new_obs[agent_id] = new_obs[agent_id] + [observation]
        new_state.agent_observations = new_obs
        return new_state

    def increment_step(self) -> "EnactToMGameState":
        """Increment the step counter. Returns new state."""
        new_state = copy.copy(self)
        new_state.current_step = self.current_step + 1
        return new_state

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for checkpointing."""
        return {
            "agent_positions": self.agent_positions,
            "agent_rooms": self.agent_rooms,
            "object_states": self.object_states,
            "object_properties": self.object_properties,
            "mirror_pairs": self.mirror_pairs,
            "inverse_objects": list(self.inverse_objects),
            "remote_mappings": self.remote_mappings,
            "current_step": self.current_step,
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "description": g.description,
                    "goal_type": g.goal_type,
                    "target": g.target,
                    "target_state": g.target_state,
                    "status": g.status.value,
                }
                for g in self.goals
            ],
            "completed_goals": list(self.completed_goals),
            "active_mechanics": self.active_mechanics,
            # Room restrictions (convert sets to lists for JSON)
            "restricted_rooms": {
                agent_id: list(rooms)
                for agent_id, rooms in self.restricted_rooms.items()
            },
            # Limited bandwidth
            "message_limits": self.message_limits,
            "messages_sent": self.messages_sent,
            # Restricted communication
            "allowed_targets": self.allowed_targets,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnactToMGameState":
        """Deserialize state from dict."""
        state = cls()
        state.agent_positions = data.get("agent_positions", {})
        state.agent_rooms = data.get("agent_rooms", {})
        state.object_states = data.get("object_states", {})
        state.object_properties = data.get("object_properties", {})
        state.mirror_pairs = data.get("mirror_pairs", [])
        state.inverse_objects = set(data.get("inverse_objects", []))
        state.remote_mappings = data.get("remote_mappings", {})
        state.current_step = data.get("current_step", 0)
        state.goals = [
            Goal(
                goal_id=g["goal_id"],
                description=g["description"],
                goal_type=g["goal_type"],
                target=g.get("target"),
                target_state=g.get("target_state"),
                status=GoalStatus(g.get("status", "pending")),
            )
            for g in data.get("goals", [])
        ]
        state.completed_goals = set(data.get("completed_goals", []))
        state.active_mechanics = data.get("active_mechanics", [])
        # Room restrictions (convert lists back to sets)
        state.restricted_rooms = {
            agent_id: set(rooms)
            for agent_id, rooms in data.get("restricted_rooms", {}).items()
        }
        # Limited bandwidth
        state.message_limits = data.get("message_limits", {})
        state.messages_sent = data.get("messages_sent", {})
        # Restricted communication
        state.allowed_targets = data.get("allowed_targets", {})
        return state

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EnactToMGameState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
