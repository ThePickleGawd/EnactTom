"""
Custom EnactToM Actions.

These actions extend the Habitat tool interface directly and can be affected by mechanics.
Each action has:
- A normal expected behavior
- Can be transformed by mechanics (inverse, remote control, counting, etc.)
- Produces observations that may differ per agent (theory of mind)

To add a new action:
1. Create a class that extends EnactToMAction
2. Decorate it with @register_action("ActionName")
3. The action will automatically be available in generation and benchmark runs.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from habitat_llm.tools.tool import Tool

from enacttom.actions.registry import ActionRegistry

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


@dataclass
class ActionResult:
    """Result of executing a custom action."""
    success: bool
    observation: str  # What the acting agent observes
    effect: Optional[str] = None  # What actually changed
    other_observations: Dict[str, str] = field(default_factory=dict)  # What other agents observe
    surprise_trigger: Optional[str] = None  # If this should trigger surprise detection


class EnactToMAction(Tool):
    """
    Base class for EnactToM custom actions.

    Extends the Habitat tool interface directly so actions can be used
    by agents in the evaluation framework without a wrapper layer.
    """

    action_name: str = "base_action"
    action_description: str = "Base action"

    def __init__(self, agent_uid: int = 0):
        super().__init__(self.action_name, agent_uid)
        self.env_interface: Optional["EnvironmentInterface"] = None
        self._game_manager = None

    def set_environment(self, env_interface: "EnvironmentInterface"):
        """Set the environment interface for this action."""
        self.env_interface = env_interface

    def set_game_manager(self, game_manager):
        """Set the GameStateManager for this action to access game state."""
        self._game_manager = game_manager

    def to(self, device):
        """Compatibility method for device placement."""
        pass

    def get_state_description(self) -> str:
        """Method to get a string describing the state for this action."""
        return "Standing"

    @property
    def description(self) -> str:
        return self.action_description

    @property
    def argument_types(self) -> List[str]:
        return ["OBJECT_INSTANCE"]

    def _build_world_state(self) -> Dict[str, Any]:
        """Build world state dict for action execution."""
        if not self.env_interface:
            return {}

        world_state = {
            "agent_location": "unknown",
            "rooms": [],
            "entities": [],
            "entity_details": {},
            "other_agents": [],
        }

        try:
            wg = self.env_interface.world_graph.get(self.agent_uid)
            if wg:
                world_state["rooms"] = [r.name for r in wg.get_all_rooms()]
                for node in wg.graph.nodes():
                    if hasattr(node, 'name'):
                        entity = {
                            "name": node.name,
                            "type": getattr(node, 'node_type', 'unknown'),
                            "properties": {},
                            "states": {},
                        }
                        world_state["entities"].append(entity)
                        world_state["entity_details"][node.name] = {
                            "properties": entity["properties"],
                            "states": entity["states"],
                        }
        except Exception:
            pass

        return world_state

    def process_high_level_action(
        self, input_query: str, observations: Any
    ) -> Tuple[Optional[Any], str]:
        """
        Execute the EnactToM action (Tool interface).

        Args:
            input_query: The target for the action (e.g., object name)
            observations: Current observations (unused for EnactToM actions)

        Returns:
            Tuple of (low_level_action, response_text)
        """
        world_state = self._build_world_state()
        result = self.execute(
            agent_id=f"agent_{self.agent_uid}",
            target=input_query,
            world_state=world_state,
        )
        return None, result.observation

    @abstractmethod
    def execute(
        self,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """
        Execute the action.

        Args:
            agent_id: The agent performing the action
            target: Optional target for the action
            world_state: Current world state info

        Returns:
            ActionResult with observation and effects
        """
        pass

    def get_available_targets(self, world_state: Dict[str, Any]) -> List[str]:
        """Get valid targets for this action in current state."""
        return []


def get_all_actions() -> Dict[str, EnactToMAction]:
    """Get all registered EnactToM actions (instantiated)."""
    return ActionRegistry.instantiate_all()


# For backwards compatibility - dynamically gets all registered actions
ENACTTOM_ACTIONS: Dict[str, EnactToMAction] = get_all_actions()


def get_enacttom_tools(agent_uid: int = 0) -> Dict[str, EnactToMAction]:
    """
    Get all EnactToM actions instantiated for a given agent.

    This is the main entry point for getting actions to use with agents.

    Args:
        agent_uid: The agent ID to create actions for

    Returns:
        Dict mapping action names to action instances
    """
    from enacttom.actions.registry import get_registry
    actions = {}
    for name, action_cls in get_registry().items():
        actions[name] = action_cls(agent_uid=agent_uid)
    return actions


class EnactToMActionExecutor:
    """
    Executor for EnactToM custom actions.

    Integrates custom actions with the Habitat environment and mechanics system.
    Uses the ActionRegistry to automatically discover all registered actions.
    """

    def __init__(
        self,
        env_interface: "EnvironmentInterface",
        mechanics: Optional[List[Any]] = None,
    ):
        self.env = env_interface
        self.mechanics = mechanics or []
        self.actions = ActionRegistry.instantiate_all()

    def get_available_actions(self, world_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of available custom actions with their targets."""
        available = []
        for name, action in self.actions.items():
            targets = action.get_available_targets(world_state)
            available.append({
                "name": name,
                "description": action.description,
                "targets": targets,
            })
        return available

    def execute(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        world_state: Dict[str, Any],
    ) -> ActionResult:
        """Execute a custom action, applying any relevant mechanics."""
        if action_name not in self.actions:
            return ActionResult(
                success=False,
                observation=f"Unknown action: {action_name}",
            )

        action = self.actions[action_name]
        result = action.execute(agent_id, target, world_state)

        for mechanic in self.mechanics:
            if hasattr(mechanic, 'transform_action_result'):
                result = mechanic.transform_action_result(
                    action_name, agent_id, target, result, world_state
                )

        return result

    def register_action(self, action: EnactToMAction) -> None:
        """Register a new custom action."""
        self.actions[action.name] = action
