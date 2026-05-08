"""
Action registry for EnactToM benchmark.

Provides a decorator-based registration system for custom actions,
enabling plug-and-play action selection via configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from enacttom.actions.custom_actions import EnactToMAction

# Global registry of action classes
_ACTION_REGISTRY: Dict[str, Type["EnactToMAction"]] = {}

# Standard habitat_llm motor skills that EnactToM uses.
# These descriptions mirror the Habitat tool YAML configs in habitat_llm/conf/tools/motor_skills/
# to ensure the task generator uses the exact API format expected by the simulator.
STANDARD_ACTIONS: Dict[str, str] = {
    "Navigate": (
        "Navigate[target]: Used for navigating to an entity. You must provide the name of the "
        "entity you want to navigate to. Example: Navigate[counter_22]"
    ),
    "Pick": (
        "Pick[object]: Used for picking up an object. You must provide the name of the object "
        "to be picked. The agent cannot hold more than one object at a time. Example: Pick[cup_1]"
    ),
    "Place": (
        "Place[object, spatial_relation, furniture, spatial_constraint, reference_object]: "
        "Used for placing an object on a target location. You need to provide: "
        "(1) name of the object to be placed, "
        "(2) spatial relation ('on' or 'within'), "
        "(3) name of the furniture where it should be placed. "
        "The object must already be held by the agent. "
        "Optional: spatial_constraint ('next_to') and reference_object for placing near another object. "
        "Set spatial_constraint and reference_object to 'None' when not needed. "
        "Example: Place[cup_1, on, table_22, None, None]"
    ),
    "Open": (
        "Open[furniture]: Used for opening an articulated entity (cabinet, drawer, fridge). "
        "You must provide the name of the furniture you want to open. Example: Open[chest_of_drawers_1]"
    ),
    "Close": (
        "Close[furniture]: Used for closing an articulated entity. "
        "You must provide the name of the furniture you want to close. Example: Close[chest_of_drawers_1]"
    ),
    "Wait": (
        "Wait[]: Used to make agent stay idle for some time. Takes no arguments. Example: Wait[]"
    ),
    "Communicate": (
        'Communicate["message", recipients]: Send a message to specific agents. '
        'The message must be in double quotes, followed by recipients. '
        'Use agent IDs (agent_0, agent_1, ...) or "all" to broadcast. '
        'Example DM: Communicate["I found the key!", agent_1] '
        'Example group: Communicate["Let\'s meet in the kitchen", agent_0, agent_2] '
        'Example broadcast: Communicate["Hello everyone!", all]'
    ),
    "FindObjectTool": (
        "FindObjectTool[query]: Find the exact name of objects matching a description. "
        "Use when you know what type of object to look for but not its ID. "
        "Example: FindObjectTool[books on the shelf] → returns 'book_1 is on shelves_13'"
    ),
    "FindReceptacleTool": (
        "FindReceptacleTool[query]: Find the exact name of furniture/receptacles matching a description. "
        "Use when you need to find where to place things or where scene objects might be stored. "
        "Example: FindReceptacleTool[a table in the living room] → returns 'table_16'"
    ),
    "FindRoomTool": (
        "FindRoomTool[query]: Find the exact name of a room matching a description. "
        "Example: FindRoomTool[a room with a bed] → returns 'bedroom_1'"
    ),
}


def register_action(name: Optional[str] = None):
    """
    Decorator to register a custom action class.

    Usage:
        @register_action("Inspect")
        class InspectAction(EnactToMAction):
            ...

        # Or use the class's name attribute:
        @register_action()
        class InspectAction(EnactToMAction):
            name = "Inspect"
    """

    def decorator(cls: Type["EnactToMAction"]) -> Type["EnactToMAction"]:
        action_name = name or getattr(cls, "name", cls.__name__)
        if action_name in _ACTION_REGISTRY:
            raise ValueError(
                f"Action '{action_name}' is already registered "
                f"(by {_ACTION_REGISTRY[action_name].__name__})"
            )
        _ACTION_REGISTRY[action_name] = cls
        return cls

    return decorator


class ActionRegistry:
    """
    Central registry for all available custom actions.

    Provides methods to query, instantiate, and compose actions.
    """

    @staticmethod
    def get(name: str) -> Type["EnactToMAction"]:
        """Get an action class by name."""
        if name not in _ACTION_REGISTRY:
            available = ", ".join(sorted(_ACTION_REGISTRY.keys()))
            raise KeyError(
                f"Unknown action: '{name}'. Available: {available}"
            )
        return _ACTION_REGISTRY[name]

    @staticmethod
    def list_all() -> List[str]:
        """List all registered action names."""
        return sorted(_ACTION_REGISTRY.keys())

    @staticmethod
    def is_registered(name: str) -> bool:
        """Check if an action is registered."""
        return name in _ACTION_REGISTRY

    @staticmethod
    def instantiate(name: str, **params) -> "EnactToMAction":
        """Create an instance of an action."""
        cls = ActionRegistry.get(name)
        return cls(**params)

    @staticmethod
    def instantiate_all() -> Dict[str, "EnactToMAction"]:
        """Instantiate all registered actions."""
        return {name: cls() for name, cls in _ACTION_REGISTRY.items()}

    @staticmethod
    def get_info(name: str) -> Dict[str, Any]:
        """Get information about a registered action."""
        cls = ActionRegistry.get(name)
        return {
            "name": name,
            "description": getattr(cls, "description", ""),
            "class": cls.__name__,
        }

    @staticmethod
    def describe_all() -> str:
        """Get a human-readable description of all registered actions."""
        lines = ["Registered Actions:", "=" * 40]
        for name in sorted(_ACTION_REGISTRY.keys()):
            info = ActionRegistry.get_info(name)
            desc = info["description"][:60] + "..." if len(info["description"]) > 60 else info["description"]
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def get_action_descriptions(include_standard: bool = False) -> str:
        """
        Get action descriptions formatted for system prompts.

        Args:
            include_standard: If True, include standard habitat_llm tools

        Returns a string with each action on a line:
        - ActionName: Description of the action
        """
        lines = []

        # Include standard actions if requested
        if include_standard:
            for name in sorted(STANDARD_ACTIONS.keys()):
                lines.append(f"- {STANDARD_ACTIONS[name]}")

        # Include registered custom actions
        for name in sorted(_ACTION_REGISTRY.keys()):
            cls = _ACTION_REGISTRY[name]
            desc = getattr(cls, "action_description", getattr(cls, "description", ""))
            lines.append(f"- {name}: {desc}")

        return "\n".join(lines)

    @staticmethod
    def get_all_action_descriptions() -> str:
        """
        Get all action descriptions (standard + custom) for prompts.

        Returns a formatted string suitable for injection into system prompts.
        """
        return ActionRegistry.get_action_descriptions(include_standard=True)


def clear_registry() -> None:
    """Clear all registered actions (useful for testing)."""
    _ACTION_REGISTRY.clear()


def get_registry() -> Dict[str, Type["EnactToMAction"]]:
    """Get the raw registry dict (useful for testing)."""
    return _ACTION_REGISTRY
