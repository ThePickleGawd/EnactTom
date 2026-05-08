"""
EnactToM Custom Actions.

These actions extend the standard Habitat tools with mechanics-aware behaviors.
They can be affected by EnactToM mechanics (inverse_state, remote_control, etc.)

To add a new action:
1. Create a class that extends EnactToMAction in custom_actions.py
2. Decorate it with @register_action("ActionName")
3. The action will automatically be available everywhere
"""

from enacttom.actions.registry import ActionRegistry, register_action
from enacttom.actions.custom_actions import (
    ActionResult,
    EnactToMAction,
    EnactToMActionExecutor,
    ENACTTOM_ACTIONS,
    get_all_actions,
    get_enacttom_tools,
)
from enacttom.actions.schema import (
    tool,
    ToolSchema,
    ToolRegistry,
    get_tool_schemas,
    get_openai_tools,
    schemas_to_prompt,
    get_global_registry,
)

__all__ = [
    # Registry
    "ActionRegistry",
    "register_action",
    # Base classes
    "ActionResult",
    "EnactToMAction",
    "EnactToMActionExecutor",
    # Helpers
    "ENACTTOM_ACTIONS",
    "get_all_actions",
    "get_enacttom_tools",
    # Tool schema generation (ARE-style)
    "tool",
    "ToolSchema",
    "ToolRegistry",
    "get_tool_schemas",
    "get_openai_tools",
    "schemas_to_prompt",
    "get_global_registry",
]
