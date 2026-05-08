"""
Tool Schema Generator for EnactToM.

Provides decorators that auto-generate OpenAI-style tool schemas from
Python function signatures and type hints. Borrowed from ARE's pattern.

Usage:
    class EnactToMTools:
        @tool(description="Open a container to find items inside")
        def open(self, target: str) -> str:
            '''Open the target container.'''
            ...

    # Get schemas for LLM tool use
    schemas = get_tool_schemas(tools_instance)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    get_type_hints,
    List,
    Optional,
    Type,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from enacttom.state.manager import GameStateManager
    from habitat_llm.agent.env import EnvironmentInterface


# Type mapping from Python types to JSON Schema types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    List: "array",
    dict: "object",
    Dict: "object",
    type(None): "null",
}


def _python_to_json_type(py_type: Type) -> str:
    """Convert Python type to JSON Schema type string."""
    # Handle Optional types
    origin = getattr(py_type, "__origin__", None)
    if origin is Union:
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_to_json_type(non_none[0])
        return "string"  # Fallback for complex unions

    # Handle List[X], Dict[K, V], etc.
    if origin is not None:
        if origin in (list, List):
            return "array"
        if origin in (dict, Dict):
            return "object"

    # Direct type lookup
    return _TYPE_MAP.get(py_type, "string")


def _get_param_schema(py_type: Type) -> Dict[str, Any]:
    """Generate JSON schema for a parameter type."""
    origin = getattr(py_type, "__origin__", None)

    # Handle List[X]
    if origin in (list, List):
        args = getattr(py_type, "__args__", ())
        item_type = args[0] if args else str
        return {
            "type": "array",
            "items": {"type": _python_to_json_type(item_type)},
        }

    # Handle Dict[K, V]
    if origin in (dict, Dict):
        return {"type": "object"}

    # Handle Optional[X]
    if origin is Union:
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _get_param_schema(non_none[0])

    return {"type": _python_to_json_type(py_type)}


@dataclass
class ToolSchema:
    """Schema for an LLM tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    target_types: List[str] = field(default_factory=lambda: ["OBJECT_INSTANCE"])
    returns: str = "string"

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "target_types": self.target_types,
            "returns": self.returns,
        }


def tool(
    description: str,
    targets: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> Callable:
    """
    Decorator to mark a method as an LLM tool with auto-generated schema.

    Args:
        description: Human-readable description of what the tool does
        targets: List of valid target types (e.g., ["FURNITURE_INSTANCE", "OBJECT_INSTANCE"])
        name: Override the function name for the tool

    Example:
        @tool(description="Open a container to find items inside",
              targets=["FURNITURE_INSTANCE"])
        def open(self, target: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        sig = inspect.signature(func)

        # Build parameters schema
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, str)
            properties[param_name] = _get_param_schema(param_type)

            # Add description from docstring if available
            if func.__doc__:
                # Simple extraction - could be enhanced
                properties[param_name]["description"] = f"The {param_name} parameter"

            # Check if required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build the schema
        schema = ToolSchema(
            name=name or func.__name__,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            target_types=targets or ["OBJECT_INSTANCE"],
            returns=_python_to_json_type(hints.get("return", str)),
        )

        # Attach schema to function
        func._tool_schema = schema
        func._is_tool = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tool_schema = schema
        wrapper._is_tool = True
        return wrapper

    return decorator


def get_tool_schemas(obj: Any) -> List[ToolSchema]:
    """
    Extract all tool schemas from an object (class instance).

    Args:
        obj: Object to extract schemas from

    Returns:
        List of ToolSchema objects
    """
    schemas = []
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(obj, attr_name)
            if callable(attr) and getattr(attr, "_is_tool", False):
                schemas.append(attr._tool_schema)
        except Exception:
            continue
    return schemas


def get_openai_tools(obj: Any) -> List[Dict[str, Any]]:
    """
    Get tool schemas in OpenAI function calling format.

    Args:
        obj: Object to extract schemas from

    Returns:
        List of dicts in OpenAI format
    """
    return [s.to_openai_format() for s in get_tool_schemas(obj)]


def schemas_to_prompt(schemas: List[ToolSchema]) -> str:
    """
    Format tool schemas as text for inclusion in prompts.

    Args:
        schemas: List of ToolSchema objects

    Returns:
        Formatted string describing available tools
    """
    lines = ["Available tools:"]
    for schema in schemas:
        params = schema.parameters.get("properties", {})
        param_strs = []
        for name, info in params.items():
            type_str = info.get("type", "string")
            param_strs.append(f"{name}: {type_str}")
        params_text = ", ".join(param_strs) if param_strs else ""
        lines.append(f"  - {schema.name}({params_text}): {schema.description}")
    return "\n".join(lines)


class ToolRegistry:
    """
    Registry for tool classes with schema generation.

    Allows collecting tools from multiple sources and generating
    unified schemas.
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}

    def register(self, func: Callable) -> None:
        """Register a tool function."""
        if not getattr(func, "_is_tool", False):
            raise ValueError(f"{func.__name__} is not decorated with @tool")
        schema = func._tool_schema
        self._tools[schema.name] = func
        self._schemas[schema.name] = schema

    def register_from_object(self, obj: Any) -> None:
        """Register all tools from an object."""
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(obj, attr_name)
                if callable(attr) and getattr(attr, "_is_tool", False):
                    self.register(attr)
            except Exception:
                continue

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Get schema by tool name."""
        return self._schemas.get(name)

    def get_all_schemas(self) -> List[ToolSchema]:
        """Get all registered schemas."""
        return list(self._schemas.values())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all schemas in OpenAI format."""
        return [s.to_openai_format() for s in self._schemas.values()]

    def to_prompt(self) -> str:
        """Format all tools as prompt text."""
        return schemas_to_prompt(list(self._schemas.values()))


# Global registry for convenience
_global_registry = ToolRegistry()


def register_tool(func: Callable) -> Callable:
    """Decorator to register a tool in the global registry."""
    _global_registry.register(func)
    return func


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry
