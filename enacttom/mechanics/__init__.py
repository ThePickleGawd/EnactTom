"""
EnactToM Mechanics Module.

Provides stateless mechanic handlers for transforming actions.
All state lives in EnactToMGameState, mechanics are pure functions.

Usage:
    from enacttom.mechanics import apply_mechanics, list_mechanics, get_mechanic_info

    # Apply mechanics to an action
    result = apply_mechanics("open", "agent_0", "door_1", game_state)

    # List available mechanics
    mechanics = list_mechanics()

    # Get info about a mechanic
    info = get_mechanic_info("inverse_state")
"""

from enacttom.mechanics.handlers import (
    # Core functions
    apply_mechanics,
    get_handler,
    list_mechanics,
    get_mechanic_info,
    get_mechanics_for_task_generation,
    # Types
    HandlerResult,
    MechanicHandler,
    # Info
    MECHANIC_INFO,
    MECHANIC_HANDLERS,
    # Individual handlers (for direct use if needed)
    handle_inverse_state,
    handle_remote_control,
    handle_state_mirroring,
    handle_limited_bandwidth,
)

__all__ = [
    # Core functions
    "apply_mechanics",
    "get_handler",
    "list_mechanics",
    "get_mechanic_info",
    "get_mechanics_for_task_generation",
    # Types
    "HandlerResult",
    "MechanicHandler",
    # Info
    "MECHANIC_INFO",
    "MECHANIC_HANDLERS",
    # Individual handlers
    "handle_inverse_state",
    "handle_remote_control",
    "handle_state_mirroring",
    "handle_limited_bandwidth",
]
