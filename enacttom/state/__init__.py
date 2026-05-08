"""
EnactToM Game State Module.

Provides centralized state management for EnactToM mechanics and actions.

Usage:
    from enacttom.state import EnactToMGameState, GameStateManager

    # Initialize from task
    manager = GameStateManager(env_interface)
    state = manager.initialize_from_task(task_data)

    # Game loop
    while not done:
        state = manager.sync_from_habitat()
        state, result = manager.apply_action("open", "agent_0", "door_1")
        state = manager.tick()
        completed = manager.check_goals()
"""

from enacttom.state.game_state import (
    EnactToMGameState,
    ActionRecord,
    Goal,
    GoalStatus,
)

from enacttom.state.manager import (
    GameStateManager,
    ActionExecutionResult,
)

__all__ = [
    # Core state
    "EnactToMGameState",
    "ActionRecord",
    "Goal",
    "GoalStatus",
    # Manager
    "GameStateManager",
    "ActionExecutionResult",
]
