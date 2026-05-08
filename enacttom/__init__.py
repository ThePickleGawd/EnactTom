"""
EnactToM: Embodied Theory of Mind Benchmark

A framework for testing theory of mind reasoning through mechanics
with "unexpected behaviors" that induce surprise and require mental modeling.

Usage:
    from enacttom import GameStateManager, EnactToMGameState

    # Initialize
    manager = GameStateManager(env_interface)
    state = manager.initialize_from_task(task_data)

    # Game loop
    state = manager.sync_from_habitat()
    state, result = manager.apply_action("open", "agent_0", "door_1")
    state, triggered = manager.tick()
    completed = manager.check_goals()
"""

from enacttom.state import EnactToMGameState, GameStateManager, ActionExecutionResult
from enacttom.mechanics import (
    apply_mechanics,
    list_mechanics,
    get_mechanic_info,
    HandlerResult,
    MECHANIC_INFO,
)
from enacttom.evaluation import (
    TaskEvaluator,
    EvaluationResult,
    PropositionResult,
    evaluate_task,
    is_open,
    is_closed,
    HABITAT_PREDICATES,
    ENACTTOM_PREDICATES,
)
from enacttom.tracing import (
    Event,
    EventType,
    EventLog,
)

__all__ = [
    # State (primary interface)
    "EnactToMGameState",
    "GameStateManager",
    "ActionExecutionResult",
    # Mechanics
    "apply_mechanics",
    "list_mechanics",
    "get_mechanic_info",
    "HandlerResult",
    "MECHANIC_INFO",
    # Evaluation
    "TaskEvaluator",
    "EvaluationResult",
    "PropositionResult",
    "evaluate_task",
    "is_open",
    "is_closed",
    "HABITAT_PREDICATES",
    "ENACTTOM_PREDICATES",
    # Tracing (ARE-style)
    "Event",
    "EventType",
    "EventLog",
]
