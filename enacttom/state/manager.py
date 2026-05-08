"""
EnactToM Game State Manager.

Handles:
- Syncing state from Habitat simulator
- Applying actions through mechanics (stateless transforms)
- Ticking time-based effects
- Setting up initial state from task definition
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import copy
import uuid

from enacttom.state.game_state import (
    EnactToMGameState,
    ActionRecord,
    Goal,
    GoalStatus,
)
from enacttom.mechanics.handlers import (
    apply_mechanics,
    HandlerResult,
    MECHANIC_INFO,
)

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from enacttom.task_gen.dag import DAGProgress


@dataclass
class ActionExecutionResult:
    """Result of executing an action."""
    state: EnactToMGameState
    observation: str
    success: bool
    effects: List[str]
    surprise_trigger: Optional[str] = None


class GameStateManager:
    """
    Manages EnactToM game state.

    Main interface for:
    - Initializing state from task definition
    - Syncing state from Habitat each step
    - Applying actions (with mechanic transforms)
    - Ticking time-based effects
    - Checking goal completion
    """

    def __init__(self, env_interface: Optional["EnvironmentInterface"] = None):
        """
        Initialize the game state manager.

        Args:
            env_interface: Habitat environment interface for syncing state.
                          Can be None for testing.
        """
        self.env = env_interface
        self.state = EnactToMGameState()

        # Story context from scenario system (for prompts)
        self._story_context: Optional[str] = None
        self._bindings_info: Optional[Dict[str, Any]] = None

        # State history for temporal evaluation.
        self._state_history: List[EnactToMGameState] = []
        self._success_condition: Optional[Dict[str, Any]] = None

        # DAG progress tracking for subtask completion
        self._dag_progress: Optional["DAGProgress"] = None
        self._subtasks: List[Any] = []  # Store subtasks for reference

    def initialize_from_task(self, task_data: Dict[str, Any]) -> EnactToMGameState:
        """
        Initialize game state from a task definition.

        Args:
            task_data: Task definition dict containing:
                - mechanics: List of mechanic bindings
                - goals: Task goals

        Returns:
            Initialized game state
        """
        state = EnactToMGameState()

        # Set active mechanics from bindings
        # Support both "mechanics" (old) and "mechanic_bindings" (new) field names
        mechanics = task_data.get("mechanic_bindings", task_data.get("mechanics", []))
        # Normalize active_mechanics to List[str] — some tasks store full
        # binding dicts here instead of just mechanic type strings.
        raw_mechanics = task_data.get("active_mechanics", [])
        state.active_mechanics = [
            m if isinstance(m, str) else m.get("mechanic_type", m.get("type"))
            for m in raw_mechanics
            if isinstance(m, str) or isinstance(m, dict)
        ]
        state.mechanic_bindings = mechanics

        for binding in mechanics:
            if isinstance(binding, str):
                mech_type = binding
            else:
                mech_type = binding.get("mechanic_type", binding.get("type"))

            if mech_type and mech_type not in state.active_mechanics:
                state.active_mechanics.append(mech_type)

            # Set up mechanic-specific state from bindings
            if isinstance(binding, dict):
                self._setup_mechanic_state(state, mech_type, binding)

        # Set up goals
        goals_data = task_data.get("goals", [])
        for g in goals_data:
            goal = Goal(
                goal_id=g.get("id", str(uuid.uuid4())),
                description=g.get("description", ""),
                goal_type=g.get("type", "unknown"),
                target=g.get("target"),
                target_state=g.get("target_state"),
                status=GoalStatus.PENDING,
            )
            state.goals.append(goal)

        # Apply initial states (unified format: object_id -> {property: value})
        # This allows tasks to start with doors open, objects dirty, etc.
        initial_states = task_data.get("initial_states", {})
        for object_id, properties in initial_states.items():
            if isinstance(properties, dict):
                for prop_name, prop_value in properties.items():
                    state = state.set_object_property(object_id, prop_name, prop_value)

        self.state = state

        # Store success condition for evaluation
        self._success_condition = task_data.get("success_condition")

        # Reset state history and record initial state
        self._state_history = [copy.deepcopy(state)]

        return state

    def _setup_mechanic_state(
        self, state: EnactToMGameState, mech_type: str, binding: Dict[str, Any]
    ) -> None:
        """Set up state for a specific mechanic binding."""
        trigger = binding.get("trigger_object")
        target_obj = binding.get("target_object")
        target_state = binding.get("target_state", "is_open")

        if mech_type == "inverse_state":
            if trigger:
                state.inverse_objects.add(trigger)

        elif mech_type == "remote_control":
            if trigger and target_obj:
                state.remote_mappings[trigger] = (target_obj, target_state)

        elif mech_type == "state_mirroring":
            if trigger and target_obj:
                state.mirror_pairs.append((trigger, target_obj, target_state))

        elif mech_type == "room_restriction":
            restricted_rooms = binding.get("restricted_rooms", [])
            for_agents = binding.get("for_agents", [])
            for agent_id in for_agents:
                if agent_id not in state.restricted_rooms:
                    state.restricted_rooms[agent_id] = set()
                state.restricted_rooms[agent_id].update(restricted_rooms)

        elif mech_type == "limited_bandwidth":
            message_limits = binding.get("message_limits", {})
            for agent_id, limit in message_limits.items():
                state.message_limits[agent_id] = int(limit)
                if agent_id not in state.messages_sent:
                    state.messages_sent[agent_id] = 0

        elif mech_type == "restricted_communication":
            allowed = binding.get("allowed_targets", {})
            for agent_id, targets in allowed.items():
                state.allowed_targets[agent_id] = targets

    def sync_from_habitat(self, state: Optional[EnactToMGameState] = None) -> EnactToMGameState:
        """
        Sync state from Habitat simulator.

        Args:
            state: State to update. If None, uses self.state.

        Returns:
            Updated game state
        """
        if state is None:
            state = self.state

        if self.env is None:
            return state

        try:
            world_graph_dict = self.env.world_graph
        except AttributeError:
            return state

        # Sync agent positions and rooms from WorldGraph objects
        agent_positions = {}
        agent_rooms = {}

        # world_graph_dict is Dict[int, WorldGraph] - use first available graph to get all agents
        wg = None
        for uid, graph in world_graph_dict.items():
            if graph is not None:
                wg = graph
                break

        if wg is not None:
            try:
                agents = wg.get_agents()
                for agent_node in agents:
                    # Get agent name from node
                    agent_name = agent_node.name if hasattr(agent_node, 'name') else None
                    if not agent_name:
                        continue

                    # Normalize to agent_N format
                    if agent_name.startswith("agent_"):
                        agent_id = agent_name
                    elif agent_name.isdigit():
                        agent_id = f"agent_{agent_name}"
                    else:
                        # Try to extract number or use as-is
                        agent_id = agent_name

                    # Get position if available
                    if hasattr(agent_node, 'position'):
                        agent_positions[agent_id] = agent_node.position

                    # Get room for this agent
                    try:
                        rooms = wg.get_room_for_entity(agent_node)
                        if rooms and len(rooms) > 0:
                            room_node = rooms[0]
                            room_name = room_node.name if hasattr(room_node, 'name') else str(room_node)
                            agent_rooms[agent_id] = room_name
                    except (ValueError, AttributeError):
                        pass
            except (ValueError, AttributeError):
                # WorldGraph might not have agents yet
                pass

        # Sync object states from entities in world graph
        object_states = {}
        entities = []

        room_by_entity: Dict[str, str] = {}
        for uid, wg in world_graph_dict.items():
            if wg is None:
                continue
            try:
                # Get all object nodes from the graph
                objects = wg.get_all_objects() if hasattr(wg, 'get_all_objects') else []
                for obj_node in objects:
                    entity_id = obj_node.name if hasattr(obj_node, 'name') else str(obj_node)
                    entity_dict = {"id": entity_id, "name": entity_id}

                    # Extract is_* states
                    states = {}
                    for attr in dir(obj_node):
                        if attr.startswith("is_") and not callable(getattr(obj_node, attr, None)):
                            try:
                                states[attr] = getattr(obj_node, attr)
                            except:
                                pass
                    if states:
                        object_states[entity_id] = states
                        entity_dict.update(states)

                    # Try to resolve room for this entity
                    try:
                        room = wg.get_room_for_entity(obj_node)
                        room_name = room.name if hasattr(room, "name") else str(room)
                        entity_dict["room"] = room_name
                        room_by_entity[entity_id] = room_name
                    except Exception:
                        pass

                    entities.append(entity_dict)
            except (ValueError, AttributeError):
                pass
            # Only need to process one agent's world graph for entities
            break

        new_state = copy.copy(state)
        new_state.agent_positions = agent_positions
        new_state.agent_rooms = agent_rooms
        new_state.object_states = object_states
        new_state.entities = entities

        # Persist room mapping into object_properties for quick lookup
        for entity_id, room_name in room_by_entity.items():
            new_state = new_state.set_object_property(entity_id, "room", room_name)

        self.state = new_state
        return new_state

    def auto_bind_mechanics(
        self, state: Optional[EnactToMGameState] = None
    ) -> Tuple[EnactToMGameState, Dict[str, Any]]:
        """
        Auto-bind mechanics to random objects in the scene.

        Call this after sync_from_habitat() to bind mechanics to real objects.

        Returns:
            (new_state, bindings_info) where bindings_info shows what was bound
        """
        import random

        if state is None:
            state = self.state

        bindings_info = {}

        # Get entities from state
        entities = getattr(state, 'entities', [])
        if not entities:
            return state, {"error": "No entities found - call sync_from_habitat first"}

        # Categorize entities
        articulated = []  # Furniture that can open/close (doors, drawers, cabinets)
        furniture = []    # All furniture
        objects = []      # Small objects

        for e in entities:
            name = e.get("name", e.get("id", ""))
            e_type = e.get("type", "")
            is_art = e.get("is_articulated", False)

            if is_art or any(k in name.lower() for k in ["door", "drawer", "cabinet", "fridge"]):
                articulated.append(name)
            if e_type == "furniture":
                furniture.append(name)
            elif e_type == "object":
                objects.append(name)

        # Shuffle for random selection
        random.shuffle(articulated)
        random.shuffle(furniture)
        random.shuffle(objects)
        new_state = copy.copy(state)

        # Bind each active mechanic
        for mech_type in state.active_mechanics:
            if mech_type == "inverse_state":
                # Bind to an articulated object
                if articulated:
                    target = articulated.pop(0)
                    new_state.inverse_objects.add(target)
                    bindings_info["inverse_state"] = {"target": target}

            elif mech_type == "remote_control":
                # Bind trigger -> target pair
                if len(articulated) >= 2:
                    trigger = articulated.pop(0)
                    target = articulated.pop(0)
                    new_state.remote_mappings[trigger] = (target, "is_open")
                    bindings_info["remote_control"] = {"trigger": trigger, "target": target}
                elif articulated and furniture:
                    trigger = articulated.pop(0)
                    target = furniture[0] if furniture else trigger
                    new_state.remote_mappings[trigger] = (target, "is_open")
                    bindings_info["remote_control"] = {"trigger": trigger, "target": target}

            elif mech_type == "state_mirroring":
                # Bind pair of articulated objects
                if len(articulated) >= 2:
                    obj1 = articulated.pop(0)
                    obj2 = articulated.pop(0)
                    new_state.mirror_pairs.append((obj1, obj2, "is_open"))
                    bindings_info["state_mirroring"] = {"pair": [obj1, obj2]}

        self.state = new_state

        # Store bindings for later retrieval by prompts and runners.
        self._bindings_info = bindings_info

        self._story_context = None

        return new_state, bindings_info

    def get_story_context(self) -> Optional[str]:
        """Get the story context from the current scenario (if any)."""
        return self._story_context

    def get_bindings_info(self) -> Optional[Dict[str, Any]]:
        """Get the full bindings info from auto_bind_mechanics."""
        return self._bindings_info

    def apply_action(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        state: Optional[EnactToMGameState] = None,
    ) -> Tuple[EnactToMGameState, ActionExecutionResult]:
        """
        Apply an action, running it through active mechanics.

        Args:
            action_name: Name of the action (e.g., "Open", "Pick")
            agent_id: Agent performing the action
            target: Target of the action
            state: State to apply to. If None, uses self.state.

        Returns:
            (new_state, result) tuple
        """
        if state is None:
            state = self.state

        # Apply mechanics first
        mech_result = apply_mechanics(action_name, agent_id, target, state)
        state = mech_result.state

        # If mechanic didn't handle it (or partially handled), apply built-in actions
        if not mech_result.applies or mech_result.success:
            state, builtin_result = self._apply_builtin_action(
                action_name, agent_id, target, state, mech_result
            )
            if builtin_result:
                mech_result = builtin_result

        # Record the action
        record = ActionRecord(
            step=state.current_step,
            agent_id=agent_id,
            action_name=action_name,
            target=target,
            success=mech_result.success,
            observation=mech_result.observation,
            effects=mech_result.effects,
        )
        state = state.record_action(record)
        state = state.add_observation(agent_id, mech_result.observation)

        result = ActionExecutionResult(
            state=state,
            observation=mech_result.observation,
            success=mech_result.success,
            effects=mech_result.effects,
            surprise_trigger=mech_result.surprise_trigger,
        )

        self.state = state

        # Record state snapshot for temporal evaluation
        self._state_history.append(copy.deepcopy(state))

        return state, result

    def _apply_builtin_action(
        self,
        action_name: str,
        agent_id: str,
        target: Optional[str],
        state: EnactToMGameState,
        mech_result: HandlerResult,
    ) -> Tuple[EnactToMGameState, Optional[HandlerResult]]:
        """Apply built-in action effects for paper-supported actions."""
        return state, None

    def tick(self, state: Optional[EnactToMGameState] = None) -> Tuple[EnactToMGameState, List[str]]:
        """
        Advance time by one step.

        Increments step counter.

        Args:
            state: State to tick. If None, uses self.state.

        Returns:
            (new_state, triggered_effects) tuple
        """
        if state is None:
            state = self.state

        new_state = state.increment_step()
        self.state = new_state
        return new_state, []

    def check_termination(self) -> bool:
        """The paper benchmark uses Habitat episode limits, not custom game termination."""
        return False

    def is_terminated(self) -> bool:
        """Compatibility wrapper for callers that poll custom termination."""
        return False

    def get_termination_reason(self) -> Optional[str]:
        """Custom termination is disabled in the paper code path."""
        return None

    def check_goals(self, state: Optional[EnactToMGameState] = None) -> List[Goal]:
        """
        Check which goals have been completed.

        Returns:
            List of newly completed goals
        """
        if state is None:
            state = self.state

        newly_completed = []

        for goal in state.goals:
            if goal.goal_id in state.completed_goals:
                continue

            completed = False

            if goal.goal_type == "change_state":
                if goal.target and goal.target_state:
                    completed = self._check_required_states(
                        [{"entity": goal.target, **goal.target_state}],
                        state
                    )

            if completed:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at_step = state.current_step
                newly_completed.append(goal)
                state.completed_goals.add(goal.goal_id)

        self.state = state
        return newly_completed

    def _check_required_states(
        self,
        required_states: List[Dict[str, Any]],
        state: EnactToMGameState,
    ) -> bool:
        """
        Check if all required states are satisfied.

        Args:
            required_states: List of {entity, property, value} dicts
            state: Game state to check

        Returns:
            True if all required states are satisfied
        """
        for req in required_states:
            entity = req.get("entity")
            prop = req.get("property")
            value = req.get("value")

            obj_states = state.object_states.get(entity, {})
            obj_props = state.object_properties.get(entity, {})
            all_states = {**obj_states, **obj_props}

            if all_states.get(prop) != value:
                return False

        return True

    def check_success_condition(
        self,
        success_condition: Dict[str, Any],
        state: Optional[EnactToMGameState] = None,
    ) -> bool:
        """
        Check if a task's success condition is met.

        Args:
            success_condition: Dict with "required_states" key
            state: Game state to check

        Returns:
            True if success condition is satisfied
        """
        if state is None:
            state = self.state

        required_states = success_condition.get("required_states", [])
        return self._check_required_states(required_states, state)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about current state."""
        state = self.state

        return {
            "current_step": state.current_step,
            "active_mechanics": state.active_mechanics,
            "inverse_objects": list(state.inverse_objects),
            "remote_mappings": {k: list(v) for k, v in state.remote_mappings.items()},
            "mirror_pairs": state.mirror_pairs,
            "goals": [
                {"id": g.goal_id, "status": g.status.value}
                for g in state.goals
            ],
        }

    def get_state(self) -> EnactToMGameState:
        """Get current game state."""
        return self.state

    def set_state(self, state: EnactToMGameState) -> None:
        """Set current game state."""
        self.state = state

    GAME_STATE_PREDICATES = {"is_unlocked"}

    def _check_game_state_predicate(self, condition: Dict[str, Any]) -> Optional[bool]:
        """Evaluate small state-overlay predicates that do not require Habitat."""
        prop = condition.get("property")
        if prop not in self.GAME_STATE_PREDICATES:
            return None

        entity = condition.get("entity")
        value = condition.get("value")
        if prop == "is_unlocked" and entity:
            is_unlocked = self.state.get_object_property(entity, "is_unlocked", None)
            if is_unlocked is None:
                is_unlocked = not bool(self.state.get_object_property(entity, "is_locked", False))
            result = bool(is_unlocked)
            return not result if value is False else result

        return False

    # ========== Predicate Evaluation ==========

    def evaluate_task(
        self,
        success_condition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion with Habitat and EnactToM predicates.

        Uses the actual Habitat simulator state for ground-truth evaluation.

        Args:
            success_condition: Task's success condition dict. If None, uses
                              the one stored from initialize_from_task().

        Returns:
            Dict with:
                - percent_complete: float (0.0 to 1.0)
                - success: bool
                - failure_explanations: List[str]
                - proposition_status: Dict[str, bool]
        """
        from enacttom.evaluation import evaluate_task

        condition = success_condition or self._success_condition
        if not condition:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": ["No success condition defined"],
                "proposition_status": {},
            }

        if not self.env:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": ["No environment interface available"],
                "proposition_status": {},
            }

        try:
            sim = self.env.sim
            result = evaluate_task(condition, sim)
            return result.to_dict()
        except Exception as e:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": [f"Evaluation error: {str(e)}"],
                "proposition_status": {},
            }

    def get_state_history(self) -> List[EnactToMGameState]:
        """Get the full state history for temporal analysis."""
        return self._state_history

    def clear_state_history(self) -> None:
        """Clear state history (useful for resetting)."""
        self._state_history = [copy.deepcopy(self.state)]

    # =========================================================================
    # DAG Progress Tracking
    # =========================================================================

    def initialize_dag_progress(self, subtasks: List[Any]) -> None:
        """
        Initialize DAG progress tracking from task subtasks.

        Args:
            subtasks: List of Subtask objects with success_conditions
        """
        from enacttom.task_gen.dag import DAGProgress

        self._subtasks = subtasks
        if subtasks:
            self._dag_progress = DAGProgress.from_subtasks(subtasks)

    def update_dag_progress(self) -> Dict[str, Any]:
        """
        Update DAG progress by checking subtask conditions against current state.

        Should be called after each action to latch completed subtasks.

        Returns:
            Dict with completed, newly_completed, percent_complete, success
        """
        if not self._dag_progress:
            return {
                "completed": [],
                "newly_completed": [],
                "percent_complete": 0.0,
                "success": False,
                "error": "No DAG progress initialized",
            }

        def check_condition(subtask) -> bool:
            """Check if a subtask's success_condition is satisfied."""
            condition = subtask.success_condition
            if not condition:
                return False

            # First, check if this is an EnactToM game state predicate.
            game_state_result = self._check_game_state_predicate(condition)
            if game_state_result is not None:
                return game_state_result

            # Fall back to simulator-based predicates (is_on_top, is_open, etc.)
            from enacttom.evaluation import evaluate_task

            # Wrap single condition in required_states format
            success_cond = {
                "description": subtask.description,
                "required_states": [condition],
            }

            if not self.env:
                return False

            try:
                sim = self.env.sim
                result = evaluate_task(success_cond, sim)
                return result.success
            except Exception:
                return False

        return self._dag_progress.update(check_condition)

    def get_dag_status(self) -> Dict[str, Any]:
        """Get current DAG progress status without updating."""
        if not self._dag_progress:
            return {
                "completed": [],
                "percent_complete": 0.0,
                "success": False,
                "remaining": [],
            }
        return self._dag_progress.get_status()

    def reset_dag_progress(self) -> None:
        """Reset DAG progress (clear all completed subtasks)."""
        if self._dag_progress:
            self._dag_progress.reset()
