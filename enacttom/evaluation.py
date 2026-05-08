"""
EnactToM Evaluation System.

Uses Habitat simulator predicates for ground-truth simulator state checks
and adds EnactToM overlay predicates.

Supports the paper task categories:
- Cooperative: All agents work toward shared goals (required=True subtasks)
- Mixed: Shared main goal + agent-specific subgoals (required="agent_X" subtasks)

Based on: https://arxiv.org/abs/2411.00081
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.sims.enacttom_sim import EnactToMSim
    from habitat_llm.world_model import Graph
    from enacttom.task_gen.task_generator import GeneratedTask
    from enacttom.state.manager import GameStateManager


@dataclass
class PropositionResult:
    """Result of checking a proposition."""
    is_satisfied: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of task evaluation (cooperative tasks)."""
    percent_complete: float
    success: bool
    failure_explanations: List[str]
    proposition_status: Dict[str, bool] = field(default_factory=dict)
    communication_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "percent_complete": self.percent_complete,
            "success": self.success,
            "failure_explanations": self.failure_explanations,
            "proposition_status": self.proposition_status,
        }
        if self.communication_metrics:
            result["communication_metrics"] = self.communication_metrics
        return result


@dataclass
class MixedResult:
    """Result of mixed task evaluation."""
    main_goal_success: bool  # Did required=True subtasks complete?
    main_goal_progress: float  # Percent of main goal subtasks completed
    agent_subgoal_status: Dict[str, bool]  # agent_id -> completed their subgoal
    proposition_status: Dict[str, bool] = field(default_factory=dict)
    communication_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "main_goal_success": self.main_goal_success,
            "main_goal_progress": self.main_goal_progress,
            "agent_subgoal_status": self.agent_subgoal_status,
            "proposition_status": self.proposition_status,
        }
        if self.communication_metrics:
            result["communication_metrics"] = self.communication_metrics
        return result


HABITAT_PREDICATES = {
    "is_on_top", "is_inside", "is_in_room", "is_on_floor", "is_next_to",
    "is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off",
}

ENACTTOM_PREDICATES = {"is_open", "is_closed", "is_held_by", "is_unlocked", "is_locked"}

# Predicates that require GameStateManager instead of simulator
GAME_STATE_PREDICATES = {"is_unlocked", "is_locked"}

ALL_PREDICATES = HABITAT_PREDICATES | ENACTTOM_PREDICATES

# Predicate function lookup for EnactToM-specific predicates
_EnactToM_PREDICATE_MAP = {
    "is_open": None,  # Populated lazily after function definitions
    "is_closed": None,
    "is_held_by": None,
}


def _build_room_name_map(sim: "EnactToMSim") -> Dict[str, Any]:
    """Build mapping from room names (and unique base names) to region IDs."""
    try:
        region_counts: Dict[str, int] = {}
        room_name_map: Dict[str, Any] = {}
        base_to_ids: Dict[str, List[Any]] = {}

        for region in sim.semantic_scene.regions:
            base_name = region.category.name().split("/")[0].replace(" ", "_")
            region_counts[base_name] = region_counts.get(base_name, 0) + 1
            room_name = f"{base_name}_{region_counts[base_name]}"
            room_name_map[room_name] = region.id
            base_to_ids.setdefault(base_name, []).append(region.id)

        # Allow base names (e.g., "kitchen") when unambiguous
        for base_name, ids in base_to_ids.items():
            if len(ids) == 1:
                room_name_map[base_name] = ids[0]

        return room_name_map
    except Exception:
        return {}


def is_open(
    sim: "EnactToMSim",
    object_handles: List[str],
    threshold: float = 0.05,
    **kwargs,
) -> PropositionResult:
    """Check if an articulated object is open using joint positions."""
    from habitat.sims.habitat_simulator.sim_utilities import (
        get_ao_default_link,
        link_is_open,
    )

    aom = sim.get_articulated_object_manager()

    for handle in object_handles:
        ao = aom.get_object_by_handle(handle)
        if ao is None:
            continue

        default_link = get_ao_default_link(ao, compute_if_not_found=True)
        if default_link is None:
            return PropositionResult(True, {"object_handles": handle})

        if link_is_open(ao, default_link, threshold=threshold):
            return PropositionResult(True, {"object_handles": handle})

    return PropositionResult(False, {"object_handles": ""})


def is_closed(
    sim: "EnactToMSim",
    object_handles: List[str],
    **kwargs,
) -> PropositionResult:
    """Check if an articulated object is closed."""
    result = is_open(sim, object_handles)
    return PropositionResult(not result.is_satisfied, result.info)


def is_held_by(
    sim: "EnactToMSim",
    object_handles: List[str],
    agent_ids: List[str],
) -> PropositionResult:
    """Check if an object is being held by an agent."""
    for agent_id in agent_ids:
        try:
            agent_idx = int(agent_id.split("_")[-1]) if "_" in agent_id else int(agent_id)
            agent = sim.agents_mgr[agent_idx]
            grasp_mgr = agent.grasp_mgr

            if not grasp_mgr.is_grasped:
                continue

            grasped_obj_id = grasp_mgr.snap_idx
            rom = sim.get_rigid_object_manager()

            # Find the handle of the actually-grasped object by scanning all
            # rigid objects, then check if it matches any requested handle.
            grasped_handle = None
            for h in rom.get_object_handles():
                try:
                    obj = rom.get_object_by_handle(h)
                    if obj is not None and obj.object_id == grasped_obj_id:
                        grasped_handle = h
                        break
                except Exception:
                    continue

            if grasped_handle is None:
                continue

            # Match the grasped object's handle against each requested handle.
            grasped_base = grasped_handle.split(":")[0].strip().rstrip("_")
            for handle in object_handles:
                if handle == grasped_handle:
                    return PropositionResult(True, {"agent": agent_id, "object": handle})
                handle_norm = handle.strip().rstrip("_")
                if handle_norm == grasped_base or grasped_handle.startswith(handle_norm):
                    return PropositionResult(True, {"agent": agent_id, "object": handle})
        except Exception:
            continue

    return PropositionResult(False, {"object_handles": object_handles, "agent_ids": agent_ids})


# Now that predicate functions are defined, populate the lookup map.
_EnactToM_PREDICATE_MAP.update({
    "is_open": is_open,
    "is_closed": is_closed,
    "is_held_by": is_held_by,
})


# ---------------------------------------------------------------------------
# Shared evaluation helpers (used by both TaskEvaluator and CategoryTaskEvaluator)
# ---------------------------------------------------------------------------

def _resolve_handle(
    name: str,
    sim: "EnactToMSim",
    world_graph: Optional["Graph"] = None,
) -> str:
    """Resolve task entity name to simulator handle with robust fallbacks."""
    if not name:
        return name

    # Handle agent entities (agent_0, agent_1, etc.) — these aren't in
    # the object managers but we can get their articulated object handle.
    agent_match = re.match(r"^agent_(\d+)$", name)
    if agent_match:
        agent_idx = int(agent_match.group(1))
        try:
            agent_obj = sim.agents_mgr[agent_idx].articulated_agent.sim_obj
            return agent_obj.handle
        except Exception:
            return name

    if world_graph:
        try:
            return world_graph.get_node_from_name(name).sim_handle
        except ValueError:
            pass

    candidate_handles: List[str] = []
    try:
        candidate_handles.extend(sim.get_rigid_object_manager().get_object_handles())
    except Exception:
        pass
    try:
        candidate_handles.extend(sim.get_articulated_object_manager().get_object_handles())
    except Exception:
        pass

    if not candidate_handles:
        return name

    # Remove duplicates while preserving order.
    candidate_handles = list(dict.fromkeys(candidate_handles))
    name_norm = str(name).strip().rstrip("_")

    def _norm_base(handle: str) -> str:
        return str(handle).split(":")[0].strip().rstrip("_")

    exact = [h for h in candidate_handles if h == name]
    if len(exact) == 1:
        return exact[0]

    base_exact = [h for h in candidate_handles if h.split(":")[0] == name]
    if len(base_exact) == 1:
        return base_exact[0]

    normalized = [h for h in candidate_handles if _norm_base(h) == name_norm]
    if len(normalized) == 1:
        return normalized[0]

    prefixed = [h for h in candidate_handles if _norm_base(h).startswith(f"{name_norm}_")]
    if len(prefixed) == 1:
        return prefixed[0]

    suffix = [h for h in candidate_handles if h.endswith(name)]
    if len(suffix) == 1:
        return suffix[0]

    return name


def _resolve_room_id(
    target: Optional[str],
    region_ids: set,
    room_name_map: Dict[str, Any],
) -> Optional[Any]:
    """Resolve a room name to a region ID."""
    if not target:
        return None
    if target in region_ids:
        return target
    return room_name_map.get(target)


def _get_predicate_fn(name: str):
    """Get predicate function by name."""
    if name in _EnactToM_PREDICATE_MAP:
        fn = _EnactToM_PREDICATE_MAP[name]
        if fn is not None:
            return fn
        raise ValueError(f"Predicate '{name}' requires GameStateManager")

    from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates
    return getattr(SimBasedPredicates, name)


def _check_proposition(
    prop: Dict[str, Any],
    sim: "EnactToMSim",
    ao_link_map: Dict,
    region_ids: set,
    room_name_map: Dict[str, Any],
    world_graph: Optional["Graph"] = None,
) -> PropositionResult:
    """Check a single proposition against simulator state."""
    entity = prop.get("entity")
    property_name = prop.get("property")
    target = prop.get("target")
    value = prop.get("value")

    if property_name not in ALL_PREDICATES:
        return PropositionResult(False, {"error": f"Unknown predicate: {property_name}"})

    if property_name in GAME_STATE_PREDICATES:
        return PropositionResult(False, {"error": f"{property_name} requires GameStateManager"})

    predicate_fn = _get_predicate_fn(property_name)
    entity_handle = _resolve_handle(entity, sim, world_graph) if entity else None
    target_handle = _resolve_handle(target, sim, world_graph) if target else None

    # Relational predicates
    if property_name in ("is_on_top", "is_inside"):
        if not target_handle:
            return PropositionResult(False, {"error": f"{property_name} requires 'target'"})
        result = predicate_fn(sim, [entity_handle], [target_handle], ao_link_map=ao_link_map)

    elif property_name == "is_in_room":
        if not target:
            return PropositionResult(False, {"error": "is_in_room requires 'target'"})
        room_id = _resolve_room_id(target, region_ids, room_name_map)
        if room_id is None:
            return PropositionResult(False, {"error": f"Unknown room: {target}"})
        result = predicate_fn(sim, [entity_handle], [room_id], ao_link_map=ao_link_map)

    elif property_name == "is_next_to":
        if not target_handle:
            return PropositionResult(False, {"error": "is_next_to requires 'target'"})
        result = predicate_fn(sim, [entity_handle], [target_handle], ao_link_map=ao_link_map)

    elif property_name == "is_held_by":
        if not target:
            return PropositionResult(False, {"error": "is_held_by requires 'target' (agent_id)"})
        result = predicate_fn(sim, [entity_handle], [target])

    # Unary predicates
    else:
        result = predicate_fn(sim, [entity_handle], ao_link_map=ao_link_map)

    # Convert Habitat predicate result to our format if needed.
    if not isinstance(result, PropositionResult):
        result = PropositionResult(result.is_satisfied, result.info)

    # Handle explicit False value
    if value is False:
        return PropositionResult(not result.is_satisfied, result.info)

    return result


class TaskEvaluator:
    """Evaluates task completion using Habitat and EnactToM predicates."""

    def __init__(
        self,
        success_condition: Dict[str, Any],
        sim: "EnactToMSim",
        world_graph: Optional["Graph"] = None,
    ):
        self.success_condition = success_condition
        self.sim = sim
        self.world_graph = world_graph
        self.required_states = success_condition.get("required_states", [])

        from habitat.sims.habitat_simulator import sim_utilities
        self.ao_link_map = sim_utilities.get_ao_link_id_map(sim)
        self._room_name_map = _build_room_name_map(sim)
        try:
            self._region_ids = {region.id for region in sim.semantic_scene.regions}
        except Exception:
            self._region_ids = set()

    def _check_proposition(self, prop: Dict[str, Any]) -> PropositionResult:
        return _check_proposition(
            prop, self.sim, self.ao_link_map,
            self._region_ids, self._room_name_map, self.world_graph,
        )

    def evaluate(self) -> EvaluationResult:
        """Evaluate task completion."""
        if not self.required_states:
            return EvaluationResult(1.0, True, [], {})

        proposition_status = {}
        failure_explanations = []
        satisfied_count = 0

        for i, prop in enumerate(self.required_states):
            prop_id = prop.get("prop_id", f"prop_{i}")
            try:
                result = self._check_proposition(prop)
                proposition_status[prop_id] = result.is_satisfied
                if result.is_satisfied:
                    satisfied_count += 1
                else:
                    failure_explanations.append(self._explain_failure(prop))
            except Exception as e:
                proposition_status[prop_id] = False
                failure_explanations.append(f"Error checking {prop_id}: {e}")

        percent_complete = satisfied_count / len(self.required_states)
        return EvaluationResult(
            percent_complete=percent_complete,
            success=percent_complete == 1.0,
            failure_explanations=failure_explanations,
            proposition_status=proposition_status,
        )

    def _explain_failure(self, prop: Dict[str, Any]) -> str:
        """Generate failure explanation."""
        entity = prop.get("entity")
        prop_name = prop.get("property")
        target = prop.get("target")

        explanations = {
            "is_on_top": f"{entity} is not on top of {target}",
            "is_inside": f"{entity} is not inside {target}",
            "is_in_room": f"{entity} is not in room {target}",
            "is_next_to": f"{entity} is not next to {target}",
            "is_on_floor": f"{entity} is not on the floor",
            "is_open": f"{entity} is not open",
            "is_closed": f"{entity} is not closed",
            "is_held_by": f"{entity} is not held by {target}",
        }
        return explanations.get(prop_name, f"{entity} is not {prop_name.replace('is_', '')}")


def evaluate_task(
    success_condition: Dict[str, Any],
    sim: "EnactToMSim",
    world_graph: Optional["Graph"] = None,
) -> EvaluationResult:
    """Convenience function to evaluate a task."""
    return TaskEvaluator(success_condition, sim, world_graph).evaluate()


class CategoryTaskEvaluator:
    """
    Category-aware task evaluator.

    Evaluates tasks based on their category:
    - Cooperative: Check required=True subtasks (evaluated each step)
    - Mixed: Check required=True for main goal, required="agent_X" for subgoals
    """

    def __init__(
        self,
        task: "GeneratedTask",
        sim: "EnactToMSim",
        world_graph: Optional["Graph"] = None,
        game_manager: Optional["GameStateManager"] = None,
    ):
        self.task = task
        self.sim = sim
        self.world_graph = world_graph
        self.game_manager = game_manager

        from habitat.sims.habitat_simulator import sim_utilities
        self.ao_link_map = sim_utilities.get_ao_link_id_map(sim)
        self._room_name_map = _build_room_name_map(sim)
        try:
            self._region_ids = {region.id for region in sim.semantic_scene.regions}
        except Exception:
            self._region_ids = set()

    def _check_proposition(self, prop: Dict[str, Any]) -> PropositionResult:
        """Check a single proposition, with game-state predicate support."""
        property_name = prop.get("property")
        if property_name in GAME_STATE_PREDICATES:
            return self._check_game_state_predicate(prop)
        return _check_proposition(
            prop, self.sim, self.ao_link_map,
            self._region_ids, self._room_name_map, self.world_graph,
        )

    def _check_game_state_predicate(self, prop: Dict[str, Any]) -> PropositionResult:
        """Check a game state predicate (requires GameStateManager)."""
        if not self.game_manager:
            return PropositionResult(False, {"error": "GameStateManager required for game state predicates"})

        entity = prop.get("entity")
        property_name = prop.get("property")
        value = prop.get("value")

        if property_name in {"is_unlocked", "is_locked"} and entity:
            state = self.game_manager.get_state()
            is_unlocked = state.get_object_property(entity, "is_unlocked", None)
            if is_unlocked is None:
                is_unlocked = not bool(state.get_object_property(entity, "is_locked", False))

            satisfied = bool(is_unlocked)
            if property_name == "is_locked":
                satisfied = not satisfied
            if value is False:
                satisfied = not satisfied
            return PropositionResult(satisfied, {"entity": entity, "property": property_name})

        return PropositionResult(False, {"error": f"Unknown game state predicate: {property_name}"})

    def evaluate(self):
        """Evaluate task completion using PDDL goals."""
        return self._evaluate_pddl()

    def _evaluate_pddl(self):
        """Evaluate using PDDL goal propositions. Dispatches by category."""
        category = self.task.category

        if category == "cooperative":
            props = self.task.get_required_pddl_propositions()
            if not props:
                return EvaluationResult(0.0, False, ["No PDDL goal propositions found"], {})

            proposition_status = {}
            failure_explanations = []
            satisfied = 0
            for i, prop in enumerate(props):
                prop_id = f"goal_{i}"
                try:
                    result = self._check_proposition(prop)
                    proposition_status[prop_id] = result.is_satisfied
                    if result.is_satisfied:
                        satisfied += 1
                    else:
                        failure_explanations.append(f"goal_{i}: {prop.get('property')}({prop.get('entity')})")
                except Exception as e:
                    proposition_status[prop_id] = False
                    failure_explanations.append(f"Error checking goal_{i}: {e}")

            pct = satisfied / len(props)
            return EvaluationResult(pct, pct == 1.0, failure_explanations, proposition_status)

        if category == "mixed":
            # Evaluate main cooperative goals
            props = self.task.get_required_pddl_propositions()
            # If all goals have owners (no unowned "required" goals),
            # treat all propositions as required to avoid vacuous success
            if not props:
                props = self.task.get_pddl_propositions()
            if not props:
                return MixedResult(False, 0.0, {}, {})
            proposition_status = {}
            satisfied = 0
            for i, prop in enumerate(props):
                pid = f"goal_{i}"
                try:
                    result = self._check_proposition(prop)
                    proposition_status[pid] = result.is_satisfied
                    if result.is_satisfied:
                        satisfied += 1
                except Exception:
                    proposition_status[pid] = False
            main_success = satisfied == len(props)
            main_progress = satisfied / len(props)

            # Evaluate per-agent subgoals
            agent_subgoal_status = {}
            for i in range(self.task.num_agents):
                agent_id = f"agent_{i}"
                agent_props = self.task.get_agent_pddl_propositions(agent_id)
                if agent_props:
                    agent_subgoal_status[agent_id] = all(
                        self._check_proposition(p).is_satisfied for p in agent_props
                    )

            return MixedResult(main_success, main_progress, agent_subgoal_status, proposition_status)

        # Fallback
        props = self.task.get_required_pddl_propositions()
        if not props:
            return EvaluationResult(0.0, False, ["No PDDL goal propositions found"], {})
        return self._evaluate_pddl()


def evaluate_category_task(
    task: "GeneratedTask",
    sim: "EnactToMSim",
    world_graph: Optional["Graph"] = None,
    game_manager: Optional["GameStateManager"] = None,
):
    """
    Evaluate a task based on its category.

    Args:
        task: The generated task to evaluate
        sim: Habitat simulator for physical state predicates
        world_graph: Optional world graph for handle resolution
        game_manager: GameStateManager for overlay predicates such as is_unlocked

    Returns:
    - EvaluationResult for cooperative tasks
    - MixedResult for mixed tasks
    """
    return CategoryTaskEvaluator(task, sim, world_graph, game_manager).evaluate()
