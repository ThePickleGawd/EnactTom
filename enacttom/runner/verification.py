"""
Verification runner for EnactToM golden trajectory testing.

This runner is designed for step-by-step trajectory verification without
the overhead of LLM planners. It only provides execute_action() and
evaluate_task() capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import EnactToMBaseRunner

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from enacttom.task_gen import GeneratedTask


class VerificationRunner(EnactToMBaseRunner):
    """
    Simple runner for trajectory verification.

    Unlike BenchmarkRunner, this doesn't create LLM planners.
    Just provides execute_action() and evaluate_task() for step-by-step verification.
    """

    def __init__(self, config):
        super().__init__(config)
        self.task: Optional["GeneratedTask"] = None
        self._llm_client = None

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        task: Optional["GeneratedTask"] = None,
        save_video: Optional[bool] = None,
    ) -> None:
        """
        Setup verification runner.

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Task data with mechanics/bindings
            output_dir: Output directory
            task: Optional GeneratedTask object for full task info
            save_video: Whether to save video. If None, uses config.evaluation.save_video
        """
        self.task = task

        # If task provided but no task_data, convert task to mechanics format
        if task and not task_data:
            task_data = self._task_to_mechanics_dict(task)

        # Get agent_actions and message_targets from task if available
        agent_actions = task.agent_actions if task else None
        message_targets = task.message_targets if task else None

        # Call parent setup (no planners created)
        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions, save_video=save_video, message_targets=message_targets)

        # Setup LLM for perception tools (FindObjectTool, etc.)
        self._setup_llm_for_tools()

    def _setup_llm_for_tools(self) -> None:
        """Setup LLM client for perception tools (FindObjectTool, etc.).

        SKIP for golden trajectory verification - trajectories use exact object IDs
        and don't need FindObjectTool/FindReceptacleTool. This saves ~2-3 seconds
        per verification by avoiding LLM client initialization.
        """
        # Skip LLM setup for verification - golden trajectories have exact IDs
        pass

    def _task_to_mechanics_dict(self, task: "GeneratedTask") -> Dict[str, Any]:
        """Convert GeneratedTask to task data for GameStateManager initialization."""
        result = {}
        if task.mechanic_bindings:
            result["mechanics"] = [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        return result

    def run(self, **kwargs) -> Dict[str, Any]:
        """Not used for verification - actions are executed individually."""
        return {"error": "Use execute_action() directly for verification"}

    def evaluate_task(
        self,
        success_condition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion using PDDL formula evaluation.

        Handles both AND and OR goals by
        evaluating the full PDDL formula structure.
        """
        if success_condition is None and self.task:
            goal_checker = self.task.get_pddl_goal_checker()
            if goal_checker:
                return self._evaluate_formula(goal_checker)

        # Fall back to parent implementation (flat AND evaluation)
        return super().evaluate_task(success_condition)

    def _evaluate_formula(self, goal_checker) -> Dict[str, Any]:
        """Evaluate PDDL formula directly against simulator state (handles OR)."""
        from enacttom.evaluation import _check_proposition
        from enacttom.pddl.dsl import Literal

        sim = self.env_interface.sim
        world_graph = getattr(self.env_interface, "full_world_graph", None)
        if not world_graph and hasattr(self.env_interface, "world_graph"):
            for uid in self.agents.keys():
                if uid in self.env_interface.world_graph:
                    world_graph = self.env_interface.world_graph[uid]
                    break

        # Build predicate check function for formula.evaluate()
        ao_link_map = sim.get_ao_link_map() if hasattr(sim, "get_ao_link_map") else {}
        region_ids = set()
        room_name_map = {}
        if world_graph:
            for room in world_graph.get_all_rooms():
                rname = room.name if hasattr(room, "name") else str(room)
                room_name_map[rname] = room

        def check_fn(predicate: str, args: tuple) -> bool:
            """Check a single predicate against the simulator."""
            prop = {"property": predicate}
            if args:
                prop["entity"] = args[0]
            if len(args) > 1:
                prop["target"] = args[1]
            result = _check_proposition(
                prop, sim, ao_link_map, region_ids, room_name_map,
                world_graph=world_graph,
            )
            return result.is_satisfied

        success = goal_checker.goal.evaluate(check_fn)

        # Collect per-branch failure info for debugging
        failure_explanations = []
        if not success:
            for lit in goal_checker.conjuncts:
                if isinstance(lit, Literal):
                    ok = check_fn(lit.predicate, lit.args)
                    expected = not ok if lit.negated else ok
                    if not expected:
                        desc = f"{lit.args[0]} is not {lit.predicate}"
                        if len(lit.args) > 1:
                            desc += f" {lit.args[1]}"
                        if lit.negated:
                            desc = f"{lit.args[0]} should not be {lit.predicate}"
                            if len(lit.args) > 1:
                                desc += f" {lit.args[1]}"
                        failure_explanations.append(desc)

        return {
            "percent_complete": 1.0 if success else 0.0,
            "success": success,
            "failure_explanations": failure_explanations,
        }
