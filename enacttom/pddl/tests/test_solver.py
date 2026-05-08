"""Tests for PDDL solver and ToM verifier."""

import pytest
from unittest.mock import MagicMock

from enacttom.pddl.dsl import Literal, And, Problem
from enacttom.pddl.domain import ENACTTOM_DOMAIN
from enacttom.pddl.solver import PDKBSolver, _max_epistemic_depth, SolverResult
from enacttom.pddl.dsl import Knows, Believes
from enacttom.pddl.epistemic import ObservabilityModel
from enacttom.pddl.tom_verifier import compute_tom_depth, explain_tom_depth, generate_tom_reasoning
from enacttom.task_gen.task_generator import GeneratedTask


class TestSolver:
    def _make_problem(self, goal, objects=None, init=None):
        return Problem(
            name="test",
            domain_name="enacttom",
            objects=objects or {"cabinet_27": "furniture", "agent_0": "agent"},
            init=init or [],
            goal=goal,
        )

    def test_solvable_simple(self):
        goal = Literal("is_open", ("cabinet_27",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable

    def test_unknown_predicate(self):
        goal = Literal("is_flying", ("cabinet_27",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        assert not result.solvable
        assert "not in domain" in result.error

    def test_unknown_object(self):
        goal = Literal("is_open", ("nonexistent_99",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        assert not result.solvable
        assert "unknown object" in result.error

    def test_no_goal(self):
        problem = self._make_problem(None)
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable
        assert result.belief_depth == 0

    def test_conjunction_solvable(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("cabinet_27", "cabinet_27")),
        ))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable

    def test_solve_time_tracked(self):
        goal = Literal("is_open", ("cabinet_27",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solve_time >= 0

    def test_belief_depth_with_asymmetry(self):
        goal = Literal("is_open", ("cabinet_27",))
        problem = self._make_problem(goal)
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert result.belief_depth == 0

    def test_static_literal_requires_exact_init_match(self):
        """is_inside is achievable via place action, so PDKBSolver treats it as dynamic."""
        goal = Literal("is_inside", ("mug_1", "cabinet_27"))
        problem = self._make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "mug_1": "object",
                "cabinet_27": "furniture",
                "cabinet_30": "furniture",
            },
            init=[Literal("is_inside", ("mug_1", "cabinet_30"))],
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        # is_inside is achievable via place action → dynamic predicate
        assert result.solvable

    def test_static_literal_satisfied_from_exact_init(self):
        goal = Literal("is_inside", ("mug_1", "cabinet_27"))
        problem = self._make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "mug_1": "object",
                "cabinet_27": "furniture",
            },
            init=[Literal("is_inside", ("mug_1", "cabinet_27"))],
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable


class TestEpistemicDepth:
    def test_literal_depth_0(self):
        assert _max_epistemic_depth(Literal("is_open", ("a",))) == 0

    def test_knows_depth_1(self):
        f = Knows("agent_0", Literal("is_open", ("a",)))
        assert _max_epistemic_depth(f) == 1

    def test_nested_knows_depth_2(self):
        inner = Knows("agent_1", Literal("is_open", ("a",)))
        outer = Knows("agent_0", inner)
        assert _max_epistemic_depth(outer) == 2

    def test_triple_nested_depth_3(self):
        l3 = Knows("agent_2", Literal("is_open", ("a",)))
        l2 = Knows("agent_1", l3)
        l1 = Knows("agent_0", l2)
        assert _max_epistemic_depth(l1) == 3

    def test_and_with_epistemic(self):
        f = And(operands=(
            Literal("is_open", ("a",)),
            Knows("agent_0", Literal("is_open", ("b",))),
        ))
        assert _max_epistemic_depth(f) == 1


class TestTomVerifier:
    def _make_task(self, pddl_goal="(is_open cabinet_27)", mechanics=None, num_agents=2):
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = num_agents
        task.initial_states = {}
        task.mechanic_bindings = mechanics or []
        task.message_targets = None
        task.problem_pddl = (
            f"(define (problem test_001)\n"
            f"  (:domain enacttom)\n"
            f"  (:objects\n"
            f"    agent_0 agent_1 - agent\n"
            f"    kitchen_1 - room\n"
            f"    cabinet_27 - furniture\n"
            f"  )\n"
            f"  (:init\n"
            f"    (agent_in_room agent_0 kitchen_1)\n"
            f"    (agent_in_room agent_1 kitchen_1)\n"
            f"    (is_in_room cabinet_27 kitchen_1)\n"
            f"  )\n"
            f"  (:goal {pddl_goal})\n"
            f")"
        )
        return task

    def test_no_asymmetry_depth_0(self):
        task = self._make_task()
        scene = {"rooms": [], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth == 0

    def test_room_restriction_depth_1(self):
        binding = MagicMock()
        binding.mechanic_type = "room_restriction"
        binding.restricted_rooms = ["kitchen_1"]
        binding.for_agents = ["agent_0"]
        binding.trigger_object = None
        binding.target_object = None
        task = self._make_task(
            pddl_goal="(K agent_0 (is_open cabinet_27))",
            mechanics=[binding],
        )
        scene = {"rooms": ["kitchen_1"], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth == 1

    def test_plain_k_goal_is_strict_tom_1(self):
        task = self._make_task(
            pddl_goal="(K agent_0 (is_open cabinet_27))",
            mechanics=[],
        )
        scene = {"rooms": ["kitchen_1"], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth == 1

    def test_room_valued_fact_depth_1(self):
        """Room-valued literals should not become trivially observable."""
        binding = MagicMock()
        binding.mechanic_type = "room_restriction"
        binding.restricted_rooms = ["kitchen_1"]
        binding.for_agents = ["agent_0"]
        binding.trigger_object = None
        binding.target_object = None
        task = self._make_task(
            pddl_goal="(K agent_0 (agent_in_room agent_1 kitchen_1))",
            mechanics=[binding],
        )
        scene = {"rooms": ["kitchen_1"], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth == 1

    def test_explain_provides_reasoning(self):
        task = self._make_task()
        scene = {"rooms": [], "furniture": ["cabinet_27"], "objects": []}
        info = explain_tom_depth(task, scene)
        assert "tom_level" in info
        assert "tom_reasoning" in info
        assert isinstance(info["tom_reasoning"], str)

    def test_unsolvable_returns_neg1(self):
        task = self._make_task(pddl_goal="(is_flying cabinet_27)")
        scene = {"rooms": [], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth == -1


class TestMechanicBindingNormalization:
    def test_shorthand_room_restriction_normalizes_for_strict_tom(self):
        task = GeneratedTask.from_dict(
            {
                "task_id": "test_001",
                "title": "Shorthand Room Restriction",
                "task": "Move the object.",
                "category": "cooperative",
                "scene_id": "scene",
                "episode_id": "episode",
                "active_mechanics": ["room_restriction"],
                "mechanic_bindings": [
                    {
                        "mechanic_type": "room_restriction",
                        "agent_id": "agent_0",
                        "allowed_rooms": ["living_room_1"],
                    }
                ],
                "agent_secrets": {"agent_0": [], "agent_1": []},
                "agent_actions": {"agent_0": ["Communicate"], "agent_1": ["Communicate"]},
                "num_agents": 2,
                "problem_pddl": (
                    "(define (problem test_001)\n"
                    "  (:domain enacttom)\n"
                    "  (:objects\n"
                    "    agent_0 agent_1 - agent\n"
                    "    kitchen_1 living_room_1 - room\n"
                    "    cabinet_27 - furniture\n"
                    "  )\n"
                    "  (:init\n"
                    "    (agent_in_room agent_0 living_room_1)\n"
                    "    (agent_in_room agent_1 kitchen_1)\n"
                    "    (is_in_room cabinet_27 kitchen_1)\n"
                    "  )\n"
                    "  (:goal (K agent_0 (is_open cabinet_27)))\n"
                    ")"
                ),
            }
        )

        obs = ObservabilityModel.from_task_with_scene(task, None)
        assert obs.restricted_rooms == {"agent_0": {"kitchen_1"}}
        assert compute_tom_depth(task, None) == 1


class TestTrivialKGoals:
    """Test that K() goals are checked against observability."""

    def _make_problem(self, goal_str, objects=None):
        from enacttom.pddl.dsl import parse_goal_string
        goal = parse_goal_string(goal_str)
        return Problem(
            name="test",
            domain_name="enacttom",
            objects=objects or {
                "agent_0": "agent", "agent_1": "agent",
                "cabinet_27": "furniture", "drawer_5": "furniture",
            },
            init=[],
            goal=goal,
        )

    def test_trivial_k_when_agent_can_observe(self):
        """K(agent_0, is_open(cabinet_27)) is trivial if agent_0 has no restrictions."""
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={"agent_1": {"kitchen_1"}},  # Only agent_1 restricted
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert len(result.trivial_k_goals) == 1
        assert "(K agent_0 (is_open cabinet_27))" in result.trivial_k_goals

    def test_non_trivial_k_when_agent_restricted(self):
        """K(agent_0, is_open(cabinet_27)) is non-trivial if agent_0 can't see kitchen."""
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},  # agent_0 restricted
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert result.trivial_k_goals == []
        assert result.belief_depth >= 1

    def test_mixed_trivial_and_non_trivial(self):
        """Goal with both trivial and non-trivial K() goals."""
        goal_str = "(and (K agent_0 (is_open cabinet_27)) (K agent_0 (is_open drawer_5)))"
        problem = self._make_problem(goal_str)
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"bedroom_1"}},  # agent_0 can't see bedroom
            object_rooms={
                "cabinet_27": "kitchen_1",   # agent_0 CAN see
                "drawer_5": "bedroom_1",     # agent_0 can't see
            },
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        # cabinet_27 K() is trivial, drawer_5 K() is not
        assert len(result.trivial_k_goals) == 1
        assert "(K agent_0 (is_open cabinet_27))" in result.trivial_k_goals

    def test_no_scene_data_falls_back(self):
        """Without object_rooms, no triviality checking (falls back to syntactic)."""
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},
            # No object_rooms
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert result.trivial_k_goals == []  # Can't determine triviality without scene data

    def test_nested_k_not_zeroed_by_outer_leaf_visibility(self):
        """Fallback heuristic should not erase the outer K() layer for nested goals."""
        problem = self._make_problem(
            "(K agent_0 (K agent_1 (is_open cabinet_27)))",
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "cabinet_27": "furniture",
                "kitchen_1": "room",
                "bedroom_1": "room",
            },
        )
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"bedroom_1"}},  # asymmetry exists, but cabinet remains visible
            object_rooms={"cabinet_27": "kitchen_1", "kitchen_1": "kitchen_1", "bedroom_1": "bedroom_1"},
        )
        result = PDKBSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert result.belief_depth == 1


class TestCommunicationBudget:
    """Test communication budget validation."""

    def _make_problem(self, goal_str, objects=None):
        from enacttom.pddl.dsl import parse_goal_string
        goal = parse_goal_string(goal_str)
        return Problem(
            name="test",
            domain_name="enacttom",
            objects=objects or {
                "agent_0": "agent", "agent_1": "agent",
                "cabinet_27": "furniture", "drawer_5": "furniture",
            },
            init=[],
            goal=goal,
        )

    def test_no_limits_no_warning(self):
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        warning = PDKBSolver().check_communication_budget(problem, obs)
        assert warning is None

    def test_sufficient_budget(self):
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
            message_limits={"agent_0": 3, "agent_1": 3},  # Plenty of budget
        )
        warning = PDKBSolver().check_communication_budget(problem, obs)
        assert warning is None

    def test_insufficient_budget(self):
        """Multiple K() goals for agent_0 but only 1 message allowed from informer."""
        goal_str = "(and (K agent_0 (is_open cabinet_27)) (K agent_0 (is_open drawer_5)))"
        problem = self._make_problem(goal_str)
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1", "bedroom_1"}},
            object_rooms={"cabinet_27": "kitchen_1", "drawer_5": "bedroom_1"},
            message_limits={"agent_0": 0, "agent_1": 1},  # agent_1 can only send 1
        )
        warning = PDKBSolver().check_communication_budget(problem, obs)
        assert warning is not None
        assert "insufficient" in warning.lower()

    def test_trivial_k_goals_dont_need_budget(self):
        """Trivially observable K() goals don't count toward communication budget."""
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={},  # No restrictions — K() goal is trivial
            object_rooms={"cabinet_27": "kitchen_1"},
            message_limits={"agent_0": 0, "agent_1": 0},  # Zero budget
        )
        warning = PDKBSolver().check_communication_budget(problem, obs)
        assert warning is None  # Trivial K() doesn't need communication

    def test_topology_restricts_informers(self):
        """Only agents with a route to the receiver should count toward budget."""
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
            message_limits={"agent_1": 1, "agent_2": 1},
            message_targets={
                "agent_1": {"agent_2"},
                "agent_2": {"agent_1"},
            },
        )
        warning = PDKBSolver().check_communication_budget(problem, obs)
        assert warning is not None
        assert "insufficient" in warning.lower()

    def test_topology_allows_direct_informer(self):
        """A sender with budget and direct access to the receiver should satisfy the check."""
        problem = self._make_problem("(K agent_0 (is_open cabinet_27))")
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
            message_limits={"agent_1": 1},
            message_targets={"agent_1": {"agent_0"}},
        )
        warning = PDKBSolver().check_communication_budget(problem, obs)
        assert warning is None


class TestObservabilityModel:
    def test_restricted_communication_binding_populates_message_targets(self):
        task = MagicMock()
        task.num_agents = 2
        task.mechanic_bindings = [
            MagicMock(
                mechanic_type="restricted_communication",
                allowed_targets={"agent_0": ["agent_1"]},
            )
        ]
        task.message_targets = None

        obs = ObservabilityModel.from_task(task)
        assert obs.message_targets == {"agent_0": {"agent_1"}}


class TestGenerateTomReasoning:
    """Tests for LLM-generated tom_reasoning."""

    SAMPLE_TASK_DATA = {
        "task": "Move the cushion to the kitchen table.",
        "category": "cooperative",
        "num_agents": 2,
        "problem_pddl": "(define (problem test)\n  (:domain enacttom)\n  (:objects agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture)\n  (:init)\n  (:goal (is_on_top cushion_0 cabinet_27)))",
        "agent_secrets": {
            "agent_0": ["You cannot enter kitchen_1."],
            "agent_1": ["You can enter all rooms."],
        },
        "mechanic_bindings": [
            {"mechanic_type": "room_restriction", "restricted_rooms": ["kitchen_1"], "for_agents": ["agent_0"]},
        ],
        "message_targets": None,
    }

    def test_success_returns_llm_response(self):
        """When LLM succeeds, returns its response."""
        from unittest.mock import patch, MagicMock

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "Agent_0 cannot enter kitchen_1 where cabinet_27 is located, "
            "so agent_0 must rely on agent_1 to place the cushion there."
        )

        with patch("habitat_llm.llm.instantiate_llm", return_value=mock_llm, create=True):
            result = generate_tom_reasoning(
                self.SAMPLE_TASK_DATA,
                tom_level=1,
                information_gaps=["agent_0 cannot see rooms: ['kitchen_1']"],
            )

        assert "agent_0" in result.lower() or "agent_1" in result.lower()
        assert len(result) > 20

    def test_raises_on_import_error(self):
        """When habitat_llm is not available, raises instead of falling back."""
        from unittest.mock import patch

        with patch.dict("sys.modules", {"habitat_llm": None, "habitat_llm.llm": None}):
            with pytest.raises(Exception):
                generate_tom_reasoning(
                    self.SAMPLE_TASK_DATA,
                    tom_level=1,
                    information_gaps=["agent_0 cannot see rooms: ['kitchen_1']"],
                )

    def test_raises_on_short_response(self):
        """When LLM returns too-short response, raises RuntimeError."""
        from unittest.mock import patch, MagicMock

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "OK"

        with patch("habitat_llm.llm.instantiate_llm", return_value=mock_llm, create=True):
            with pytest.raises(RuntimeError, match="unusable"):
                generate_tom_reasoning(
                    self.SAMPLE_TASK_DATA,
                    tom_level=1,
                    information_gaps=["agent_0 cannot see rooms: ['kitchen_1']"],
                )
