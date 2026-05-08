"""Tests for Fast Downward solver and domain fixes."""

import os
from pathlib import Path

import pytest

from enacttom.pddl.dsl import (
    And,
    ForallEffect,
    Knows,
    Literal,
    Param,
    Problem,
    parse_goal_string,
)
from enacttom.pddl.domain import ENACTTOM_DOMAIN
from enacttom.pddl.epistemic import ObservabilityModel
from enacttom.pddl.fd_solver import (
    FastDownwardSolver,
    HAS_UP,
    _deduplicate_conjuncts,
    _strip_epistemic,
)
from enacttom.pddl.problem_pddl import (
    ParsedProblemPDDL,
    parse_problem_pddl,
    strip_goal_owners_pddl,
)
from enacttom.pddl.goal_checker import PDDLGoalChecker


# ---------------------------------------------------------------------------
# Epistemic stripping tests
# ---------------------------------------------------------------------------

class TestStripEpistemic:
    def test_literal_unchanged(self):
        lit = Literal("is_open", ("cabinet_27",))
        assert _strip_epistemic(lit) is lit

    def test_knows_unwrapped(self):
        k = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        result = _strip_epistemic(k)
        assert isinstance(result, Literal)
        assert result.predicate == "is_open"

    def test_nested_knows_fully_unwrapped(self):
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        result = _strip_epistemic(outer)
        assert isinstance(result, Literal)
        assert result.predicate == "is_open"

    def test_and_with_epistemic(self):
        goal = And((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        result = _strip_epistemic(goal)
        assert isinstance(result, And)
        assert isinstance(result.operands[0], Literal)
        assert result.operands[0].predicate == "is_open"
        assert isinstance(result.operands[1], Literal)

    def test_dedup_conjuncts(self):
        """Multiple K() goals wrapping same literal → deduplicated."""
        goal = And((
            Literal("is_open", ("cabinet_27",)),
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        result = _deduplicate_conjuncts(goal)
        assert isinstance(result, And)
        assert len(result.operands) == 2


# ---------------------------------------------------------------------------
# ForallEffect tests
# ---------------------------------------------------------------------------

class TestForallEffect:
    def test_simple_forall(self):
        eff = ForallEffect(
            variable=Param("g", "furniture"),
            condition=Literal("mirrors", ("?f", "?g")),
            effect=Literal("is_open", ("?g",)),
        )
        pddl = eff.to_pddl()
        assert "(forall (?g - furniture)" in pddl
        assert "(when (mirrors ?f ?g) (is_open ?g))" in pddl

    def test_forall_with_negative(self):
        eff = ForallEffect(
            variable=Param("old", "room"),
            condition=Literal("agent_in_room", ("?a", "?old")),
            effect=Literal("agent_in_room", ("?a", "?r")),
            negative_effect=Literal("agent_in_room", ("?a", "?old")),
        )
        pddl = eff.to_pddl()
        assert "(forall (?old - room)" in pddl
        assert "(and (agent_in_room ?a ?r) (not (agent_in_room ?a ?old)))" in pddl


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------

class TestDomainFixes:
    def test_domain_has_can_communicate(self):
        pred_names = {p.name for p in ENACTTOM_DOMAIN.predicates}
        assert "can_communicate" in pred_names

    def test_communicate_has_precondition(self):
        comm = next(a for a in ENACTTOM_DOMAIN.actions if a.name == "communicate")
        assert comm.preconditions is not None
        assert "can_communicate" in comm.preconditions.to_pddl()

    def test_navigate_has_forall(self):
        nav = next(a for a in ENACTTOM_DOMAIN.actions if a.name == "navigate")
        has_forall = any(isinstance(e, ForallEffect) for e in nav.effects)
        assert has_forall

    def test_open_has_forall_mirrors(self):
        open_action = next(a for a in ENACTTOM_DOMAIN.actions if a.name == "open")
        forall_effects = [e for e in open_action.effects if isinstance(e, ForallEffect)]
        mirror_foralls = [e for e in forall_effects if e.condition.to_pddl() == "(mirrors ?f ?g)"]
        assert len(mirror_foralls) == 1

    def test_open_inverse_negates(self):
        """Opening an inverse furniture should also negate is_open."""
        from enacttom.pddl.dsl import Effect
        open_action = next(a for a in ENACTTOM_DOMAIN.actions if a.name == "open")
        inverse_effects = [
            e for e in open_action.effects
            if isinstance(e, Effect) and e.condition and "is_inverse" in e.condition.to_pddl()
        ]
        # Should have both: set is_closed AND negate is_open
        assert len(inverse_effects) == 2

    def test_planning_pddl_no_epistemic(self):
        pddl = ENACTTOM_DOMAIN.to_planning_pddl()
        assert ":conditional-effects" in pddl
        assert ":epistemic" not in pddl
        assert "do not edit manually" in pddl


# ---------------------------------------------------------------------------
# Goal owners parser tests
# ---------------------------------------------------------------------------

class TestGoalOwners:
    def test_parse_goal_owners(self):
        pddl = (
            "(define (problem test)\n"
            "  (:domain enacttom)\n"
            "  (:objects agent_0 agent_1 - agent trophy_1 - object cabinet_10 cabinet_20 - furniture)\n"
            "  (:init)\n"
            "  (:goal (and (is_inside trophy_1 cabinet_10) (is_inside trophy_1 cabinet_20)))\n"
            "  (:goal-owners\n"
            "    (agent_0 (is_inside trophy_1 cabinet_10))\n"
            "    (agent_1 (is_inside trophy_1 cabinet_20)))\n"
            ")"
        )
        parsed = parse_problem_pddl(pddl)
        assert "(is_inside trophy_1 cabinet_10)" in parsed.owners
        assert parsed.owners["(is_inside trophy_1 cabinet_10)"] == "agent_0"
        assert parsed.owners["(is_inside trophy_1 cabinet_20)"] == "agent_1"

    def test_no_goal_owners(self):
        pddl = (
            "(define (problem test)\n"
            "  (:domain enacttom)\n"
            "  (:objects)\n"
            "  (:init)\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        parsed = parse_problem_pddl(pddl)
        assert parsed.owners == {}

    def test_strip_goal_owners(self):
        pddl = (
            "(define (problem test)\n"
            "  (:domain enacttom)\n"
            "  (:objects)\n"
            "  (:init)\n"
            "  (:goal (is_open cabinet_27))\n"
            "  (:goal-owners\n"
            "    (agent_0 (is_open cabinet_27)))\n"
            ")"
        )
        stripped = strip_goal_owners_pddl(pddl)
        assert ":goal-owners" not in stripped
        assert ":goal" in stripped

    def test_goal_checker_from_task_data_with_owners(self):
        task_data = {
            "problem_pddl": (
                "(define (problem test)\n"
                "  (:domain enacttom)\n"
                "  (:objects agent_0 agent_1 - agent trophy_1 - object cabinet_10 cabinet_20 - furniture)\n"
                "  (:init)\n"
                "  (:goal (and (is_inside trophy_1 cabinet_10) (is_inside trophy_1 cabinet_20) (is_open cabinet_10)))\n"
                "  (:goal-owners\n"
                "    (agent_0 (is_inside trophy_1 cabinet_10))\n"
                "    (agent_1 (is_inside trophy_1 cabinet_20)))\n"
                ")"
            ),
        }
        checker = PDDLGoalChecker.from_task_data(task_data)
        assert checker is not None
        assert len(checker.conjuncts) == 3
        assert len(checker.get_required_conjuncts()) == 1  # is_open unowned
        assert len(checker.get_agent_conjuncts("agent_0")) == 1
        assert len(checker.get_agent_conjuncts("agent_1")) == 1


# ---------------------------------------------------------------------------
# FastDownwardSolver integration tests
# ---------------------------------------------------------------------------

def _make_problem(goal, objects=None, init=None):
    return Problem(
        name="test",
        domain_name="enacttom",
        objects=objects or {
            "cabinet_27": "furniture",
            "agent_0": "agent",
            "agent_1": "agent",
            "kitchen_1": "room",
        },
        init=init or [
            Literal("agent_in_room", ("agent_0", "kitchen_1")),
            Literal("agent_in_room", ("agent_1", "kitchen_1")),
            Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            Literal("is_closed", ("cabinet_27",)),
            Literal("can_communicate", ("agent_0", "agent_1")),
            Literal("can_communicate", ("agent_1", "agent_0")),
        ],
        goal=goal,
    )


@pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
class TestFastDownwardSolver:
    def test_run_planner_uses_isolated_cwd(self, monkeypatch, tmp_path):
        """Planner.solve should not run in the shared repo cwd."""
        monkeypatch.chdir(tmp_path)
        original_cwd = os.getcwd()
        seen = {}

        class _FakeStatus:
            SOLVED_SATISFICING = "SOLVED_SATISFICING"
            SOLVED_OPTIMALLY = "SOLVED_OPTIMALLY"

            def __init__(self, name):
                self.name = name

            def __eq__(self, other):
                return self.name == getattr(other, "name", other)

        class _FakeAction:
            def __str__(self):
                return "open(agent_0, cabinet_27, kitchen_1)"

        class _FakePlan:
            actions = [_FakeAction()]

        class _FakeResult:
            status = _FakeStatus("SOLVED_SATISFICING")
            plan = _FakePlan()

        class _FakePlanner:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def solve(self, problem, timeout=None):
                seen["cwd"] = os.getcwd()
                Path("output.sas").write_text("planner scratch")
                return _FakeResult()

        monkeypatch.setattr(
            "enacttom.pddl.fd_solver.OneshotPlanner",
            lambda name: _FakePlanner(),
        )

        domain_pddl = ENACTTOM_DOMAIN.to_planning_pddl()
        problem_pddl = (
            "(define (problem test)\n"
            "  (:domain enacttom)\n"
            "  (:objects\n"
            "    agent_0 - agent\n"
            "    cabinet_27 - furniture\n"
            "    kitchen_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 kitchen_1)\n"
            "    (is_in_room cabinet_27 kitchen_1)\n"
            "    (is_closed cabinet_27)\n"
            "  )\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )

        result = FastDownwardSolver()._run_planner(domain_pddl, problem_pddl, timeout=1.0)

        assert result["solvable"]
        assert seen["cwd"] != original_cwd
        assert os.getcwd() == original_cwd
        assert not (tmp_path / "output.sas").exists()

    def test_solvable_simple(self):
        """Simple: cabinet in init is closed, goal is to open it."""
        goal = Literal("is_open", ("cabinet_27",))
        problem = _make_problem(goal)
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable
        assert result.plan is not None
        assert len(result.plan) > 0

    def test_unsolvable_restricted(self):
        """Agent restricted from only room with target, no other agents can help
        because communicate has no physical effects."""
        goal = Literal("is_open", ("cabinet_27",))
        problem = _make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "cabinet_27": "furniture",
                "bedroom_1": "room",
                "kitchen_1": "room",
            },
            init=[
                Literal("agent_in_room", ("agent_0", "bedroom_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                Literal("is_closed", ("cabinet_27",)),
                Literal("is_restricted", ("agent_0", "kitchen_1")),
            ],
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem)
        assert not result.solvable

    def test_epistemic_stripped(self):
        """K() goals unwrapped, physical goal checked."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable

    def test_no_goal_solvable(self):
        problem = _make_problem(None)
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable

    def test_place_achieves_on_top(self):
        """Place action achieves is_on_top."""
        goal = Literal("is_on_top", ("cabinet_27", "cabinet_27"))
        problem = _make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "cabinet_27": "furniture",
                "kitchen_1": "room",
            },
            init=[
                Literal("agent_in_room", ("agent_0", "kitchen_1")),
                Literal("agent_in_room", ("agent_1", "kitchen_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                Literal("is_held_by", ("cabinet_27", "agent_0")),
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_0")),
            ],
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable

    def test_conjunction_solvable(self):
        goal = And((
            Literal("is_open", ("cabinet_27",)),
            Literal("is_held_by", ("cabinet_27", "agent_0")),
        ))
        problem = _make_problem(
            goal,
            init=[
                Literal("agent_in_room", ("agent_0", "kitchen_1")),
                Literal("agent_in_room", ("agent_1", "kitchen_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                Literal("is_closed", ("cabinet_27",)),
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_0")),
            ],
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem)
        assert result.solvable


class TestFastDownwardFallback:
    def test_fallback_when_no_up(self):
        """When HAS_UP is False, falls back to PDKBSolver."""
        import enacttom.pddl.fd_solver as mod
        original = mod.HAS_UP
        try:
            mod.HAS_UP = False
            goal = Literal("is_open", ("cabinet_27",))
            problem = _make_problem(goal)
            solver = FastDownwardSolver()
            result = solver.solve(ENACTTOM_DOMAIN, problem)
            # PDKBSolver does structural check — should still say solvable
            assert result.solvable
        finally:
            mod.HAS_UP = original


# ---------------------------------------------------------------------------
# Compiler can_communicate tests
# ---------------------------------------------------------------------------

class TestCompilerTypeInference:
    """The compiler now requires self-contained raw PDDL instead of inference."""

    def _make_task(self, goal_str, objects_str=""):
        from unittest.mock import MagicMock
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 2
        task.initial_states = {}
        task.mechanic_bindings = []
        task.message_targets = None
        task.problem_pddl = (
            f"(define (problem test_001)\n"
            f"  (:domain enacttom)\n"
            f"  (:objects {objects_str})\n" if objects_str else
            f"(define (problem test_001)\n"
            f"  (:domain enacttom)\n"
        ) + (
            f"  (:init)\n"
            f"  (:goal {goal_str})\n"
            f")"
        )
        return task

    def test_furniture_inferred_from_is_open(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task("(is_open cabinet_27)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_furniture_inferred_from_is_on_top_second_arg(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task("(is_on_top laptop_0 table_29)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_room_inferred_from_agent_in_room(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task("(agent_in_room agent_0 kitchen_1)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_object_inferred_from_is_held_by(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task("(is_held_by mug_1 agent_0)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_epistemic_goal_objects_inferred(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task("(K agent_0 (is_open drawer_5))")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_default_closed_added_for_furniture(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task(
            "(is_open cabinet_27)",
            "agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture",
        )
        task.problem_pddl = task.problem_pddl.replace(
            "  (:init)\n",
            "  (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) "
            "(is_in_room cabinet_27 kitchen_1))\n",
        )
        problem = compile_task(task)
        closed_lits = [l for l in problem.init if l.predicate == "is_closed"]
        assert any(l.args == ("cabinet_27",) for l in closed_lits)

    def test_no_default_closed_if_already_open(self):
        """If init has is_open, don't add is_closed."""
        from unittest.mock import MagicMock
        from enacttom.pddl.compiler import compile_task
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 1
        task.initial_states = {}
        task.mechanic_bindings = []
        task.message_targets = None
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain enacttom)\n"
            "  (:objects agent_0 - agent kitchen_1 - room cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (is_in_room cabinet_27 kitchen_1) (is_open cabinet_27))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        problem = compile_task(task)
        closed_lits = [l for l in problem.init
                       if l.predicate == "is_closed" and l.args == ("cabinet_27",)]
        assert len(closed_lits) == 0

    def test_default_closed_not_added_for_non_articulated_furniture(self):
        from unittest.mock import MagicMock
        from enacttom.pddl.compiler import compile_task

        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 1
        task.initial_states = {}
        task.mechanic_bindings = []
        task.message_targets = None
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain enacttom)\n"
            "  (:objects agent_0 - agent kitchen_1 - room table_13 cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (is_in_room table_13 kitchen_1) (is_in_room cabinet_27 kitchen_1))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )

        problem = compile_task(
            task,
            scene_data={"articulated_furniture": ["cabinet_27"]},
        )
        closed_lits = {(l.predicate, l.args) for l in problem.init if l.predicate == "is_closed"}

        assert ("is_closed", ("cabinet_27",)) in closed_lits
        assert ("is_closed", ("table_13",)) not in closed_lits

    def test_default_closed_uses_conservative_name_fallback_without_scene_data(self):
        from unittest.mock import MagicMock
        from enacttom.pddl.compiler import compile_task

        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 1
        task.initial_states = {}
        task.mechanic_bindings = []
        task.message_targets = None
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain enacttom)\n"
            "  (:objects agent_0 - agent kitchen_1 - room table_13 cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (is_in_room table_13 kitchen_1) (is_in_room cabinet_27 kitchen_1))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )

        problem = compile_task(task)
        closed_lits = {(l.predicate, l.args) for l in problem.init if l.predicate == "is_closed"}

        assert ("is_closed", ("cabinet_27",)) in closed_lits
        assert ("is_closed", ("table_13",)) not in closed_lits


class TestCompilerCanCommunicate:
    def _make_task(self, mechanic_bindings=None, message_targets=None):
        from unittest.mock import MagicMock
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 2
        task.initial_states = {}
        task.mechanic_bindings = mechanic_bindings or []
        task.message_targets = message_targets
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain enacttom)\n"
            "  (:objects agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) (is_in_room cabinet_27 kitchen_1))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        return task

    def test_default_all_pairs(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task()
        problem = compile_task(task)
        can_comm = [
            l for l in problem.init if l.predicate == "can_communicate"
        ]
        # 2 agents → 2 pairs (0→1, 1→0)
        assert len(can_comm) == 2

    def test_message_targets_restricts(self):
        from enacttom.pddl.compiler import compile_task
        task = self._make_task(
            message_targets={"agent_0": ["agent_1"]}  # only 0→1
        )
        problem = compile_task(task)
        can_comm = [
            l for l in problem.init if l.predicate == "can_communicate"
        ]
        assert len(can_comm) == 1
        assert can_comm[0].args == ("agent_0", "agent_1")

    def test_existing_can_communicate_not_duplicated(self):
        """If problem_pddl already has can_communicate, don't add more."""
        from unittest.mock import MagicMock
        from enacttom.pddl.compiler import compile_task
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 2
        task.initial_states = {}
        task.mechanic_bindings = []
        task.message_targets = None
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain enacttom)\n"
            "  (:objects agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) "
            "         (is_in_room cabinet_27 kitchen_1) (can_communicate agent_0 agent_1))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        problem = compile_task(task)
        can_comm = [
            l for l in problem.init if l.predicate == "can_communicate"
        ]
        # Should only have the one from init, not add defaults
        assert len(can_comm) == 1


class TestCompilerRoomRestrictions:
    def _make_task(self, mechanic_bindings=None, problem_pddl=None):
        from unittest.mock import MagicMock

        task = MagicMock()
        task.task_id = "test_room_restrictions"
        task.num_agents = 2
        task.initial_states = {}
        task.mechanic_bindings = mechanic_bindings or []
        task.message_targets = None
        task.problem_pddl = problem_pddl or (
            "(define (problem test_room_restrictions)\n"
            "  (:domain enacttom)\n"
            "  (:objects agent_0 agent_1 - agent kitchen_1 living_room_1 - room cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 living_room_1) (agent_in_room agent_1 kitchen_1) "
            "         (is_in_room cabinet_27 kitchen_1))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        return task

    def test_room_restrictions_derived_from_mechanics(self):
        from enacttom.pddl.compiler import compile_task
        from enacttom.task_gen.task_generator import MechanicBinding

        task = self._make_task(
            mechanic_bindings=[
                MechanicBinding(
                    mechanic_type="room_restriction",
                    restricted_rooms=["kitchen_1"],
                    for_agents=["agent_0"],
                )
            ]
        )

        problem = compile_task(task)
        restrictions = [
            l for l in problem.init if l.predicate == "is_restricted"
        ]

        assert len(restrictions) == 1
        assert restrictions[0].args == ("agent_0", "kitchen_1")

    def test_existing_is_restricted_not_duplicated(self):
        from enacttom.pddl.compiler import compile_task
        from enacttom.task_gen.task_generator import MechanicBinding

        task = self._make_task(
            mechanic_bindings=[
                MechanicBinding(
                    mechanic_type="room_restriction",
                    restricted_rooms=["kitchen_1"],
                    for_agents=["agent_0"],
                )
            ],
            problem_pddl=(
                "(define (problem test_room_restrictions)\n"
                "  (:domain enacttom)\n"
                "  (:objects agent_0 agent_1 - agent kitchen_1 living_room_1 - room cabinet_27 - furniture)\n"
                "  (:init (agent_in_room agent_0 living_room_1) (agent_in_room agent_1 kitchen_1) "
                "         (is_in_room cabinet_27 kitchen_1) (is_restricted agent_0 kitchen_1))\n"
                "  (:goal (is_open cabinet_27))\n"
                ")"
            ),
        )

        problem = compile_task(task)
        restrictions = [
            l for l in problem.init if l.predicate == "is_restricted"
        ]

        assert len(restrictions) == 1
