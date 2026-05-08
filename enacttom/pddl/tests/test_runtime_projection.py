from enacttom.pddl.goal_checker import PDDLGoalChecker
from enacttom.pddl.planner import generate_deterministic_trajectory
from enacttom.pddl.problem_pddl import parse_problem_pddl
from enacttom.pddl.runtime_projection import (
    derive_literal_tom_probes,
    evaluate_literal_tom_probe,
    project_runtime_from_parsed_problem,
)


def _task_with_nested_k() -> dict:
    return {
        "task_id": "nested_k",
        "title": "Nested K",
        "category": "mixed",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 2,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem nested_k)\n"
            "  (:domain enacttom)\n"
            "  (:objects\n"
            "    agent_0 agent_1 - agent\n"
            "    cabinet_27 table_5 - furniture\n"
            "    bottle_4 - object\n"
            "    kitchen_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 kitchen_1)\n"
            "    (agent_in_room agent_1 kitchen_1)\n"
            "    (is_in_room cabinet_27 kitchen_1)\n"
            "    (is_in_room table_5 kitchen_1)\n"
            "    (is_in_room bottle_4 kitchen_1)\n"
            "    (is_on_top bottle_4 cabinet_27)\n"
            "    (is_closed cabinet_27)\n"
            "  )\n"
            "  (:goal (and\n"
            "    (is_open cabinet_27)\n"
            "    (K agent_0 (K agent_1 (is_open cabinet_27)))\n"
            "    (is_on_top bottle_4 table_5)\n"
            "  ))\n"
            "  (:goal-owners\n"
            "    (agent_1 (K agent_0 (K agent_1 (is_open cabinet_27))))\n"
            "  )\n"
            ")"
        ),
        "initial_states": {},
    }


def test_runtime_projection_drops_epistemic_conjuncts_and_owners():
    parsed = parse_problem_pddl(_task_with_nested_k()["problem_pddl"])
    projection = project_runtime_from_parsed_problem(parsed)

    assert projection.is_valid
    assert projection.functional_goal_pddl == "(and (is_open cabinet_27) (is_on_top bottle_4 table_5))"
    assert projection.functional_owners == {}
    assert projection.epistemic_conjuncts_removed == 1


def test_runtime_projection_derives_nested_probe():
    parsed = parse_problem_pddl(_task_with_nested_k()["problem_pddl"])
    probes = derive_literal_tom_probes(parsed.goal_formula, owners=parsed.owners)

    assert len(probes) == 2
    outer = next(p for p in probes if p.depth == 2)
    inner = next(p for p in probes if p.depth == 1)
    assert outer.agent_id == "agent_0"
    assert outer.subject_agents == ("agent_1",)
    assert outer.fact_pddl == "(is_open cabinet_27)"
    assert outer.expected_response["predicate"] == "is_open"
    assert outer.expected_response["holds"] is True
    assert outer.expected_response["args"] == ["cabinet_27"]
    assert "predict what agent_1 would report" in outer.question
    assert "is open" in outer.question
    assert 'predicate "unknown"' in outer.question
    assert 'report whether "cabinet 27 is open"' in inner.question
    assert inner.agent_id == "agent_1"


def test_goal_checker_uses_functional_projection_by_default():
    checker = PDDLGoalChecker.from_task_data(_task_with_nested_k())
    assert checker is not None
    assert [c.to_pddl() for c in checker.conjuncts] == [
        "(is_open cabinet_27)",
        "(is_on_top bottle_4 table_5)",
    ]


def test_golden_trajectory_ignores_epistemic_steps():
    result = generate_deterministic_trajectory(_task_with_nested_k())
    actions = [
        action["action"]
        for step in result["trajectory"]
        for action in step["actions"]
        if action["action"] != "Wait[]"
    ]
    assert all("Communicate" not in action for action in actions)
    assert result["communication_derived"] is False
    assert result["functional_goal_pddl"] == "(and (is_open cabinet_27) (is_on_top bottle_4 table_5))"


def test_literal_probe_unknown_response_fails_cleanly():
    parsed = parse_problem_pddl(_task_with_nested_k()["problem_pddl"])
    probe = next(p for p in derive_literal_tom_probes(parsed.goal_formula, owners=parsed.owners) if p.depth == 1)

    passed, details = evaluate_literal_tom_probe(
        probe,
        {"predicate": "unknown", "args": [], "subject_agents": []},
        lambda pred, args: pred == "is_open" and args == ("cabinet_27",),
    )

    assert passed is False
    assert details["parsed_response"]["predicate"] == "unknown"
