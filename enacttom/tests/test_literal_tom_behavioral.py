from enacttom.pddl.problem_pddl import parse_problem_pddl
from enacttom.pddl.runtime_projection import derive_literal_tom_probes
from enacttom.runner.benchmark import (
    BenchmarkRunner,
    _build_probe_dependency_maps,
    _normalize_probe_answer,
    _probe_answers_match,
)


def _nested_problem_pddl() -> str:
    return (
        "(define (problem nested_k)\n"
        "  (:domain enacttom)\n"
        "  (:objects\n"
        "    agent_0 agent_1 - agent\n"
        "    cabinet_27 - furniture\n"
        "  )\n"
        "  (:init)\n"
        "  (:goal (K agent_0 (K agent_1 (is_open cabinet_27))))\n"
        ")"
    )


def test_build_probe_dependency_maps_treats_only_outer_probe_as_scored() -> None:
    parsed = parse_problem_pddl(_nested_problem_pddl())
    probes = derive_literal_tom_probes(parsed.goal_formula)

    child_by_probe, parent_by_probe, root_probe_ids = _build_probe_dependency_maps(probes)

    outer = next(probe for probe in probes if probe.depth == 2)
    inner = next(probe for probe in probes if probe.depth == 1)

    assert child_by_probe[outer.probe_id] == inner.probe_id
    assert parent_by_probe[inner.probe_id] == outer.probe_id
    assert root_probe_ids == {outer.probe_id}


def test_probe_answers_match_uses_normalized_schema() -> None:
    child_answer = {"predicate": "is_open", "holds": "true", "args": ["cabinet_27"]}
    outer_answer = {"predicate": "is_open", "holds": True, "args": ["cabinet_27"]}

    assert _normalize_probe_answer(child_answer) == {
        "predicate": "is_open",
        "holds": True,
        "args": ["cabinet_27"],
    }
    assert _probe_answers_match(child_answer, outer_answer) is True
    assert _probe_answers_match(child_answer, {"predicate": "unknown", "holds": None, "args": []}) is False


def test_format_literal_tom_probe_question_mentions_behavioral_prediction() -> None:
    parsed = parse_problem_pddl(_nested_problem_pddl())
    probes = derive_literal_tom_probes(parsed.goal_formula)
    outer = next(probe for probe in probes if probe.depth == 2)
    inner = next(probe for probe in probes if probe.depth == 1)

    assert "Predict what agent_1 would report" in BenchmarkRunner._format_literal_tom_probe_question(outer)
    assert "Report whether" in BenchmarkRunner._format_literal_tom_probe_question(inner)
