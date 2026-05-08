from types import SimpleNamespace

import enacttom.pddl.planner as planner
from enacttom.pddl.dsl import And, Literal, Problem
from enacttom.pddl.planner import generate_deterministic_trajectory


def _actions(result):
    return [
        action["action"]
        for step in result["trajectory"]
        for action in step["actions"]
        if action["action"] != "Wait[]"
    ]


def test_open_inserts_target_navigation_before_interaction():
    task_data = {
        "task_id": "open_target_nav",
        "title": "Open target navigation",
        "category": "cooperative",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 1,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem open_target_nav)\n"
            "  (:domain enacttom)\n"
            "  (:objects\n"
            "    agent_0 - agent\n"
            "    chest_of_drawers_27 - furniture\n"
            "    bedroom_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 bedroom_1)\n"
            "    (is_in_room chest_of_drawers_27 bedroom_1)\n"
            "    (is_closed chest_of_drawers_27)\n"
            "  )\n"
            "  (:goal (is_open chest_of_drawers_27))\n"
            ")"
        ),
        "initial_states": {},
    }

    result = generate_deterministic_trajectory(task_data)

    assert _actions(result) == [
        "Navigate[chest_of_drawers_27]",
        "Open[chest_of_drawers_27]",
    ]


def _step_actions(result, step_idx):
    return {
        action["agent"]: action["action"]
        for action in result["trajectory"][step_idx]["actions"]
    }


def test_parallel_scheduler_merges_independent_agent_workstreams(monkeypatch):
    task_data = {
        "task_id": "parallel_opens",
        "title": "Parallel Opens",
        "category": "cooperative",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 2,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem parallel_opens)\n"
            "  (:domain enacttom)\n"
            "  (:objects\n"
            "    agent_0 agent_1 - agent\n"
            "    cabinet_27 drawer_9 - furniture\n"
            "    kitchen_1 bedroom_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 kitchen_1)\n"
            "    (agent_in_room agent_1 bedroom_1)\n"
            "    (is_in_room cabinet_27 kitchen_1)\n"
            "    (is_in_room drawer_9 bedroom_1)\n"
            "    (is_closed cabinet_27)\n"
            "    (is_closed drawer_9)\n"
            "  )\n"
            "  (:goal (and (is_open cabinet_27) (is_open drawer_9)))\n"
            ")"
        ),
        "initial_states": {},
    }

    problem = Problem(
        name="parallel_opens",
        domain_name="enacttom",
        objects={
            "agent_0": "agent",
            "agent_1": "agent",
            "cabinet_27": "furniture",
            "drawer_9": "furniture",
            "kitchen_1": "room",
            "bedroom_1": "room",
        },
        init=[
            Literal("agent_in_room", ("agent_0", "kitchen_1")),
            Literal("agent_in_room", ("agent_1", "bedroom_1")),
            Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            Literal("is_in_room", ("drawer_9", "bedroom_1")),
            Literal("is_closed", ("cabinet_27",)),
            Literal("is_closed", ("drawer_9",)),
        ],
        goal=And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_open", ("drawer_9",)),
        )),
    )
    solver_result = SimpleNamespace(
        plan=[
            "open(agent_0, cabinet_27, kitchen_1)",
            "open(agent_1, drawer_9, bedroom_1)",
        ],
        belief_depth=0,
    )

    monkeypatch.setattr(
        planner,
        "_solve_task_for_trajectory",
        lambda task_data, scene_data: (problem, None, solver_result),
    )

    result = generate_deterministic_trajectory(task_data)

    assert len(result["trajectory"]) == 2
    assert _step_actions(result, 0) == {
        "agent_0": "Navigate[cabinet_27]",
        "agent_1": "Navigate[drawer_9]",
    }
    assert _step_actions(result, 1) == {
        "agent_0": "Open[cabinet_27]",
        "agent_1": "Open[drawer_9]",
    }


def test_parallel_scheduler_keeps_shared_furniture_dependency_order(monkeypatch):
    task_data = {
        "task_id": "open_then_pick",
        "title": "Open Then Pick",
        "category": "cooperative",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": 2,
        "mechanic_bindings": [],
        "problem_pddl": (
            "(define (problem open_then_pick)\n"
            "  (:domain enacttom)\n"
            "  (:objects\n"
            "    agent_0 agent_1 - agent\n"
            "    cabinet_27 - furniture\n"
            "    bottle_4 - object\n"
            "    kitchen_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (agent_in_room agent_0 kitchen_1)\n"
            "    (agent_in_room agent_1 kitchen_1)\n"
            "    (is_in_room cabinet_27 kitchen_1)\n"
            "    (is_in_room bottle_4 kitchen_1)\n"
            "    (is_inside bottle_4 cabinet_27)\n"
            "    (is_closed cabinet_27)\n"
            "  )\n"
            "  (:goal (is_held_by bottle_4 agent_1))\n"
            ")"
        ),
        "initial_states": {},
    }

    problem = Problem(
        name="open_then_pick",
        domain_name="enacttom",
        objects={
            "agent_0": "agent",
            "agent_1": "agent",
            "cabinet_27": "furniture",
            "bottle_4": "object",
            "kitchen_1": "room",
        },
        init=[
            Literal("agent_in_room", ("agent_0", "kitchen_1")),
            Literal("agent_in_room", ("agent_1", "kitchen_1")),
            Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            Literal("is_in_room", ("bottle_4", "kitchen_1")),
            Literal("is_inside", ("bottle_4", "cabinet_27")),
            Literal("is_closed", ("cabinet_27",)),
        ],
        goal=Literal("is_held_by", ("bottle_4", "agent_1")),
    )
    solver_result = SimpleNamespace(
        plan=[
            "open(agent_0, cabinet_27, kitchen_1)",
            "pick(agent_1, bottle_4, kitchen_1)",
        ],
        belief_depth=0,
    )

    monkeypatch.setattr(
        planner,
        "_solve_task_for_trajectory",
        lambda task_data, scene_data: (problem, None, solver_result),
    )

    result = generate_deterministic_trajectory(task_data)

    assert len(result["trajectory"]) == 3
    assert _step_actions(result, 0) == {
        "agent_0": "Navigate[cabinet_27]",
        "agent_1": "Navigate[bottle_4]",
    }
    assert _step_actions(result, 1) == {
        "agent_0": "Open[cabinet_27]",
        "agent_1": "Wait[]",
    }
    assert _step_actions(result, 2) == {
        "agent_0": "Wait[]",
        "agent_1": "Pick[bottle_4]",
    }
