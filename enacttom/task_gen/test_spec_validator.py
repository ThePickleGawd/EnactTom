from enacttom.task_gen.spec_validator import validate_blocking_spec


def _base_task() -> dict:
    return {
        "task_id": "test-task",
        "title": "Test Task",
        "task": "This is a sufficiently long task description.",
        "episode_id": "episode_1",
        "num_agents": 2,
        "mechanic_bindings": [],
        "agent_secrets": {
            "agent_0": [],
            "agent_1": [],
        },
        "agent_actions": {
            "agent_0": ["Communicate", "Wait", "Open"],
            "agent_1": ["Communicate", "Wait", "Open"],
        },
    }


def test_communicate_recipient_counts_as_active_agent() -> None:
    task = _base_task()
    task["golden_trajectory"] = [
        {
            "actions": [
                {"agent": "agent_0", "action": 'Communicate["stand_1 is open", agent_1]'},
                {"agent": "agent_1", "action": "Wait[]"},
            ]
        }
    ]

    errors = validate_blocking_spec(task)

    assert not any("only one active agent" in error for error in errors)


def test_single_actor_without_recipient_still_fails_multi_agent_guard() -> None:
    task = _base_task()
    task["golden_trajectory"] = [
        {
            "actions": [
                {"agent": "agent_0", "action": "Open[stand_1]"},
                {"agent": "agent_1", "action": "Wait[]"},
            ]
        }
    ]

    errors = validate_blocking_spec(task)

    assert any("only one active agent" in error for error in errors)


def test_shorthand_mechanic_bindings_are_accepted() -> None:
    task = _base_task()
    task["problem_pddl"] = (
        "(define (problem test_task)\n"
        "  (:domain enacttom)\n"
        "  (:objects\n"
        "    agent_0 agent_1 - agent\n"
        "    kitchen_1 living_room_1 - room\n"
        "    bottle_1 - object\n"
        "    table_1 - furniture\n"
        "  )\n"
        "  (:init\n"
        "    (agent_in_room agent_0 living_room_1)\n"
        "    (agent_in_room agent_1 kitchen_1)\n"
        "    (is_in_room bottle_1 kitchen_1)\n"
        "    (is_in_room table_1 kitchen_1)\n"
        "  )\n"
        "  (:goal (and (is_on_top bottle_1 table_1) (K agent_0 (is_on_top bottle_1 table_1))))\n"
        ")"
    )
    task["mechanic_bindings"] = [
        {
            "mechanic_type": "room_restriction",
            "agent_id": "agent_0",
            "allowed_rooms": ["living_room_1"],
        },
        {
            "mechanic_type": "limited_bandwidth",
            "agent_id": "agent_1",
            "max_messages": 1,
        },
    ]

    errors = validate_blocking_spec(task)

    assert not any("room_restriction" in error for error in errors)
    assert not any("limited_bandwidth" in error for error in errors)


def test_message_targets_do_not_require_duplicate_restricted_communication_binding() -> None:
    task = _base_task()
    task["message_targets"] = {"agent_0": ["agent_1"]}

    errors = validate_blocking_spec(task)

    assert not any(
        "restricted_communication mechanic is missing" in error for error in errors
    )
