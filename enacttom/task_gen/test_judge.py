from enacttom.task_gen.judge import _analyze_id_leakage


def test_analyze_id_leakage_flags_public_task_and_ignorance_secret_object_ids():
    task = {
        "task": "Put vase_0 on the living-room table.",
        "agent_secrets": {
            "agent_0": ["You do not know where vase_0 currently is."],
            "agent_1": ["Before the task starts, you saw that vase_0 is on bed_7."],
        },
        "problem_pddl": """(define (problem leak-test)
  (:domain enacttom)
  (:objects
    agent_0 agent_1 - agent
    living_room_1 bedroom_1 - room
    vase_0 - object
    table_9 bed_7 - furniture
  )
  (:init)
  (:goal (and (is_on_top vase_0 table_9)))
)""",
    }

    leakage = _analyze_id_leakage(task)

    assert leakage["public_task_object_ids"] == ["vase_0"]
    assert leakage["ignorance_secret_ids"] == ["vase_0"]
    assert leakage["ignorance_secret_lines"] == ["You do not know where vase_0 currently is."]


def test_analyze_id_leakage_flags_epistemic_and_boilerplate_secret_lines():
    task = {
        "task": "Put the vase on the living-room table.",
        "agent_secrets": {
            "agent_0": [
                "By the end, you must be confident about whether fridge_4 in kitchen_1 is open.",
                "You are agent_0. Shared objective is the public task.",
            ],
            "agent_1": ["Before the task starts, you saw that vase_0 is on bed_7."],
        },
        "problem_pddl": """(define (problem leak-test)
  (:domain enacttom)
  (:objects
    agent_0 agent_1 - agent
    living_room_1 bedroom_1 kitchen_1 - room
    vase_0 - object
    table_9 bed_7 fridge_4 - furniture
  )
  (:init)
  (:goal (and (is_on_top vase_0 table_9) (K agent_0 (is_open fridge_4))))
)""",
    }

    leakage = _analyze_id_leakage(task)

    assert leakage["epistemic_prompt_lines"] == [
        "By the end, you must be confident about whether fridge_4 in kitchen_1 is open."
    ]
    assert leakage["boilerplate_secret_lines"] == [
        "You are agent_0. Shared objective is the public task."
    ]
