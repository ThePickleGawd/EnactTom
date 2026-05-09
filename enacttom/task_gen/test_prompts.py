from enacttom.task_gen.prompts import build_external_taskgen_prompt


def test_build_external_taskgen_prompt_inlines_runtime_context(tmp_path):
    sampled_dir = tmp_path / "sampled_tasks"
    sampled_dir.mkdir(parents=True)
    (sampled_dir / "failed_example.json").write_text("{}", encoding="utf-8")

    prompt = build_external_taskgen_prompt(
        working_dir=str(tmp_path),
        task_file=str(tmp_path / "working_task.json"),
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=4,
        subtasks_min=3,
        subtasks_max=6,
        query="Keep it grounded in the scene.",
        verification_feedback={
            "overall_reasoning": "The prior task leaked the plan in secrets.",
            "required_fixes": ["Remove prescriptive language."],
        },
        calibration_stats={
            "model": "gpt-5.2",
            "target_rate": 0.10,
            "rate": 0.45,
        },
        current_k_level=2,
        seed_tasks_dir="/tmp/taskgen/sampled_tasks",
        seed_pass_ratio=0.20,
        seed_fail_ratio=0.80,
    )

    assert "## User Requirements" in prompt
    assert "## Previous ToM Verification Failed" in prompt
    assert prompt.count("## Dataset Calibration") == 1
    assert "## Required K-Level: 2" in prompt
    assert "## Sampled Task Context" in prompt
    assert "## Required Commands" in prompt
    assert "`taskgen verify_task`" in prompt
    assert "benchmark_retry_feedback.md" in prompt
    assert "1. Run `taskgen status`." in prompt
    assert "2. Run `taskgen new_scene N`" in prompt
    assert "Read all 1 sampled tasks" in prompt
    assert "Start with any `*_fields.json` compact views when present" in prompt
    assert "`task`, `active_mechanics`, `mechanic_bindings`, `agent_secrets`, `agent_actions`, `problem_pddl`, and `num_agents`" in prompt
    assert "Open the matching raw task JSON only" in prompt
    assert "Submit only if `gpt-5.4-mini` fails" in prompt
    assert "extra_sections" not in prompt


def test_build_external_taskgen_prompt_omits_sampled_task_context_when_seed_sampling_removed():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="mixed",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=5,
        calibration_stats={},
        current_k_level=1,
        seed_tasks_dir="/tmp/taskgen/sampled_tasks",
        skip_steps=["seed-sampling"],
    )

    assert "## Sampled Task Context" not in prompt
    assert "task_*_fields.json" not in prompt


def test_build_external_taskgen_prompt_does_not_push_easy_tasks():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=4,
        difficulty="easy",
    )

    assert "## Difficulty: EASY" not in prompt
    assert "Generate SIMPLE tasks" not in prompt
    assert "do not weaken secrets" in prompt


def test_build_external_taskgen_prompt_warns_against_hidden_object_id_leaks():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=4,
    )

    assert "do NOT reveal the exact runtime object ID" in prompt
    assert "Do NOT leak hidden target object IDs" in prompt
    assert "NEVER add ignorance lines like 'You do not know where ...'" in prompt
    assert "NEVER add epistemic coaching like 'By the end, you must be confident ...'" in prompt


def test_build_external_taskgen_prompt_encourages_full_supported_mechanic_set_for_hard_tasks():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=4,
        subtasks_min=2,
        subtasks_max=5,
        difficulty="hard",
    )

    assert "remote_control" in prompt
    assert "state_mirroring" in prompt
    assert "inverse_state" in prompt
    assert "standard benchmark progress to stay below 45%" in prompt
    assert "the affected agent's secret may briefly state that mechanic fact in plain language" in prompt
    assert "Use the mechanic that creates the cleanest ToM bottleneck for the scene." in prompt


def test_build_external_taskgen_prompt_uses_hard_cap_language_for_low_target_rate():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=4,
        subtasks_min=2,
        subtasks_max=5,
        calibration_stats={
            "model": "gpt-5.4",
            "target_rate": 0.05,
            "rate": 0.07,
        },
    )

    assert "hard cap is 5%" in prompt
    assert "Anything below the cap is acceptable." in prompt
    assert "Discard any task whose standard pass would leave the calibrated pool above the cap." in prompt
    assert "standard benchmark progress below 45%" in prompt


def test_build_external_taskgen_prompt_removes_secret_strategy_rules_when_requested():
    prompt = build_external_taskgen_prompt(
        working_dir="/tmp/taskgen",
        task_file="/tmp/taskgen/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=4,
        skip_steps=["secret-strategy"],
    )

    assert "NEVER use prescriptive language" not in prompt
    assert "it must NOT tell the agent what message or plan to use" not in prompt
    assert "secrets do not explain the coordination plan" not in prompt
    assert "`secret-strategy`: the prompt omits the rule that forbids strategy instructions in `agent_secrets`." in prompt
