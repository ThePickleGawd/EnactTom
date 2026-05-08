import json

from enacttom.task_gen.runner import (
    _sample_bucket_counts,
    _write_sampled_task_field_views,
    compute_calibration_stats,
)


def _write_task(path, title, tom_level, calibration):
    path.write_text(
        json.dumps(
            {
                "task_id": title,
                "title": title,
                "task": title,
                "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
                "tom_level": tom_level,
                "calibration": calibration,
            }
        )
    )


def test_compute_calibration_stats_ignores_k_zero_and_merges_dirs(tmp_path):
    model = "gpt-5.2"
    seed_dir = tmp_path / "seed"
    out_dir = tmp_path / "out"
    seed_dir.mkdir()
    out_dir.mkdir()

    _write_task(
        seed_dir / "k0.json",
        "k0",
        0,
        [
            {
                "run_mode": "standard",
                "agent_models": {"agent_0": model, "agent_1": model},
                "results": {"passed": True, "progress": 1.0},
            }
        ],
    )
    _write_task(
        seed_dir / "fail.json",
        "fail",
        1,
        [
            {
                "run_mode": "standard",
                "agent_models": {"agent_0": model, "agent_1": model},
                "results": {"passed": False, "progress": 0.2},
            }
        ],
    )
    _write_task(
        out_dir / "pass.json",
        "pass",
        2,
        [
            {
                "run_mode": "standard",
                "agent_models": {"agent_0": model, "agent_1": model},
                "results": {"passed": True, "progress": 1.0},
            }
        ],
    )

    stats = compute_calibration_stats([str(seed_dir), str(out_dir)], model)

    assert stats["excluded_tom_level_zero"] == 1
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["total"] == 2


def test_write_sampled_task_field_views_include_only_useful_fields(tmp_path):
    task_path = tmp_path / "task_1.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "gap-task",
                "title": "Gap Task",
                "task": "Move the bottle to the table.",
                "category": "cooperative",
                "num_agents": 2,
                "active_mechanics": ["room_restriction", "limited_bandwidth"],
                "mechanic_bindings": [{"mechanic_type": "room_restriction"}],
                "problem_pddl": "(:goal (and (is_on_top bottle_1 table_1) (K agent_0 (is_open cabinet_1))))",
                "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
                "agent_secrets": {"agent_0": ["secret"], "agent_1": ["other secret"]},
                "extra_field": "should not survive",
                "calibration": [
                    {
                        "run_mode": "standard",
                        "agent_models": {"agent_0": "gpt-5.2", "agent_1": "gpt-5.2"},
                    },
                ],
            }
        )
    )

    _write_sampled_task_field_views(tmp_path)

    filtered = json.loads((tmp_path / "task_1_fields.json").read_text())

    assert set(filtered) == {
        "task",
        "active_mechanics",
        "mechanic_bindings",
        "agent_secrets",
        "agent_actions",
        "problem_pddl",
        "num_agents",
    }
    assert filtered["task"] == "Move the bottle to the table."
    assert filtered["mechanic_bindings"] == [{"mechanic_type": "room_restriction"}]
    assert filtered["agent_secrets"] == {"agent_0": ["secret"], "agent_1": ["other secret"]}


def test_write_sampled_task_field_views_supports_failed_and_passed_sample_names(tmp_path):
    task_path = tmp_path / "failed_1_25pct.json"
    task_path.write_text(
        json.dumps(
            {
                "title": "Failed Sample",
                "task": "Open the cabinet.",
                "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
                "num_agents": 2,
            }
        )
    )

    _write_sampled_task_field_views(tmp_path)

    filtered = json.loads((tmp_path / "failed_1_25pct_fields.json").read_text())
    assert filtered["task"] == "Open the cabinet."
    assert filtered["num_agents"] == 2


def test_sample_bucket_counts_scale_default_mix_to_thirty_examples():
    assert _sample_bucket_counts(30, fail_ratio=0.8, pass_ratio=0.2) == (24, 6)
    assert _sample_bucket_counts(30, fail_ratio=0.9, pass_ratio=0.1) == (27, 3)
