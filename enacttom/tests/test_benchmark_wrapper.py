from __future__ import annotations

import json
from pathlib import Path

import pytest

from enacttom.benchmark_results import _screen_benchmarkable_tasks, parse_benchmark_results


def test_screen_benchmarkable_tasks_rejects_synthetic_placeholders(tmp_path: Path) -> None:
    valid_task = tmp_path / "valid.json"
    valid_task.write_text(
        json.dumps(
            {
                "task_id": "grounded_task",
                "scene_id": "103997919_171031233",
                "episode_id": "1768",
            }
        ),
        encoding="utf-8",
    )

    invalid_task = tmp_path / "invalid.json"
    invalid_task.write_text(
        json.dumps(
            {
                "task_id": "synthetic_task",
                "scene_id": "synthetic_scene",
                "episode_id": "synthetic_episode",
            }
        ),
        encoding="utf-8",
    )

    valid, errors = _screen_benchmarkable_tasks([valid_task, invalid_task])

    assert valid == [valid_task]
    assert len(errors) == 1
    assert "synthetic_task" in errors[0]
    assert "not benchmarkable" in errors[0]


def test_parse_benchmark_results_merges_agent_group_outputs(tmp_path: Path) -> None:
    output_base = tmp_path / "run"
    results_dir = tmp_path / "run-2agents" / "results"
    results_dir.mkdir(parents=True)

    summary = {
        "run_mode": "standard",
        "results": [
            {
                "task_id": "task_1",
                "title": "Grounded Task",
                "success": True,
                "steps": 5,
                "turns": 3,
                "category": "cooperative",
                "evaluation": {"percent_complete": 1.0},
            }
        ],
    }
    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    parsed = parse_benchmark_results(str(output_base), model="gpt-5.2")

    assert parsed.total == 1
    assert parsed.passed == 1
    assert parsed.failed == 0
    assert parsed.results[0].task_id == "task_1"


def test_parse_benchmark_results_raises_for_all_skipped(tmp_path: Path) -> None:
    output_base = tmp_path / "run"
    results_dir = tmp_path / "run-2agents" / "results"
    results_dir.mkdir(parents=True)

    summary = {
        "results": [
            {
                "task_id": "task_1",
                "title": "Skipped Task",
                "skipped": True,
            }
        ],
    }
    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="zero non-skipped results"):
        parse_benchmark_results(str(output_base), model="gpt-5.2")
