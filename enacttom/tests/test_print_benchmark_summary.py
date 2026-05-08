from __future__ import annotations

import json
from pathlib import Path

from enacttom.scripts.print_benchmark_summary import collect_benchmark_cost_summary


def test_collect_benchmark_cost_summary_reads_sequential_layout(tmp_path: Path) -> None:
    output_base = tmp_path / "run"
    results_dir = tmp_path / "run-2agents" / "results"
    task_dir = results_dir / "task_1"
    task_dir.mkdir(parents=True)

    (results_dir / "benchmark_summary.json").write_text(
        json.dumps({"results": [{"task_id": "task_1", "success": True}]}),
        encoding="utf-8",
    )
    (task_dir / "api_usage.jsonl").write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "gpt-5.2",
                "api_calls": 2,
                "input_tokens": 1000,
                "output_tokens": 200,
                "cached_input_tokens": 100,
                "cost": 0.011,
                "source": "benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = collect_benchmark_cost_summary(str(output_base), parallel=False, repeat=False)

    assert summary["total_api_calls"] == 2
    assert abs(summary["total_cost"] - 0.011) < 1e-9
    assert summary["models"]["gpt-5.2"]["api_calls"] == 2


def test_collect_benchmark_cost_summary_reads_parallel_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "parallel_run"
    task_dir = output_dir / "task_a" / "benchmark-2agents" / "results" / "task_a"
    task_dir.mkdir(parents=True)

    (task_dir / "api_usage.jsonl").write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "api_calls": 1,
                "input_tokens": 400,
                "output_tokens": 50,
                "cached_input_tokens": 0,
                "cost": 0.007,
                "source": "benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = collect_benchmark_cost_summary(str(output_dir), parallel=True, repeat=False)

    assert summary["total_api_calls"] == 1
    assert abs(summary["total_cost"] - 0.007) < 1e-9
    assert summary["models"]["claude-sonnet-4-6"]["api_calls"] == 1
