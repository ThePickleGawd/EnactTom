import json

from enacttom.api_costs import format_cost_table, summarize_path_costs, summarize_worker_costs


def test_summarize_worker_costs_merges_trace_and_usage_log(tmp_path):
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()

    (worker_dir / "api_usage.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "openai",
                        "model": "gpt-5.2",
                        "api_calls": 1,
                        "input_tokens": 1000,
                        "output_tokens": 200,
                        "cached_input_tokens": 100,
                        "cost": 0.01,
                        "source": "judge",
                    }
                ),
                json.dumps(
                    {
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "api_calls": 1,
                        "input_tokens": 500,
                        "output_tokens": 50,
                        "cached_input_tokens": 0,
                        "cost": 0.02,
                        "source": "test_task",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    trace_payload = {
        "info": {"model_stats": {"instance_cost": 0.03, "api_calls": 2}},
        "messages": [
            {
                "extra": {
                    "response": {
                        "model": "gpt-5.2-2025-12-11",
                        "usage": {
                            "prompt_tokens": 200,
                            "completion_tokens": 20,
                            "prompt_tokens_details": {"cached_tokens": 10},
                        },
                    },
                    "cost": 0.01,
                }
            },
            {
                "extra": {
                    "response": {
                        "model": "gpt-5.2-2025-12-11",
                        "usage": {
                            "prompt_tokens": 300,
                            "completion_tokens": 30,
                            "prompt_tokens_details": {"cached_tokens": 15},
                        },
                    },
                    "cost": 0.02,
                }
            },
        ],
    }
    (worker_dir / "agent_trace.json").write_text(
        json.dumps(trace_payload),
        encoding="utf-8",
    )

    summary = summarize_worker_costs(worker_dir)

    assert summary["total_api_calls"] == 4
    assert abs(summary["total_cost"] - 0.06) < 1e-9
    assert set(summary["models"].keys()) == {"gpt-5.2", "gpt-5.4"}
    assert summary["models"]["gpt-5.2"]["api_calls"] == 3
    assert summary["models"]["gpt-5.2"]["input_tokens"] == 1500
    assert summary["models"]["gpt-5.2"]["output_tokens"] == 250
    assert summary["models"]["gpt-5.2"]["cached_input_tokens"] == 125


def test_summarize_path_costs_recurses_through_nested_dirs(tmp_path):
    task_a = tmp_path / "task_a"
    task_a.mkdir()
    (task_a / "api_usage.jsonl").write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "gpt-5.2",
                "api_calls": 2,
                "input_tokens": 1200,
                "output_tokens": 180,
                "cached_input_tokens": 100,
                "cost": 0.012,
                "source": "benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    task_b = tmp_path / "nested" / "task_b"
    task_b.mkdir(parents=True)
    (task_b / "api_usage.jsonl").write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "api_calls": 1,
                "input_tokens": 500,
                "output_tokens": 60,
                "cached_input_tokens": 0,
                "cost": 0.009,
                "source": "benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_path_costs(tmp_path)

    assert summary["total_api_calls"] == 3
    assert abs(summary["total_cost"] - 0.021) < 1e-9
    assert summary["models"]["gpt-5.2"]["api_calls"] == 2
    assert summary["models"]["claude-sonnet-4-6"]["api_calls"] == 1


def test_format_cost_table_lists_total_and_per_model():
    lines = format_cost_table(
        {
            "models": {
                "gpt-5.2": {"api_calls": 3, "cost": 0.012, "has_cost": True},
                "claude-sonnet-4-6": {"api_calls": 1, "cost": 0.009, "has_cost": True},
            },
            "total_api_calls": 4,
            "total_cost": 0.021,
            "has_any_cost": True,
            "has_incomplete_costs": False,
        },
        heading="BENCHMARK API COSTS",
    )

    rendered = "\n".join(lines)
    assert "BENCHMARK API COSTS" in rendered
    assert "gpt-5.2" in rendered
    assert "claude-sonnet-4-6" in rendered
    assert "Total API calls:" in rendered
    assert "Total cost:" in rendered
