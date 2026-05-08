from __future__ import annotations

from pathlib import Path

from enacttom.benchmark_metrics import build_repeat_summary, build_single_run_summary
from enacttom.benchmark_results import BenchmarkResults, TaskResult


def _make_task_result(task_id: str, success: bool) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        title=task_id,
        task_path=task_id,
        success=success,
        steps=1,
        turns=1,
        percent_complete=1.0 if success else 0.0,
        skipped=False,
        error=None,
        evaluation={"percent_complete": 1.0 if success else 0.0},
        category="cooperative",
        run_mode="standard",
    )


def test_build_repeat_summary_computes_mean_std_and_pass_k_metrics() -> None:
    run_1 = BenchmarkResults(
        model="gpt-5.4",
        total=2,
        passed=1,
        failed=1,
        pass_rate=50.0,
        results=[_make_task_result("task_a", True), _make_task_result("task_b", False)],
    )
    run_2 = BenchmarkResults(
        model="gpt-5.4",
        total=2,
        passed=0,
        failed=2,
        pass_rate=0.0,
        results=[_make_task_result("task_a", False), _make_task_result("task_b", False)],
    )
    run_3 = BenchmarkResults(
        model="gpt-5.4",
        total=2,
        passed=2,
        failed=0,
        pass_rate=100.0,
        results=[_make_task_result("task_a", True), _make_task_result("task_b", True)],
    )

    summary = build_repeat_summary(
        model="gpt-5.4",
        num_times=3,
        runs=[],
        parsed_runs={1: run_1, 2: run_2, 3: run_3},
        expected_task_ids=["task_a", "task_b"],
    )

    assert summary.average_pass_rate == 50.0
    assert summary.std_pass_rate == 50.0
    assert round(summary.pass_at_k or 0.0, 3) == 100.0
    assert round(summary.pass_power_k or 0.0, 3) == 16.667
    assert summary.scored_task_count == 2


def test_build_single_run_summary_matches_pass_rate_for_k_one(tmp_path: Path) -> None:
    parsed = BenchmarkResults(
        model="gpt-5.4",
        total=2,
        passed=1,
        failed=1,
        pass_rate=50.0,
        results=[_make_task_result("task_a", True), _make_task_result("task_b", False)],
    )

    summary = build_single_run_summary(
        model="gpt-5.4",
        output_dir=str(tmp_path),
        parsed=parsed,
    )

    assert summary.num_times == 1
    assert summary.average_pass_rate == 50.0
    assert summary.std_pass_rate == 0.0
    assert summary.pass_at_k == 50.0
    assert summary.pass_power_k == 50.0


def test_build_repeat_summary_counts_missing_attempts_in_n() -> None:
    run_1 = BenchmarkResults(
        model="gpt-5.4",
        total=1,
        passed=1,
        failed=0,
        pass_rate=100.0,
        results=[_make_task_result("task_a", True)],
    )
    run_2 = BenchmarkResults(
        model="gpt-5.4",
        total=1,
        passed=0,
        failed=1,
        pass_rate=0.0,
        results=[_make_task_result("task_a", False)],
    )

    summary = build_repeat_summary(
        model="gpt-5.4",
        num_times=3,
        runs=[],
        parsed_runs={1: run_1, 2: run_2},
        expected_task_ids=["task_a"],
    )

    assert summary.pass_at_k == 100.0
    assert round(summary.pass_power_k or 0.0, 3) == 3.704


def test_build_repeat_summary_defaults_k_to_num_times() -> None:
    run_1 = BenchmarkResults(
        model="gpt-5.4",
        total=1,
        passed=1,
        failed=0,
        pass_rate=100.0,
        results=[_make_task_result("task_a", True)],
    )

    summary = build_repeat_summary(
        model="gpt-5.4",
        num_times=4,
        runs=[],
        parsed_runs={1: run_1},
        expected_task_ids=["task_a"],
    )

    assert summary.num_times == 4
    assert summary.k == 4
