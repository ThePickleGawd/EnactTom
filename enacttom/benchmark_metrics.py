"""Shared helpers for repeated benchmark summaries and reliability metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import comb
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from enacttom.benchmark_results import BenchmarkResults


REPEAT_SUMMARY_FILENAME = "benchmark_repeat_summary.json"


@dataclass
class BenchmarkRepeatRun:
    """One repeated benchmark attempt."""

    run_index: int
    output_dir: str
    status: str
    return_code: int
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: Optional[float] = None
    error: str = ""


@dataclass
class BenchmarkRepeatSummary:
    """Aggregate summary across repeated benchmark runs."""

    model: str
    num_times: int
    k: int
    completed_runs: int
    scored_task_count: int
    average_pass_rate: Optional[float]
    std_pass_rate: float
    pass_at_k: Optional[float]
    pass_power_k: Optional[float]
    runs: List[BenchmarkRepeatRun] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["summary_type"] = "benchmark_repeat"
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkRepeatSummary":
        return cls(
            model=str(payload.get("model", "")),
            num_times=int(payload.get("num_times", 1) or 1),
            k=int(payload.get("k", payload.get("num_times", 1)) or 1),
            completed_runs=int(payload.get("completed_runs", 0) or 0),
            scored_task_count=int(payload.get("scored_task_count", 0) or 0),
            average_pass_rate=_coerce_optional_float(payload.get("average_pass_rate")),
            std_pass_rate=float(payload.get("std_pass_rate", 0.0) or 0.0),
            pass_at_k=_coerce_optional_float(payload.get("pass_at_k")),
            pass_power_k=_coerce_optional_float(payload.get("pass_power_k")),
            runs=[
                BenchmarkRepeatRun(**run_payload)
                for run_payload in payload.get("runs", [])
                if isinstance(run_payload, dict)
            ],
        )


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def repeat_summary_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / REPEAT_SUMMARY_FILENAME


def load_repeat_summary(output_dir: str | Path) -> BenchmarkRepeatSummary:
    with open(repeat_summary_path(output_dir), encoding="utf-8") as f:
        return BenchmarkRepeatSummary.from_dict(json.load(f))


def write_repeat_summary(
    output_dir: str | Path,
    summary: BenchmarkRepeatSummary,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Path:
    summary_path = repeat_summary_path(output_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    payload = summary.to_dict()
    if extra_fields:
        payload.update(extra_fields)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return summary_path


def build_repeat_summary(
    model: str,
    num_times: int,
    runs: List[BenchmarkRepeatRun],
    parsed_runs: Dict[int, BenchmarkResults],
    expected_task_ids: Optional[List[str]] = None,
) -> BenchmarkRepeatSummary:
    """Compute repeated-run reliability metrics from parsed benchmark outputs.

    Uses the exact formulas from the cited pass@k / pass^k article:
    - pass@k = 1 - C(n - c, k) / C(n, k)
    - pass^k = (c / n)^k
    where k == num_times, n is the number of requested repeated attempts per
    task, and c is the number of successful attempts for that task.
    """
    metric_k = num_times
    pass_rates = [results.pass_rate for _, results in sorted(parsed_runs.items())]

    task_successes: Dict[str, int] = {}
    task_order: List[str] = []
    expected_set = set(expected_task_ids or [])

    if expected_task_ids:
        for task_id in expected_task_ids:
            if task_id not in task_successes:
                task_order.append(task_id)
                task_successes[task_id] = 0

    for _, results in sorted(parsed_runs.items()):
        for task_result in results.results:
            task_key = task_result.task_id or task_result.title
            if not task_key:
                continue

            if expected_set and task_key not in expected_set:
                continue
            if task_key not in task_successes:
                task_order.append(task_key)
                task_successes[task_key] = 0
            if task_result.success:
                task_successes[task_key] = task_successes.get(task_key, 0) + 1

    def _pass_at_k_from_counts(n: int, c: int, k: int) -> Optional[float]:
        if n <= 0 or k <= 0 or k > n:
            return None
        return 1.0 - (comb(n - c, k) / comb(n, k))

    def _pass_power_k_from_counts(n: int, c: int, k: int) -> Optional[float]:
        if n <= 0 or k <= 0:
            return None
        return (c / n) ** k

    pass_at_k_values = []
    pass_power_k_values = []
    for task_id in task_order:
        n = num_times
        c = task_successes.get(task_id, 0)
        pass_at_k_value = _pass_at_k_from_counts(n=n, c=c, k=metric_k)
        pass_power_k_value = _pass_power_k_from_counts(n=n, c=c, k=metric_k)
        if pass_at_k_value is not None:
            pass_at_k_values.append(pass_at_k_value)
        if pass_power_k_value is not None:
            pass_power_k_values.append(pass_power_k_value)

    average_pass_rate = mean(pass_rates) if pass_rates else None
    std_pass_rate = stdev(pass_rates) if len(pass_rates) > 1 else 0.0
    pass_at_k = (
        mean(pass_at_k_values) * 100.0
        if pass_at_k_values
        else None
    )
    pass_power_k = (
        mean(pass_power_k_values) * 100.0
        if pass_power_k_values
        else None
    )

    completed_runs = sum(1 for run in runs if run.status in {"complete", "partial"})

    return BenchmarkRepeatSummary(
        model=model,
        num_times=num_times,
        k=metric_k,
        completed_runs=completed_runs,
        scored_task_count=len(task_order),
        average_pass_rate=average_pass_rate,
        std_pass_rate=std_pass_rate,
        pass_at_k=pass_at_k,
        pass_power_k=pass_power_k,
        runs=runs,
    )


def build_single_run_summary(
    model: str,
    output_dir: str,
    parsed: BenchmarkResults,
    return_code: int = 0,
    error: str = "",
) -> BenchmarkRepeatSummary:
    """Normalize a single benchmark result into the repeated-summary schema."""
    status = "complete" if return_code == 0 else "partial"
    run = BenchmarkRepeatRun(
        run_index=1,
        output_dir=output_dir,
        status=status,
        return_code=return_code,
        total=parsed.total,
        passed=parsed.passed,
        failed=parsed.failed,
        pass_rate=parsed.pass_rate,
        error=error,
    )
    return build_repeat_summary(
        model=model,
        num_times=1,
        runs=[run],
        parsed_runs={1: parsed},
    )
