"""API usage and cost tracking for EnactToM task generation.

This module combines two sources:

1. A lightweight JSONL usage ledger written by internal LLM wrappers
   (`habitat_llm.llm.*`) when ``ENACTTOM_API_USAGE_LOG`` is set.
2. External taskgen-agent traces such as mini-swe-agent's ``agent_trace.json``.

Pricing notes:
- OpenAI pricing is based on OpenAI's official pricing pages crawled on
  2026-04-07.
- Anthropic pricing is based on Anthropic's official pricing pages crawled on
  2026-04-07.
- Fireworks pricing is based on Fireworks' official pricing pages crawled on
  2026-04-07.

Costs computed from token usage are estimates. Costs read from external traces
are treated as exact when the trace reports them directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
WHITE = "\033[1;37m"


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: Optional[float]
    output_per_million: Optional[float]
    cached_input_per_million: Optional[float] = None


PRICING_BY_MODEL: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-5": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5-chat-latest": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5-mini": ModelPricing(0.25, 2.0, 0.025),
    "gpt-5.1": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5.1-chat-latest": ModelPricing(1.25, 10.0, 0.125),
    "gpt-5.2": ModelPricing(1.75, 14.0, 0.175),
    "gpt-5.2-chat-latest": ModelPricing(1.75, 14.0, 0.175),
    "gpt-5.4": ModelPricing(2.5, 15.0, 0.25),
    "gpt-5.4-mini": ModelPricing(0.75, 4.5, 0.075),
    "o3": ModelPricing(None, None, None),
    # Anthropic
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.30),
    "claude-sonnet-4-5-20250929": ModelPricing(3.0, 15.0, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(1.0, 5.0, 0.10),
    "claude-opus-4-6": ModelPricing(5.0, 25.0, 0.50),
    "claude-opus-4-5-20251101": ModelPricing(5.0, 25.0, 0.50),
    # Fireworks
    "accounts/fireworks/models/kimi-k2p5": ModelPricing(0.60, 3.0, 0.10),
    "kimi-k2.5": ModelPricing(0.60, 3.0, 0.10),
}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_model(model: Optional[str]) -> str:
    raw = (model or "").strip()
    if not raw:
        return "unknown"
    lower = raw.lower()
    if lower.startswith("gpt-5.2-"):
        return "gpt-5.2"
    if lower.startswith("gpt-5.1-"):
        return "gpt-5.1"
    if lower.startswith("gpt-5.4-"):
        return "gpt-5.4"
    if lower.startswith("gpt-5-mini-"):
        return "gpt-5-mini"
    if lower.startswith("gpt-5-"):
        return "gpt-5"
    aliases = {
        "gpt5": "gpt-5",
        "gpt5-mini": "gpt-5-mini",
        "gpt5.1": "gpt-5.1",
        "gpt5.2": "gpt-5.2",
        "gpt5.4": "gpt-5.4",
        "gpt5.4-mini": "gpt-5.4-mini",
        "sonnet": "claude-sonnet-4-6",
        "sonnet-4.6": "claude-sonnet-4-6",
        "sonnet4.6": "claude-sonnet-4-6",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "haiku": "claude-haiku-4-5-20251001",
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "opus": "claude-opus-4-6",
        "opus-4.6": "claude-opus-4-6",
        "opus-4.5": "claude-opus-4-5-20251101",
        "openai/gpt-5": "gpt-5",
        "openai/gpt-5-mini": "gpt-5-mini",
        "openai/gpt-5.1": "gpt-5.1",
        "openai/gpt-5.2": "gpt-5.2",
        "openai/gpt-5.4": "gpt-5.4",
        "openai/gpt-5.4-mini": "gpt-5.4-mini",
        "anthropic/claude-sonnet-4-6": "claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
        "anthropic/claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
        "anthropic/claude-opus-4-6": "claude-opus-4-6",
        "anthropic/claude-opus-4-5-20251101": "claude-opus-4-5-20251101",
    }
    return aliases.get(lower, raw)


def estimate_cost_from_tokens(
    *,
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> Optional[float]:
    normalized = _normalize_model(model)
    pricing = PRICING_BY_MODEL.get(normalized)
    if pricing is None or pricing.input_per_million is None or pricing.output_per_million is None:
        return None

    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    total = 0.0
    total += uncached_input_tokens * pricing.input_per_million / 1_000_000.0
    total += output_tokens * pricing.output_per_million / 1_000_000.0
    if pricing.cached_input_per_million is not None:
        total += cached_input_tokens * pricing.cached_input_per_million / 1_000_000.0
    else:
        total += cached_input_tokens * pricing.input_per_million / 1_000_000.0
    return total


def append_usage_event(log_path: str, payload: Dict[str, Any]) -> None:
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def maybe_append_usage_event(
    *,
    provider: str,
    model: Optional[str],
    usage: Optional[Dict[str, Any]],
    cost: Optional[float] = None,
    source: str,
) -> None:
    log_path = os.environ.get("ENACTTOM_API_USAGE_LOG", "").strip()
    if not log_path:
        return

    usage = usage or {}
    input_tokens = _safe_int(
        usage.get("input_tokens", usage.get("prompt_tokens", usage.get("prompt_token_count")))
    )
    output_tokens = _safe_int(
        usage.get("output_tokens", usage.get("completion_tokens", usage.get("candidates_token_count")))
    )
    cached_input_tokens = _safe_int(
        usage.get("cached_input_tokens", usage.get("cached_tokens", usage.get("cache_read_input_tokens")))
    )
    if cost is None:
        cost = estimate_cost_from_tokens(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    append_usage_event(
        log_path,
        {
            "provider": provider,
            "model": _normalize_model(model),
            "api_calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "cost": cost,
            "cost_is_estimated": cost is not None,
            "source": source,
        },
    )


def _empty_summary() -> Dict[str, Any]:
    return {
        "models": {},
        "total_api_calls": 0,
        "total_cost": 0.0,
        "has_any_cost": False,
        "has_incomplete_costs": False,
    }


def _merge_one(summary: Dict[str, Any], model: str, entry: Dict[str, Any]) -> None:
    bucket = summary["models"].setdefault(
        model,
        {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cost": 0.0,
            "has_cost": False,
            "sources": [],
        },
    )
    bucket["api_calls"] += _safe_int(entry.get("api_calls"))
    bucket["input_tokens"] += _safe_int(entry.get("input_tokens"))
    bucket["output_tokens"] += _safe_int(entry.get("output_tokens"))
    bucket["cached_input_tokens"] += _safe_int(entry.get("cached_input_tokens"))
    if entry.get("source") and entry.get("source") not in bucket["sources"]:
        bucket["sources"].append(entry["source"])
    if entry.get("cost") is not None:
        bucket["cost"] += _safe_float(entry.get("cost"))
        bucket["has_cost"] = True


def read_usage_log(log_path: Path) -> Dict[str, Any]:
    summary = _empty_summary()
    if not log_path.exists():
        return summary

    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        model = _normalize_model(event.get("model"))
        _merge_one(summary, model, event)

    _finalize_summary(summary)
    return summary


def read_external_trace(trace_path: Path) -> Dict[str, Any]:
    summary = _empty_summary()
    if not trace_path.exists():
        return summary

    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return summary

    info = trace.get("info") or {}
    model_stats = info.get("model_stats") or {}
    messages = trace.get("messages") or []

    per_model_calls: Dict[str, int] = {}
    per_model_cost: Dict[str, float] = {}
    per_model_input_tokens: Dict[str, int] = {}
    per_model_output_tokens: Dict[str, int] = {}
    per_model_cached_tokens: Dict[str, int] = {}

    for message in messages:
        extra = message.get("extra") or {}
        response = extra.get("response") or {}
        usage = response.get("usage") or {}
        model = _normalize_model(response.get("model") or info.get("config", {}).get("model", {}).get("model_name"))
        if not model:
            continue
        per_model_calls[model] = per_model_calls.get(model, 0) + 1
        per_model_cost[model] = per_model_cost.get(model, 0.0) + _safe_float(extra.get("cost"))
        per_model_input_tokens[model] = per_model_input_tokens.get(model, 0) + _safe_int(usage.get("prompt_tokens"))
        per_model_output_tokens[model] = per_model_output_tokens.get(model, 0) + _safe_int(usage.get("completion_tokens"))
        per_model_cached_tokens[model] = per_model_cached_tokens.get(model, 0) + _safe_int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        )

    if not per_model_calls:
        model = _normalize_model(info.get("config", {}).get("model", {}).get("model_name"))
        if model != "unknown":
            _merge_one(
                summary,
                model,
                {
                    "api_calls": model_stats.get("api_calls", 0),
                    "cost": model_stats.get("instance_cost"),
                    "source": "external_trace",
                },
            )
            _finalize_summary(summary)
        return summary

    for model, api_calls in per_model_calls.items():
        _merge_one(
            summary,
            model,
            {
                "api_calls": api_calls,
                "input_tokens": per_model_input_tokens.get(model, 0),
                "output_tokens": per_model_output_tokens.get(model, 0),
                "cached_input_tokens": per_model_cached_tokens.get(model, 0),
                "cost": per_model_cost.get(model),
                "source": "external_trace",
            },
        )

    trace_cost = _safe_float(model_stats.get("instance_cost"))
    summed_cost = sum(bucket["cost"] for bucket in summary["models"].values())
    if trace_cost > 0 and abs(summed_cost - trace_cost) > 1e-9 and len(summary["models"]) == 1:
        only_model = next(iter(summary["models"].keys()))
        summary["models"][only_model]["cost"] = trace_cost
        summary["models"][only_model]["has_cost"] = True

    _finalize_summary(summary)
    return summary


def merge_summaries(*summaries: Dict[str, Any]) -> Dict[str, Any]:
    merged = _empty_summary()
    for summary in summaries:
        for model, bucket in (summary or {}).get("models", {}).items():
            _merge_one(merged, model, bucket)
    _finalize_summary(merged)
    return merged


def summarize_worker_costs(worker_dir: Path) -> Dict[str, Any]:
    return merge_summaries(
        read_usage_log(worker_dir / "api_usage.jsonl"),
        read_external_trace(worker_dir / "agent_trace.json"),
    )


def summarize_task_costs(task_dir: Path) -> Dict[str, Any]:
    return merge_summaries(
        read_usage_log(task_dir / "api_usage.jsonl"),
        read_external_trace(task_dir / "agent_trace.json"),
    )


def summarize_path_costs(root_dir: Path) -> Dict[str, Any]:
    merged = _empty_summary()
    if not root_dir.exists():
        return merged

    for usage_log in sorted(root_dir.rglob("api_usage.jsonl")):
        if usage_log.is_file():
            merged = merge_summaries(merged, read_usage_log(usage_log))

    for trace_path in sorted(root_dir.rglob("agent_trace.json")):
        if trace_path.is_file():
            merged = merge_summaries(merged, read_external_trace(trace_path))

    return merged


def summarize_run_costs(run_dir: Path) -> Dict[str, Any]:
    merged = _empty_summary()
    workers_dir = run_dir / "workers"
    if not workers_dir.exists():
        return merged
    for worker_dir in sorted(p for p in workers_dir.iterdir() if p.is_dir()):
        merged = merge_summaries(merged, summarize_worker_costs(worker_dir))
    return merged


def _finalize_summary(summary: Dict[str, Any]) -> None:
    total_cost = 0.0
    has_any_cost = False
    has_incomplete_costs = False
    total_calls = 0
    for bucket in summary["models"].values():
        total_calls += bucket["api_calls"]
        if bucket["has_cost"]:
            has_any_cost = True
            total_cost += bucket["cost"]
        else:
            has_incomplete_costs = True
    summary["total_api_calls"] = total_calls
    summary["total_cost"] = total_cost
    summary["has_any_cost"] = has_any_cost
    summary["has_incomplete_costs"] = has_incomplete_costs


def format_cost_summary(summary: Dict[str, Any], *, heading: str = "API Cost Summary") -> list[str]:
    lines = [
        f"{BOLD}{CYAN}{'=' * 60}{RESET}",
        f"{BOLD}{CYAN}{heading}{RESET}",
        f"{BOLD}{CYAN}{'=' * 60}{RESET}",
    ]
    models = (summary or {}).get("models", {})
    if not models:
        lines.append(f"{YELLOW}No API usage recorded.{RESET}")
        return lines

    for model in sorted(models.keys()):
        bucket = models[model]
        lines.append(f"{BOLD}{WHITE}Model:{RESET} {CYAN}{model}{RESET}")
        lines.append(f"  {BOLD}API calls:{RESET} {WHITE}{bucket['api_calls']}{RESET}")
        if bucket["input_tokens"] or bucket["output_tokens"] or bucket["cached_input_tokens"]:
            lines.append(
                f"  {BOLD}Tokens:{RESET} "
                f"{DIM}input={bucket['input_tokens']}, "
                f"cached_input={bucket['cached_input_tokens']}, "
                f"output={bucket['output_tokens']}{RESET}"
            )
        if bucket["has_cost"]:
            lines.append(f"  {BOLD}Cost:{RESET} {GREEN}${bucket['cost']:.6f}{RESET}")
        else:
            lines.append(f"  {BOLD}Cost:{RESET} {YELLOW}unavailable{RESET}")
    lines.append(f"{BOLD}{WHITE}Net API calls:{RESET} {WHITE}{summary['total_api_calls']}{RESET}")
    if summary["has_any_cost"]:
        suffix = " (partial)" if summary["has_incomplete_costs"] else ""
        cost_color = YELLOW if summary["has_incomplete_costs"] else GREEN
        lines.append(f"{BOLD}{WHITE}Net cost:{RESET} {cost_color}${summary['total_cost']:.6f}{suffix}{RESET}")
    else:
        lines.append(f"{BOLD}{WHITE}Net cost:{RESET} {YELLOW}unavailable{RESET}")
    return lines


def format_cost_table(summary: Dict[str, Any], *, heading: str = "API Cost Summary") -> list[str]:
    lines = [
        f"{BOLD}{CYAN}{'=' * 60}{RESET}",
        f"{BOLD}{CYAN}{heading}{RESET}",
        f"{BOLD}{CYAN}{'=' * 60}{RESET}",
    ]

    models = (summary or {}).get("models", {})
    if not models:
        lines.append(f"{YELLOW}No API usage recorded.{RESET}")
        return lines

    lines.append(
        f"{BOLD}{WHITE}{'Model':<34} {'Calls':>8} {'Cost':>14}{RESET}"
    )
    lines.append(f"{DIM}{'-' * 60}{RESET}")

    sorted_models = sorted(
        models.items(),
        key=lambda item: (
            0 if item[1].get("has_cost") else 1,
            -(item[1].get("cost", 0.0) if item[1].get("has_cost") else 0.0),
            item[0],
        ),
    )

    for model, bucket in sorted_models:
        cost_text = f"${bucket['cost']:.6f}" if bucket.get("has_cost") else "unavailable"
        cost_color = GREEN if bucket.get("has_cost") else YELLOW
        lines.append(
            f"{WHITE}{model:<34}{RESET} "
            f"{CYAN}{bucket.get('api_calls', 0):>8}{RESET} "
            f"{cost_color}{cost_text:>14}{RESET}"
        )

    lines.append(f"{DIM}{'-' * 60}{RESET}")
    lines.append(
        f"{BOLD}{WHITE}Total API calls:{RESET} {CYAN}{summary.get('total_api_calls', 0)}{RESET}"
    )
    if summary.get("has_any_cost"):
        suffix = " (partial)" if summary.get("has_incomplete_costs") else ""
        cost_color = YELLOW if summary.get("has_incomplete_costs") else GREEN
        lines.append(
            f"{BOLD}{WHITE}Total cost:{RESET} "
            f"{cost_color}${summary.get('total_cost', 0.0):.6f}{suffix}{RESET}"
        )
    else:
        lines.append(f"{BOLD}{WHITE}Total cost:{RESET} {YELLOW}unavailable{RESET}")
    return lines
