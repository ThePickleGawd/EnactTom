#!/usr/bin/env python3
"""
Summarize benchmark results from a run directory.

Usage:
    python -m enacttom.utils.summarize <output_dir>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def load_planner_logs(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all planner log files from a directory."""
    logs = []
    log_dir = output_dir / "planner-log"

    if not log_dir.exists():
        # Try direct path if output_dir is already the planner-log dir
        if output_dir.name == "planner-log":
            log_dir = output_dir
        else:
            return logs

    for log_file in sorted(log_dir.glob("planner-log-*.json")):
        try:
            with open(log_file) as f:
                data = json.load(f)
                data["_log_file"] = str(log_file)
                logs.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {log_file}: {e}")

    return logs


def summarize_results(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from logs."""
    if not logs:
        return {"error": "No logs found"}

    total = len(logs)
    successes = sum(1 for log in logs if log.get("success", False))

    # Collect per-task results
    task_results = []
    total_subtasks = 0
    completed_subtasks = 0
    total_required = 0
    completed_required = 0
    total_steps = 0
    total_turns = 0

    for log in logs:
        eval_data = log.get("evaluation", {})

        task_result = {
            "task_id": log.get("task_id", "unknown"),
            "task_title": log.get("task_title", "Unknown"),
            "success": log.get("success", False),
            "steps": log.get("steps", log.get("sim_steps", 0)),
            "turns": log.get("turns", 0),
            "completed_subtasks": eval_data.get("completed_subtasks", []),
            "total_subtasks": eval_data.get("total_subtasks", 0),
            "required_subtasks": eval_data.get("required_subtasks", 0),
            "completed_required": eval_data.get("completed_required", 0),
            "percent_complete": eval_data.get("percent_complete", 0.0),
            "percent_required_complete": eval_data.get("percent_required_complete", 0.0),
        }
        task_results.append(task_result)

        # Aggregate
        total_subtasks += task_result["total_subtasks"]
        completed_subtasks += len(task_result["completed_subtasks"])
        total_required += task_result["required_subtasks"]
        completed_required += task_result["completed_required"]
        total_steps += task_result["steps"]
        total_turns += task_result["turns"]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_tasks": total,
        "successful_tasks": successes,
        "success_rate": successes / total if total > 0 else 0.0,
        "aggregate": {
            "total_subtasks": total_subtasks,
            "completed_subtasks": completed_subtasks,
            "subtask_completion_rate": completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0,
            "total_required_subtasks": total_required,
            "completed_required_subtasks": completed_required,
            "required_completion_rate": completed_required / total_required if total_required > 0 else 0.0,
            "avg_steps": total_steps / total if total > 0 else 0,
            "avg_turns": total_turns / total if total > 0 else 0,
        },
        "task_results": task_results,
    }

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary."""
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Generated: {summary.get('generated_at', 'N/A')}")
    print()

    print(f"Tasks: {summary['successful_tasks']}/{summary['total_tasks']} successful")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print()

    agg = summary.get("aggregate", {})
    print("Subtask Completion:")
    print(f"  All subtasks: {agg.get('completed_subtasks', 0)}/{agg.get('total_subtasks', 0)} ({agg.get('subtask_completion_rate', 0):.1%})")
    print(f"  Required only: {agg.get('completed_required_subtasks', 0)}/{agg.get('total_required_subtasks', 0)} ({agg.get('required_completion_rate', 0):.1%})")
    print()

    print(f"Average steps: {agg.get('avg_steps', 0):.1f}")
    print(f"Average turns: {agg.get('avg_turns', 0):.1f}")
    print()

    print("-" * 60)
    print("Per-Task Results:")
    print("-" * 60)
    for result in summary.get("task_results", []):
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['task_id']}: {result['task_title']}")
        print(f"    Subtasks: {len(result['completed_subtasks'])}/{result['total_subtasks']} | Required: {result['completed_required']}/{result['required_subtasks']}")
        print(f"    Steps: {result['steps']} | Turns: {result['turns']}")

    print("=" * 60)


def main(output_dir: str) -> None:
    """Main entry point."""
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    logs = load_planner_logs(output_path)

    if not logs:
        print(f"No planner logs found in {output_dir}")
        sys.exit(1)

    summary = summarize_results(logs)

    # Print human-readable summary
    print_summary(summary)

    # Save summary JSON
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m enacttom.utils.summarize <output_dir>")
        sys.exit(1)

    main(sys.argv[1])
