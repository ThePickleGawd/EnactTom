"""
Task-generation command surface for external SWE agents.

Usage:
    python -m enacttom.cli.taskgen --working-dir DIR status
    python -m enacttom.cli.taskgen --working-dir DIR new_scene 3 [--keep]
    python -m enacttom.cli.taskgen --working-dir DIR judge
    python -m enacttom.cli.taskgen --working-dir DIR test_task
    python -m enacttom.cli.taskgen --working-dir DIR verify_task
    python -m enacttom.cli.taskgen --working-dir DIR submit_task
    python -m enacttom.cli.taskgen --working-dir DIR finish
    python -m enacttom.cli.taskgen --working-dir DIR fail "reason"
"""

from __future__ import annotations

import argparse

from enacttom.cli import print_result
from enacttom.task_gen.event_log import append_event, write_worker_snapshot
from enacttom.task_gen.session import TaskGenSession


def main() -> None:
    parser = argparse.ArgumentParser(description="External-agent task-generation commands")
    parser.add_argument("--working-dir", required=True, help="Task generation working directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    new_scene_parser = subparsers.add_parser("new_scene", help="Load or reload a scene")
    new_scene_parser.add_argument("num_agents", type=int, help="Number of agents")
    new_scene_parser.add_argument("--keep", action="store_true", help="Keep current scene/task")

    subparsers.add_parser("judge", help="Judge the current task")
    subparsers.add_parser("test_task", help="Run standard and baseline calibration runs")
    subparsers.add_parser("verify_task", help="Run the pre-submit three-model verification gate")
    subparsers.add_parser("submit_task", help="Submit the current task")
    subparsers.add_parser("status", help="Show current task-generation state")
    subparsers.add_parser("finish", help="Mark the run complete")

    fail_parser = subparsers.add_parser("fail", help="Mark the run failed")
    fail_parser.add_argument("reason", help="Failure reason")

    args = parser.parse_args()
    session = TaskGenSession(args.working_dir)
    generation_worker_dir = session.state.get("generation_worker_dir") or args.working_dir
    event_payload = {"command": args.command}
    if args.command == "new_scene":
        event_payload["num_agents"] = args.num_agents
        event_payload["keep"] = args.keep
    elif args.command == "fail":
        event_payload["reason"] = args.reason
    append_event(generation_worker_dir, "taskgen_command_started", **event_payload)

    if args.command == "new_scene":
        result = session.new_scene(args.num_agents, keep=args.keep)
    elif args.command == "judge":
        result = session.judge()
    elif args.command == "test_task":
        result = session.test_task()
    elif args.command == "verify_task":
        result = session.verify_task()
    elif args.command == "submit_task":
        result = session.submit_task()
    elif args.command == "status":
        result = session.status()
    elif args.command == "finish":
        result = session.finish()
    else:
        result = session.fail(args.reason)

    append_event(
        generation_worker_dir,
        "taskgen_command_finished",
        **event_payload,
        success=result["success"],
        error=result["error"],
        data=result["data"],
    )
    write_worker_snapshot(
        generation_worker_dir,
        status="failed"
        if session.state.get("failed")
        else "finished"
        if session.state.get("finished")
        else "running",
        submitted_count=len(session.state.get("submitted_tasks", [])),
        target_tasks=session.state.get("num_tasks_target"),
        current_task_index=session.state.get("current_task_index"),
        current_k_level=session.state.get("current_k_level"),
        scene_id=session.state.get("scene_id"),
        episode_id=session.state.get("episode_id"),
        finished=session.state.get("finished", False),
        failed=session.state.get("failed", False),
        fail_reason=session.state.get("fail_reason", ""),
        submitted_tasks=session.state.get("submitted_tasks", []),
        last_command=args.command,
        last_command_success=result["success"],
        last_command_error=result["error"],
    )

    print_result(result)


if __name__ == "__main__":
    main()
