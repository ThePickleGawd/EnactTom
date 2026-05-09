#!/usr/bin/env python3
"""
Standalone script to verify golden trajectory.

This runs as a subprocess to get a fresh GL context.

Usage:
    python enacttom/task_gen/verify_trajectory.py --task-file <path> --config-name <config>
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path


def parse_action_string(action_str: str) -> tuple:
    """
    Parse benchmark action string.

    Examples:
        'Navigate[table_22]' -> ('Navigate', 'table_22')
        'Place[cup_5, on, table_22, None, None]' -> ('Place', 'cup_5, on, table_22, None, None')
        'Communicate[Hello there]' -> ('Communicate', 'Hello there')
        'Wait' -> ('Wait', None)

    Returns:
        Tuple of (action_name, args_string or None)
    """
    # Allow empty bracket args for actions like Wait[]
    match = re.match(r'(\w+)(?:\[(.*)\])?$', action_str)
    if match:
        action_name, args = match.group(1), match.group(2)
        if args == "":
            args = None
        return action_name, args
    return action_str, None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Verify golden trajectory")
    parser.add_argument("--task-file", required=True, help="Path to task JSON")
    parser.add_argument("--result-file", required=True, help="Path to write result JSON")
    parser.add_argument("--working-dir", default=None, help="Working directory for Hydra config output")
    parser.add_argument("--config-name", default="examples/enacttom_2_robots")
    args = parser.parse_args()

    def write_result(result: dict):
        """Write result to file instead of stdout."""
        with open(args.result_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Load task
    try:
        with open(args.task_file) as f:
            task_data = json.load(f)
    except Exception as e:
        write_result({"valid": False, "error": f"Failed to load task: {e}"})
        sys.exit(1)

    golden = task_data.get("golden_trajectory", [])
    if not golden:
        write_result({"valid": False, "error": "No golden_trajectory found"})
        sys.exit(1)

    # Import and setup Habitat
    try:
        import hydra
        from omegaconf import DictConfig
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        from habitat_llm.agent.env import register_actions, register_measures, register_sensors
        from habitat_llm.agent.env.dataset import EnactToMDatasetV0
        from habitat_llm.agent.env.environment_interface import EnvironmentInterface
        from habitat_llm.utils import fix_config, setup_config

        from enacttom.runner.verification import VerificationRunner
        from enacttom.task_gen import GeneratedTask
        from enacttom.task_gen.scene_loader import apply_agent_spawns
    except ImportError as e:
        write_result({"valid": False, "error": f"Import error: {e}"})
        sys.exit(1)

    # Initialize Hydra config
    try:
        from omegaconf import open_dict, OmegaConf

        GlobalHydra.instance().clear()
        config_dir = str(project_root / "habitat_llm" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        config = compose(config_name=args.config_name)

        # Manually override Hydra interpolations BEFORE fix_config tries to resolve them
        # These would fail with "HydraConfig was not set" otherwise
        # Use working_dir if provided, otherwise fall back to PID-based tmp directory
        if args.working_dir:
            output_dir = f"{args.working_dir}/hydra_verify_{os.getpid()}"
        else:
            output_dir = f"/tmp/enacttom_verify_{os.getpid()}"
        with open_dict(config):
            # Override evaluation.output_dir (uses ${hydra:runtime.output_dir})
            if "evaluation" in config:
                config.evaluation.output_dir = output_dir

            # Override paths that use ${hydra:runtime.output_dir}
            if "paths" in config:
                config.paths.results_dir = f"{output_dir}/results"
                config.paths.epi_result_file_path = f"{output_dir}/results/episode_result_log.csv"
                config.paths.run_result_file_path = f"{output_dir}/results/run_result_log.csv"
                config.paths.end_result_file_path = f"{output_dir}/results/end_result_log.csv"

        fix_config(config)
        config = setup_config(config, seed=47668090)
    except Exception as e:
        write_result({"valid": False, "error": f"Config error: {e}"})
        sys.exit(1)

    # Convert to GeneratedTask
    try:
        task = GeneratedTask.from_dict(task_data)
    except Exception as e:
        write_result({"valid": False, "error": f"Invalid task format: {e}"})
        sys.exit(1)

    # Setup environment
    try:
        register_sensors(config)
        register_actions(config)
        register_measures(config)

        dataset = EnactToMDatasetV0(config.habitat.dataset)
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=True)

        # Load the specific episode from the task
        print(f"Loading episode: {task.episode_id} (scene: {task.scene_id})", file=sys.stderr)

        # Reset with error detection for navmesh issues
        try:
            env_interface.reset_environment(episode_id=task.episode_id)
        except Exception as reset_error:
            error_msg = str(reset_error)
            if "PathFinder" in error_msg or "navigable" in error_msg.lower():
                write_result({
                    "valid": False,
                    "error": f"Scene navmesh error: {error_msg}",
                    "hint": "This scene has navigation issues. Try new_scene[] to get a different scene.",
                    "navmesh_issue": True
                })
                sys.exit(1)
            raise

        # Apply cached spawn positions from task.json (if available)
        apply_agent_spawns(env_interface, task_data.get("agent_spawns", {}))

        runner = VerificationRunner(config)

        # Build task data with mechanic bindings.
        task_mechanics = {}
        if task.mechanic_bindings:
            task_mechanics["mechanics"] = [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        task_mechanics = task_mechanics if task_mechanics else None

        runner.setup(
            env_interface=env_interface,
            task_data=task_mechanics,
            output_dir=output_dir,
            task=task,
            save_video=False,
        )
    except Exception as e:
        write_result({"valid": False, "error": f"Environment setup failed: {e}"})
        sys.exit(1)

    # Execute trajectory
    # New format: each step has "actions" array with all agents' actions for that step
    # Execute all agents' actions CONCURRENTLY within each step
    executed_steps = []
    print(f"\n=== Executing Golden Trajectory ({len(golden)} steps) ===", file=sys.stderr)
    try:
        for step_idx, step in enumerate(golden):
            # Get the actions for this step
            actions = step.get("actions", [])
            if not actions:
                print(f"  Step {step_idx+1}: No actions found in step, skipping", file=sys.stderr)
                continue

            print(f"  Step {step_idx+1}:", file=sys.stderr)
            step_results = []

            # Collect actions to execute concurrently (skip Wait/Communicate)
            concurrent_actions = {}  # uid -> (action_name, target)
            skipped_actions = []

            for action_entry in actions:
                agent_str = action_entry.get("agent", "agent_0")
                parts_split = agent_str.split("_")
                if len(parts_split) < 2:
                    print(f"    Skipping malformed agent string: {agent_str!r}", file=sys.stderr)
                    continue
                try:
                    agent_id = int(parts_split[1])
                except ValueError:
                    print(f"    Skipping non-integer agent id: {agent_str!r}", file=sys.stderr)
                    continue

                # Parse action string: "Navigate[table_22]" -> ("Navigate", "table_22")
                action_str = action_entry.get("action", "")
                action, target = parse_action_string(action_str)

                # Normalize Place relation naming across simulators/PDDL
                if action == "Place" and isinstance(target, str):
                    parts = [p.strip() for p in target.split(",")]
                    if len(parts) >= 2 and parts[1].lower() == "within":
                        parts[1] = "inside"
                        target = ", ".join(parts)

                # Skip Wait actions
                if action == "Wait":
                    print(f"    {agent_str}: Wait [SKIP]", file=sys.stderr)
                    skipped_actions.append({
                        "agent": agent_str, "action": action_str,
                        "success": True, "skipped": True
                    })
                    continue

                # Skip Communicate (but log the message)
                if action == "Communicate":
                    # Extract just the message text for preview
                    raw = target or ""
                    if raw.startswith('"'):
                        end_q = raw.find('"', 1)
                        msg_preview = raw[1:end_q][:50] if end_q > 0 else raw[:50]
                    else:
                        msg_preview = raw[:50]
                    print(f"    {agent_str}: Communicate[\"{msg_preview}...\"] [SKIP]", file=sys.stderr)
                    skipped_actions.append({
                        "agent": agent_str, "action": action_str,
                        "success": True, "skipped": True
                    })
                    continue

                # Add to concurrent execution
                concurrent_actions[agent_id] = (action, target or "")

            # Execute all actions concurrently
            if concurrent_actions:
                results = runner.execute_actions_concurrent(concurrent_actions)

                # Process results
                for agent_id, result in sorted(results.items()):
                    agent_str = f"agent_{agent_id}"
                    action, target = concurrent_actions[agent_id]
                    action_str = f"{action}[{target}]" if target else f"{action}[]"

                    success = result.get("success", False)
                    obs = result.get("observation", "")

                    # Retry Place after re-navigating if it failed due to distance/occlusion
                    if not success and action == "Place" and target:
                        obs_lower = (obs or "").lower()
                        if "too far" in obs_lower or "occluded" in obs_lower or "not close enough" in obs_lower:
                            parts = [p.strip() for p in target.split(",")]
                            nav_target = parts[2] if len(parts) >= 3 else None
                            if nav_target and nav_target not in ("None", "none", ""):
                                print(f"    {agent_str}: Place failed (distance/occlusion), retrying with Navigate[{nav_target}]...", file=sys.stderr)
                                nav_result = runner.execute_actions_concurrent({agent_id: ("Navigate", nav_target)})
                                nav_ok = nav_result.get(agent_id, {}).get("success", False)
                                print(f"    {agent_str}: Navigate[{nav_target}] {'✓' if nav_ok else '✗'}", file=sys.stderr)
                                if nav_ok:
                                    retry_result = runner.execute_actions_concurrent({agent_id: (action, target)})
                                    result = retry_result.get(agent_id, result)
                                    success = result.get("success", False)
                                    obs = result.get("observation", "")
                                    print(f"    {agent_str}: {action_str} retry {'✓' if success else '✗'}", file=sys.stderr)

                    # Print action + observation
                    status = "✓" if success else "✗"
                    print(f"    {agent_str}: {action_str} {status}", file=sys.stderr)
                    if obs:
                        print(f"      → {obs}", file=sys.stderr)

                    step_results.append({
                        "agent": agent_str, "action": action_str,
                        "success": success,
                        "observation": obs[:200] if obs else ""
                    })

                    if not success:
                        # Auto-retry Pick/Place "not close enough" with Navigate
                        obs_lower = (obs or "").lower()
                        retryable = (
                            action in ("Pick", "Place", "Open", "Close")
                            and ("not close enough" in obs_lower or "occluded" in obs_lower
                                 or "postcondition" in obs_lower or "failed to" in obs_lower)
                        )
                        if retryable:
                            # Determine Navigate target
                            if action == "Pick":
                                nav_target = (target or "").split(",")[0].strip()
                            elif action == "Place":
                                # Place format: "obj, on, receptacle, ..."
                                parts = [p.strip() for p in (target or "").split(",")]
                                # Normalize Place relation naming across simulators
                                if len(parts) >= 2 and parts[1].lower() == "within":
                                    parts[1] = "inside"
                                target = ", ".join(parts)
                                nav_target = parts[2] if len(parts) >= 3 else parts[0]
                            else:
                                # Open/Close: target is the object itself
                                nav_target = (target or "").split(",")[0].strip()
                            print(f"    {agent_str}: Auto-retry: Navigate[{nav_target}] then {action}", file=sys.stderr)
                            nav_result = runner.execute_actions_concurrent({agent_id: ("Navigate", nav_target)})
                            retry_result = runner.execute_actions_concurrent({agent_id: (action, target or "")})
                            retry_res = retry_result.get(agent_id, {})
                            if retry_res.get("success", False):
                                retry_obs = retry_res.get("observation", "")
                                print(f"    {agent_str}: {action_str} ✓ (after retry)", file=sys.stderr)
                                step_results[-1] = {
                                    "agent": agent_str, "action": action_str,
                                    "success": True,
                                    "observation": f"(retried) {retry_obs[:180]}" if retry_obs else "(retried)"
                                }
                                continue

                        runner.cleanup()
                        write_result({
                            "valid": False,
                            "failed_step": step_idx,
                            "action": f"{agent_str}: {action_str}",
                            "error": result.get("observation", "Action failed"),
                            "executed_steps": executed_steps + [{"step": step_idx, "actions": step_results + skipped_actions}],
                        })
                        sys.exit(0)

            # Add skipped actions to results
            step_results.extend(skipped_actions)
            executed_steps.append({"step": step_idx, "actions": step_results})

        # Evaluate success
        print(f"\n=== Evaluating Success Condition ===", file=sys.stderr)
        evaluation = runner.evaluate_task()
        success_met = evaluation.get("success", False)
        runner.cleanup()

        if success_met:
            print(f"  Result: SUCCESS", file=sys.stderr)
            write_result({
                "valid": True,
                "steps_executed": len(executed_steps),
                "success_condition_met": True,
                "executed_steps": executed_steps,
            })
        else:
            print(f"  Result: FAILED", file=sys.stderr)
            print(f"  Reason: {evaluation.get('failure_explanations', ['Unknown'])}", file=sys.stderr)
            write_result({
                "valid": False,
                "error": "Success condition not met after trajectory",
                "evaluation": evaluation,
                "executed_steps": executed_steps,
            })

    except Exception as e:
        runner.cleanup()
        write_result({
            "valid": False,
            "error": f"Verification error: {e}",
            "executed_steps": executed_steps,
            "traceback": traceback.format_exc(),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
