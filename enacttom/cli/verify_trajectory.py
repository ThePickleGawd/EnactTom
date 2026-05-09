"""
Verify golden trajectory by executing it step-by-step in the Habitat simulator.

Includes static pre-validation (no GL context) followed by full simulation.
Requires GL context — always runs as a subprocess.

Usage:
    # CLI
    python -m enacttom.cli.verify_trajectory task.json --working-dir DIR [--config-name CFG]

    # Agent spawns subprocess:
    subprocess.run([sys.executable, "-m", "enacttom.cli.verify_trajectory", ...])
"""

from __future__ import annotations

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
    match = re.match(r'(\w+)(?:\[(.*)\])?$', action_str)
    if match:
        action_name, args = match.group(1), match.group(2)
        if args == "":
            args = None
        return action_name, args
    return action_str, None


def main():
    parser = argparse.ArgumentParser(description="Verify golden trajectory in simulator")
    parser.add_argument("task_file", help="Path to task JSON")
    parser.add_argument("--working-dir", default=None, help="Working directory for Hydra output")
    parser.add_argument("--config-name", default=None, help="Hydra config name (auto-detected from task)")
    args = parser.parse_args()

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from enacttom.cli import failure, print_result, success

    # Load task
    try:
        with open(args.task_file) as f:
            task_data = json.load(f)
    except Exception as e:
        print_result(failure(f"Failed to load task: {e}"))
        sys.exit(1)

    # Load scene data for deterministic planner (needed before regeneration)
    scene_data_obj = None
    if args.working_dir:
        scene_path = Path(args.working_dir) / "current_scene.json"
        if scene_path.exists():
            try:
                from enacttom.task_gen.scene_loader import SceneData

                with open(scene_path) as sf:
                    sd = json.load(sf)
                scene_data_obj = SceneData(
                    episode_id=sd["episode_id"],
                    scene_id=sd["scene_id"],
                    rooms=sd.get("rooms", []),
                    furniture=sd.get("furniture", []),
                    objects=sd.get("objects", []),
                    articulated_furniture=sd.get("articulated_furniture", []),
                    furniture_in_rooms=sd.get("furniture_in_rooms", {}),
                    objects_on_furniture=sd.get("objects_on_furniture", {}),
                    agent_spawns=sd.get("agent_spawns", {}),
                )
            except Exception:
                pass

    # Validate task structure
    from enacttom.cli.validate_task import static_validate_trajectory, validate

    validation = validate(task_data, scene_data_obj)
    if not validation["success"]:
        print_result(validation)
        sys.exit(1)

    # Deterministically regenerate golden trajectory from authoritative task spec.
    try:
        from enacttom.pddl.planner import regenerate_golden_trajectory

        regen = regenerate_golden_trajectory(
            task_data,
            scene_data=scene_data_obj,
            source="cli_verify",
            task_file=args.task_file,
        )
        print(
            f"Regenerated trajectory: {regen['num_steps']} steps "
            f"(spec_hash={regen['spec_hash'][:8]})",
            file=sys.stderr,
        )
    except Exception as e:
        print_result(failure(f"Failed to regenerate trajectory from task spec: {e}"))
        sys.exit(1)

    golden = task_data.get("golden_trajectory", [])
    if not golden:
        print_result(failure("Deterministic planner produced empty trajectory."))
        sys.exit(1)

    # Static trajectory validation
    static_errors = static_validate_trajectory(task_data, golden, scene_data_obj)
    if static_errors:
        print_result(failure(
            f"Static validation failed: {static_errors[0]}",
            data={"all_errors": static_errors, "hint": "Fix these errors before running simulation."},
        ))
        sys.exit(1)

    # Auto-detect config
    num_agents = task_data.get("num_agents", 2)
    config_name = args.config_name or f"examples/enacttom_{num_agents}_robots"

    # Import and setup Habitat
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict

        from habitat_llm.agent.env import register_actions, register_measures, register_sensors
        from habitat_llm.agent.env.dataset import EnactToMDatasetV0
        from habitat_llm.agent.env.environment_interface import EnvironmentInterface
        from habitat_llm.utils import fix_config, setup_config

        from enacttom.runner.verification import VerificationRunner
        from enacttom.task_gen import GeneratedTask
        from enacttom.task_gen.scene_loader import apply_agent_spawns
    except ImportError as e:
        print_result(failure(f"Import error: {e}"))
        sys.exit(1)

    # Initialize Hydra config
    try:
        GlobalHydra.instance().clear()
        config_dir = str(project_root / "habitat_llm" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        config = compose(config_name=config_name)

        if args.working_dir:
            output_dir = f"{args.working_dir}/hydra_verify_{os.getpid()}"
        else:
            output_dir = f"/tmp/enacttom_verify_{os.getpid()}"
        with open_dict(config):
            if "evaluation" in config:
                config.evaluation.output_dir = output_dir
            if "paths" in config:
                config.paths.results_dir = f"{output_dir}/results"
                config.paths.epi_result_file_path = f"{output_dir}/results/episode_result_log.csv"
                config.paths.run_result_file_path = f"{output_dir}/results/run_result_log.csv"
                config.paths.end_result_file_path = f"{output_dir}/results/end_result_log.csv"

        fix_config(config)
        config = setup_config(config, seed=47668090)
    except Exception as e:
        print_result(failure(f"Config error: {e}"))
        sys.exit(1)

    # Convert to GeneratedTask
    try:
        task = GeneratedTask.from_dict(task_data)
    except Exception as e:
        print_result(failure(f"Invalid task format: {e}"))
        sys.exit(1)

    # Setup environment
    try:
        register_sensors(config)
        register_actions(config)
        register_measures(config)

        dataset = EnactToMDatasetV0(config.habitat.dataset)
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=True)

        print(f"Loading episode: {task.episode_id} (scene: {task.scene_id})", file=sys.stderr)

        try:
            env_interface.reset_environment(episode_id=task.episode_id)
        except Exception as reset_error:
            error_msg = str(reset_error)
            if "PathFinder" in error_msg or "navigable" in error_msg.lower():
                print_result(failure(
                    f"Scene navmesh error: {error_msg}",
                    data={"navmesh_issue": True, "hint": "This scene has navigation issues. Try a different scene."},
                ))
                sys.exit(1)
            raise

        apply_agent_spawns(env_interface, task_data.get("agent_spawns", {}))

        runner = VerificationRunner(config)

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
        print_result(failure(f"Environment setup failed: {e}"))
        sys.exit(1)

    # Execute trajectory
    executed_steps = []
    print(f"\n=== Executing Golden Trajectory ({len(golden)} steps) ===", file=sys.stderr)
    try:
        for step_idx, step in enumerate(golden):
            actions = step.get("actions", [])
            if not actions:
                print(f"  Step {step_idx+1}: No actions found in step, skipping", file=sys.stderr)
                continue

            print(f"  Step {step_idx+1}:", file=sys.stderr)
            step_results = []

            concurrent_actions = {}
            skipped_actions = []

            for action_entry in actions:
                agent_str = action_entry.get("agent", "agent_0")
                agent_id = int(agent_str.split("_")[1])

                action_str = action_entry.get("action", "")
                action, target = parse_action_string(action_str)

                if action == "Wait":
                    print(f"    {agent_str}: Wait [SKIP]", file=sys.stderr)
                    skipped_actions.append({
                        "agent": agent_str, "action": action_str,
                        "success": True, "skipped": True,
                    })
                    continue

                if action == "Communicate":
                    raw = target or ""
                    if raw.startswith('"'):
                        end_q = raw.find('"', 1)
                        msg_preview = raw[1:end_q][:50] if end_q > 0 else raw[:50]
                    else:
                        msg_preview = raw[:50]
                    print(f"    {agent_str}: Communicate[\"{msg_preview}...\"] [SKIP]", file=sys.stderr)
                    skipped_actions.append({
                        "agent": agent_str, "action": action_str,
                        "success": True, "skipped": True,
                    })
                    continue

                concurrent_actions[agent_id] = (action, target or "")

            if concurrent_actions:
                results = runner.execute_actions_concurrent(concurrent_actions)

                for agent_id, result in sorted(results.items()):
                    agent_str = f"agent_{agent_id}"
                    action, target = concurrent_actions[agent_id]
                    action_str = f"{action}[{target}]" if target else f"{action}[]"

                    ok = result.get("success", False)
                    obs = result.get("observation", "")

                    # Retry Place after re-navigating if distance/occlusion
                    if not ok and action == "Place" and target:
                        obs_lower = (obs or "").lower()
                        if "too far" in obs_lower or "occluded" in obs_lower or "not close enough" in obs_lower:
                            parts = [p.strip() for p in target.split(",")]
                            nav_target = parts[2] if len(parts) >= 3 else None
                            if nav_target and nav_target not in ("None", "none", ""):
                                print(f"    {agent_str}: Place failed, retrying with Navigate[{nav_target}]...", file=sys.stderr)
                                nav_result = runner.execute_actions_concurrent({agent_id: ("Navigate", nav_target)})
                                nav_ok = nav_result.get(agent_id, {}).get("success", False)
                                print(f"    {agent_str}: Navigate[{nav_target}] {'OK' if nav_ok else 'FAIL'}", file=sys.stderr)
                                if nav_ok:
                                    retry_result = runner.execute_actions_concurrent({agent_id: (action, target)})
                                    result = retry_result.get(agent_id, result)
                                    ok = result.get("success", False)
                                    obs = result.get("observation", "")
                                    print(f"    {agent_str}: {action_str} retry {'OK' if ok else 'FAIL'}", file=sys.stderr)

                    status = "OK" if ok else "FAIL"
                    print(f"    {agent_str}: {action_str} {status}", file=sys.stderr)
                    if obs:
                        print(f"      -> {obs}", file=sys.stderr)

                    step_results.append({
                        "agent": agent_str, "action": action_str,
                        "success": ok,
                        "observation": obs[:200] if obs else "",
                    })

                    if not ok:
                        obs_lower = (obs or "").lower()
                        retryable = (
                            action in ("Pick", "Place", "Open", "Close")
                            and ("not close enough" in obs_lower or "occluded" in obs_lower
                                 or "postcondition" in obs_lower or "failed to" in obs_lower)
                        )
                        if retryable:
                            if action == "Pick":
                                nav_target = (target or "").split(",")[0].strip()
                            elif action == "Place":
                                parts = [p.strip() for p in (target or "").split(",")]
                                nav_target = parts[2] if len(parts) >= 3 else parts[0]
                            else:
                                nav_target = (target or "").split(",")[0].strip()
                            # Retry up to 3 times (simulator physics can be flaky)
                            retry_succeeded = False
                            for retry_num in range(3):
                                print(f"    {agent_str}: Auto-retry {retry_num+1}/3: Navigate[{nav_target}] then {action}", file=sys.stderr)
                                runner.execute_actions_concurrent({agent_id: ("Navigate", nav_target)})
                                retry_result = runner.execute_actions_concurrent({agent_id: (action, target or "")})
                                retry_res = retry_result.get(agent_id, {})
                                if retry_res.get("success", False):
                                    retry_obs = retry_res.get("observation", "")
                                    print(f"    {agent_str}: {action_str} OK (after retry {retry_num+1})", file=sys.stderr)
                                    step_results[-1] = {
                                        "agent": agent_str, "action": action_str,
                                        "success": True,
                                        "observation": f"(retried x{retry_num+1}) {retry_obs[:170]}" if retry_obs else f"(retried x{retry_num+1})",
                                    }
                                    retry_succeeded = True
                                    break
                            if retry_succeeded:
                                continue

                        runner.cleanup()
                        print_result(failure(
                            result.get("observation", "Action failed"),
                            data={
                                "valid": False,
                                "failed_step": step_idx,
                                "action": f"{agent_str}: {action_str}",
                                "executed_steps": executed_steps + [{"step": step_idx, "actions": step_results + skipped_actions}],
                            },
                        ))
                        sys.exit(0)

            step_results.extend(skipped_actions)
            executed_steps.append({"step": step_idx, "actions": step_results})

        # Evaluate success
        print(f"\n=== Evaluating Success Condition ===", file=sys.stderr)
        evaluation = runner.evaluate_task()
        success_met = evaluation.get("success", False)
        runner.cleanup()

        if success_met:
            print(f"  Result: SUCCESS", file=sys.stderr)
            print_result(success({
                "valid": True,
                "steps_executed": len(executed_steps),
                "success_condition_met": True,
                "executed_steps": executed_steps,
            }))
        else:
            print(f"  Result: FAILED", file=sys.stderr)
            print(f"  Reason: {evaluation.get('failure_explanations', ['Unknown'])}", file=sys.stderr)
            print_result(failure(
                "Success condition not met after trajectory",
                data={
                    "valid": False,
                    "evaluation": evaluation,
                    "executed_steps": executed_steps,
                },
            ))

    except Exception as e:
        runner.cleanup()
        print_result(failure(
            f"Verification error: {e}",
            data={
                "executed_steps": executed_steps,
                "traceback": traceback.format_exc(),
            },
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
