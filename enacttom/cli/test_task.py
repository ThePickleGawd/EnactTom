"""
Test a task by running LLM agents against it in the Habitat simulator.

Validates task structure, then runs a full benchmark episode with LLM planners.
Saves agent trajectories, planner traces, and evaluation results.

Requires GL context — always runs as a subprocess.

Usage:
    # CLI
    python -m enacttom.cli.test_task task.json --working-dir DIR [--trajectory-dir DIR]

    # Agent spawns subprocess:
    subprocess.run([sys.executable, "-m", "enacttom.cli.test_task", ...])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test task with LLM agents")
    parser.add_argument("task_file", help="Path to task JSON")
    parser.add_argument("--working-dir", default=None, help="Working directory")
    parser.add_argument("--trajectory-dir", default=None, help="Directory to save agent trajectory files")
    parser.add_argument("--config-name", default=None, help="Hydra config name (auto-detected from task)")
    parser.add_argument("--max-turns", type=int, default=None, help="Max LLM turns (default: 4x golden trajectory)")
    parser.add_argument("--test-model", type=str, default=None, help="Override model for LLM agents")
    parser.add_argument("--run-mode", default="standard", choices=["standard", "baseline"])
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
        import traceback
        print(traceback.format_exc(), file=sys.stderr)

        print_result(failure(f"Failed to load task: {e}"))
        sys.exit(1)

    # Validate task structure first
    from enacttom.cli.validate_task import validate

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

    validation = validate(task_data, scene_data_obj)
    if not validation["success"]:
        print_result(validation)
        sys.exit(1)

    # Auto-detect config
    num_agents = task_data.get("num_agents", 2)
    config_name = args.config_name or f"examples/enacttom_{num_agents}_robots"

    # Import and setup Habitat
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict

        from habitat_llm.agent.env import (
            register_actions,
            register_measures,
            register_sensors,
            remove_visual_sensors,
        )
        from habitat_llm.agent.env.dataset import EnactToMDatasetV0
        from habitat_llm.agent.env.environment_interface import EnvironmentInterface
        from habitat_llm.utils import fix_config, setup_config

        from enacttom.runner import BenchmarkRunner
        from enacttom.runner.benchmark import BenchmarkExecutionError, task_to_instruction
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

        if args.trajectory_dir:
            output_dir = str(Path(args.trajectory_dir).parent / f"hydra_test_{os.getpid()}")
        else:
            output_dir = f"/tmp/enacttom_test_{os.getpid()}"
        with open_dict(config):
            config.benchmark_run_mode = args.run_mode
            is_standard = str(args.run_mode).strip().lower() == "standard"
            if "world_model" in config:
                config.world_model.partial_obs = is_standard
            config.agent_asymmetry = is_standard
            if "evaluation" in config:
                config.evaluation.output_dir = output_dir
            if "paths" in config:
                config.paths.results_dir = f"{output_dir}/results"
                config.paths.epi_result_file_path = f"{output_dir}/results/episode_result_log.csv"
                config.paths.run_result_file_path = f"{output_dir}/results/run_result_log.csv"
                config.paths.end_result_file_path = f"{output_dir}/results/end_result_log.csv"

        fix_config(config)
        config = setup_config(config, seed=47668090)

        # Override agent models if --test-model is specified.
        if args.test_model:
            from enacttom.examples.run_habitat_benchmark import (
                apply_agent_llm_configs,
                detect_llm_provider,
                expand_model_name,
            )

            test_model = expand_model_name(args.test_model)
            test_provider = detect_llm_provider(test_model)
            if not test_provider:
                print_result(failure(f"Unknown provider for test model: {args.test_model}"))
                sys.exit(1)
            agent_model_mapping = {}
            if hasattr(config, "evaluation") and hasattr(config.evaluation, "agents"):
                for agent_id in config.evaluation.agents:
                    agent_model_mapping[agent_id] = {"model": test_model, "llm_provider": test_provider}
            apply_agent_llm_configs(config, agent_model_mapping)
    except Exception as e:
        print_result(failure(f"Config error: {e}"))
        sys.exit(1)

    # Convert to GeneratedTask
    try:
        task = GeneratedTask.from_dict(task_data)
    except Exception as e:
        print_result(failure(f"Invalid task format: {e}"))
        sys.exit(1)

    # Setup environment and run
    try:
        remove_visual_sensors(config)
        register_sensors(config)
        register_actions(config)
        register_measures(config)

        dataset = EnactToMDatasetV0(config.habitat.dataset)
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        print(f"Loading episode: {task.episode_id} (scene: {task.scene_id})", file=sys.stderr)
        env_interface.reset_environment(episode_id=task.episode_id)

        apply_agent_spawns(env_interface, task_data.get("agent_spawns", {}))

        runner = BenchmarkRunner(config)
        runner.setup(
            env_interface=env_interface,
            output_dir=output_dir,
            task=task,
            save_video=False,
        )

        instruction = task_to_instruction(task)

        max_steps = config.habitat.environment.get("max_episode_steps", 20000)

        # Determine turn budget.
        #
        # Older/external task specs may omit golden_trajectory entirely or set it to
        # a non-list placeholder. In those cases, default to a safe minimum rather
        # than crashing with IndexError/TypeError inside len().
        if args.max_turns is not None:
            max_turns = args.max_turns
        else:
            golden_trajectory = task_data.get("golden_trajectory")
            if not isinstance(golden_trajectory, list):
                golden_trajectory = []
            max_turns = max(len(golden_trajectory) * 5, 20)

        results = runner.run(instruction=instruction, max_steps=max_steps, max_turns=max_turns)

        planner_traces = runner.get_planner_traces()
        runner.cleanup()

        action_history = results.get("action_history", [])
        evaluation = results.get("evaluation", {})

        # Save trajectory files if directory specified
        if args.trajectory_dir:
            trajectory_dir = Path(args.trajectory_dir)
            trajectory_dir.mkdir(parents=True, exist_ok=True)

            for agent_id, trace in planner_traces.items():
                trace_file = trajectory_dir / f"{agent_id}.txt"
                with open(trace_file, 'w') as f:
                    f.write(trace)

            result_file = trajectory_dir / "result.txt"
            with open(result_file, 'w') as f:
                f.write("=== BENCHMARK RESULT ===\n\n")
                f.write(f"Success: {results.get('done', False)}\n")
                f.write(f"Steps: {results.get('steps', 0)}\n")
                f.write(f"Turns: {results.get('turns', 0)}\n")
                f.write(f"\n=== EVALUATION ===\n")
                f.write(f"Percent Complete: {evaluation.get('percent_complete', 0):.1%}\n")
                f.write(f"Success: {evaluation.get('success', False)}\n")

                if evaluation.get('failure_explanations'):
                    f.write(f"\nFailure Reasons:\n")
                    for reason in evaluation['failure_explanations']:
                        f.write(f"  - {reason}\n")

                if evaluation.get('proposition_status'):
                    f.write(f"\n=== SUBTASK PROGRESS ===\n")
                    for prop_id, status in evaluation['proposition_status'].items():
                        status_str = "COMPLETE" if status else "INCOMPLETE"
                        f.write(f"  {prop_id}: {status_str}\n")

                f.write(f"\n=== ACTION HISTORY ===\n")
                for record in action_history:
                    f.write(f"[T{record.get('turn', '?')}] {record.get('agent', '?')}: {record.get('action', '?')}\n")

            # Persist task snapshot metadata so judge can reject stale rollouts.
            try:
                from enacttom.pddl.planner import compute_task_spec_hash

                snapshot = {
                    "task_id": task_data.get("task_id"),
                    "title": task_data.get("title"),
                    "task": task_data.get("task"),
                    "scene_id": task_data.get("scene_id"),
                    "episode_id": task_data.get("episode_id"),
                    "spec_hash": compute_task_spec_hash(task_data),
                    "saved_at": datetime.now().isoformat(),
                }
                with open(trajectory_dir / "task_snapshot.json", "w") as f:
                    json.dump(snapshot, f, indent=2)
            except Exception:
                # Rollout trace remains usable even if snapshot persistence fails.
                pass

        print_result(success({
            "steps": results.get("steps", 0),
            "turns": results.get("turns", 0),
            "done": results.get("done", False),
            "episode_over": results.get("episode_over", False),
            "run_mode": args.run_mode,
            "summary": (
                f"Task {'completed' if results.get('done') else 'not completed'} "
                f"in {results.get('turns', 0)} turns ({results.get('steps', 0)} steps) "
                f"[mode={args.run_mode}]"
            ),
            "action_history": action_history,
            "evaluation": evaluation,
            "trajectory_dir": args.trajectory_dir,
        }))

    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        is_benchmark_execution_error = isinstance(e, BenchmarkExecutionError)
        error_type = "benchmark_execution" if is_benchmark_execution_error else "benchmark_error"
        print_result(failure(
            str(e),
            data={
                "steps": 0,
                "turns": 0,
                "done": False,
                "run_mode": args.run_mode,
                "summary": f"Benchmark aborted: {e}" if is_benchmark_execution_error else f"Benchmark error: {e}",
                "fatal_infra": is_benchmark_execution_error,
                "error_type": error_type,
            },
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
