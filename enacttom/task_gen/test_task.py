#!/usr/bin/env python3
"""
Standalone script to test task with LLM agents.

This runs as a subprocess to get a fresh GL context.

Usage:
    python enacttom/task_gen/test_task.py --task-file <path> --config-name <config>
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Test task with LLM agents")
    parser.add_argument("--task-file", required=True, help="Path to task JSON")
    parser.add_argument("--result-file", required=True, help="Path to write result JSON")
    parser.add_argument("--trajectory-dir", default=None, help="Directory to save agent trajectory files")
    parser.add_argument("--config-name", default="examples/enacttom_2_robots")
    parser.add_argument("--max-turns", type=int, default=None, help="Max LLM turns (default: 4x golden trajectory)")
    parser.add_argument("--test-model", type=str, default=None, help="Override model for LLM agents (e.g. gpt-5-mini)")
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
        write_result({"success": False, "error": f"Failed to load task: {e}"})
        sys.exit(1)

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
        write_result({"success": False, "error": f"Import error: {e}"})
        sys.exit(1)

    # Initialize Hydra config
    try:
        GlobalHydra.instance().clear()
        config_dir = str(project_root / "habitat_llm" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        config = compose(config_name=args.config_name)

        # Manually override Hydra interpolations BEFORE fix_config tries to resolve them
        # Use trajectory_dir parent if provided, otherwise fall back to PID-based tmp directory
        if args.trajectory_dir:
            output_dir = str(Path(args.trajectory_dir).parent / f"hydra_test_{os.getpid()}")
        else:
            output_dir = f"/tmp/enacttom_test_{os.getpid()}"
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

        # Override agent models if --test-model is specified
        if args.test_model:
            from enacttom.examples.run_habitat_benchmark import (
                expand_model_name,
                detect_llm_provider,
                apply_agent_llm_configs,
            )
            test_model = expand_model_name(args.test_model)
            test_provider = detect_llm_provider(test_model)
            if not test_provider:
                write_result({"success": False, "error": f"Unknown provider for test model: {args.test_model}"})
                sys.exit(1)
            # Build agent_model_mapping for all agents in config
            agent_model_mapping = {}
            if hasattr(config, "evaluation") and hasattr(config.evaluation, "agents"):
                for agent_id in config.evaluation.agents:
                    agent_model_mapping[agent_id] = {"model": test_model, "llm_provider": test_provider}
            apply_agent_llm_configs(config, agent_model_mapping)
    except Exception as e:
        write_result({"success": False, "error": f"Config error: {e}"})
        sys.exit(1)

    # Convert to GeneratedTask
    try:
        task = GeneratedTask.from_dict(task_data)
    except Exception as e:
        write_result({"success": False, "error": f"Invalid task format: {e}"})
        sys.exit(1)

    # Setup environment
    try:
        # Remove visual sensors - LLM planners don't need them
        remove_visual_sensors(config)

        register_sensors(config)
        register_actions(config)
        register_measures(config)

        dataset = EnactToMDatasetV0(config.habitat.dataset)
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        # Load the specific episode - reset_environment initializes world graph internally
        print(f"Loading episode: {task.episode_id} (scene: {task.scene_id})", file=sys.stderr)
        env_interface.reset_environment(episode_id=task.episode_id)

        # Apply cached spawn positions from task.json (if available)
        apply_agent_spawns(env_interface, task_data.get("agent_spawns", {}))

        runner = BenchmarkRunner(config)

        runner.setup(
            env_interface=env_interface,
            output_dir=output_dir,
            task=task,
            save_video=False,
        )

        # Generate instruction and run
        instruction = task_to_instruction(task)

        # Get max_steps from config (same as run_habitat_benchmark.py)
        max_steps = config.habitat.environment.get("max_episode_steps", 20000)

        # Calculate max_turns: use CLI arg if provided, otherwise 4x golden trajectory.
        if args.max_turns is not None:
            max_turns = args.max_turns
        else:
            golden_trajectory = task_data.get("golden_trajectory", [])
            max_turns = len(golden_trajectory) * 4  # Match benchmark default.

        results = runner.run(instruction=instruction, max_steps=max_steps, max_turns=max_turns)

        # Get planner traces before cleanup
        planner_traces = runner.get_planner_traces()

        runner.cleanup()

        # Include action history for debugging
        action_history = results.get("action_history", [])
        evaluation = results.get("evaluation", {})

        # Save trajectory files if directory specified
        if args.trajectory_dir:
            trajectory_dir = Path(args.trajectory_dir)
            trajectory_dir.mkdir(parents=True, exist_ok=True)

            # Save agent planner traces
            for agent_id, trace in planner_traces.items():
                trace_file = trajectory_dir / f"{agent_id}.txt"
                with open(trace_file, 'w') as f:
                    f.write(trace)

            # Save result.txt with evaluation summary
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

                # Include subtask progress if available
                if evaluation.get('proposition_status'):
                    f.write(f"\n=== SUBTASK PROGRESS ===\n")
                    for prop_id, status in evaluation['proposition_status'].items():
                        status_str = "COMPLETE" if status else "INCOMPLETE"
                        f.write(f"  {prop_id}: {status_str}\n")

                f.write(f"\n=== ACTION HISTORY ===\n")
                for record in action_history:
                    f.write(f"[T{record.get('turn', '?')}] {record.get('agent', '?')}: {record.get('action', '?')}\n")

        write_result({
            "success": True,
            "steps": results.get("steps", 0),
            "turns": results.get("turns", 0),
            "done": results.get("done", False),
            "episode_over": results.get("episode_over", False),
            "summary": f"Task {'completed' if results.get('done') else 'not completed'} in {results.get('turns', 0)} turns ({results.get('steps', 0)} steps)",
            "action_history": action_history,
            "evaluation": evaluation,
            "planner_traces": planner_traces,
        })

    except Exception as e:
        is_benchmark_execution_error = isinstance(e, BenchmarkExecutionError)
        write_result({
            "success": False,
            "steps": 0,
            "turns": 0,
            "done": False,
            "error": str(e),
            "summary": (
                f"Benchmark aborted: {e}"
                if is_benchmark_execution_error
                else f"Benchmark error: {e}"
            ),
            "fatal_infra": is_benchmark_execution_error,
            "error_type": "benchmark_execution" if is_benchmark_execution_error else "benchmark_error",
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
