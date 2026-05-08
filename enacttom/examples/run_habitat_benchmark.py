#!/usr/bin/env python3
"""
Run EnactToM benchmark with Habitat integration.

This script runs EnactToM tasks in the Habitat simulator with LLM planners
for multi-agent evaluation.

By default, runs ALL tasks in data/enacttom/tasks/. Use --task to run a single task.

Usage:
    ./enacttom/run_enacttom.sh benchmark                    # Run all tasks
    ./enacttom/run_enacttom.sh benchmark --model sonnet     # Run all tasks with Claude Sonnet
    ./enacttom/run_enacttom.sh benchmark --task task.json   # Run single task
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import EnactToMDatasetV0
from habitat_llm.utils import cprint, setup_config, fix_config

from enacttom.api_costs import summarize_task_costs
from enacttom.runner.benchmark import BenchmarkExecutionError
from enacttom.task_gen import GeneratedTask


MODEL_ALIASES = {
    "kimi-k2.5": "accounts/fireworks/models/kimi-k2p5",
    "kimi-k2-thinking": "moonshot.kimi-k2-thinking",
    "deepseek": "accounts/fireworks/models/deepseek-v3p2",
    "deepseek-v3.2": "accounts/fireworks/models/deepseek-v3p2",
    "gemini-pro": "gemini-3.1-pro-preview",
    "gemini-flash": "gemini-3-flash-preview",
    "ministral-3-8b": "mistral.ministral-3-8b-instruct",
    "ministral-3-14b": "mistral.ministral-3-14b-instruct",
    "mistral-large-3": "mistral.mistral-large-3-675b-instruct",
    "qwen3-next-80b": "qwen.qwen3-next-80b-a3b",
    "qwen3-vl-235b": "qwen.qwen3-vl-235b-a22b",
}

MODEL_PROVIDER_MAP = {
    "gpt-4o": "openai_chat",
    "gpt-4o-mini": "openai_chat",
    "gpt-5": "openai_chat",
    "gpt-5-mini": "openai_chat",
    "gpt-5.1": "openai_chat",
    "gpt-5.2": "openai_chat",
    "gpt-5.4": "openai_chat",
    "gpt-5.4-mini": "openai_chat",
    "o3": "openai_chat",
    "us.anthropic.claude-sonnet-4-6-v1:0": "bedrock_claude",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "bedrock_claude",
    "us.anthropic.claude-opus-4-6-v1:0": "bedrock_claude",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "bedrock_claude",
    "us.anthropic.claude-opus-4-5-20251101-v1:0": "bedrock_claude",
    "claude-sonnet-4-6": "anthropic_claude",
    "claude-opus-4-6": "anthropic_claude",
    "claude-haiku-4-5-20251001": "anthropic_claude",
    "claude-sonnet-4-5-20250929": "anthropic_claude",
    "claude-opus-4-5-20251101": "anthropic_claude",
    "kimi-k2.5": "openai_chat",
    "accounts/fireworks/models/kimi-k2p5": "openai_chat",
    "deepseek": "openai_chat",
    "deepseek-v3.2": "openai_chat",
    "accounts/fireworks/models/deepseek-v3p2": "openai_chat",
    "gemini-pro": "openai_chat",
    "gemini-flash": "openai_chat",
    "gemini-3.1-pro-preview": "openai_chat",
    "gemini-3-flash-preview": "openai_chat",
    "kimi-k2-thinking": "bedrock_kimi",
    "moonshot.kimi-k2-thinking": "bedrock_kimi",
    "ministral-3-8b": "bedrock_mistral",
    "ministral-3-14b": "bedrock_mistral",
    "mistral-large-3": "bedrock_mistral",
    "mistral.ministral-3-8b-instruct": "bedrock_mistral",
    "mistral.ministral-3-14b-instruct": "bedrock_mistral",
    "mistral.mistral-large-3-675b-instruct": "bedrock_mistral",
    "qwen3-next-80b": "bedrock_qwen",
    "qwen3-vl-235b": "bedrock_qwen",
    "qwen.qwen3-next-80b-a3b": "bedrock_qwen",
    "qwen.qwen3-vl-235b-a22b": "bedrock_qwen",
}

CLAUDE_ALIAS_MODELS = {
    "sonnet",
    "sonnet-4.5",
    "sonnet4.5",
    "sonnet-4.6",
    "sonnet4.6",
    "haiku",
    "haiku-4.5",
    "haiku4.5",
    "opus",
    "opus-4.5",
    "opus4.5",
    "opus-4.6",
    "opus4.6",
}

def expand_model_name(model: str) -> str:
    """Expand shorthand model names to full IDs."""
    return MODEL_ALIASES.get(model, model)


def _anthropic_api_key_available() -> bool:
    """Check whether ANTHROPIC_API_KEY is available via env or local .env file."""
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return True

    env_path = project_root / ".env"
    if not env_path.exists():
        return False

    try:
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() != "ANTHROPIC_API_KEY":
                continue
            if value.strip().strip('"').strip("'"):
                return True
    except Exception:
        return False

    return False


def _preferred_claude_provider() -> str:
    """Prefer Anthropic direct API when ANTHROPIC_API_KEY is set."""
    return "anthropic_claude" if _anthropic_api_key_available() else "bedrock_claude"


def detect_llm_provider(model: str) -> Optional[str]:
    """Auto-detect provider from model name."""
    normalized = (model or "").strip()
    if not normalized:
        return None
    normalized_lower = normalized.lower()

    if normalized_lower in CLAUDE_ALIAS_MODELS:
        return _preferred_claude_provider()

    return MODEL_PROVIDER_MAP.get(normalized_lower)


def apply_agent_llm_configs(config: DictConfig, agent_model_mapping: Dict[str, Dict[str, str]]) -> None:
    """Apply per-agent llm provider/model configs into Hydra config."""
    import habitat_llm

    if not hasattr(config, "evaluation") or not hasattr(config.evaluation, "agents"):
        return

    habitat_llm_dir = os.path.dirname(habitat_llm.__file__)
    llm_cfg_cache: Dict[Tuple[str, str], Any] = {}

    with open_dict(config):
        for agent_id in config.evaluation.agents:
            agent_spec = agent_model_mapping.get(agent_id)
            if not agent_spec:
                continue

            agent_conf = config.evaluation.agents[agent_id]
            if not hasattr(agent_conf, "planner") or not hasattr(agent_conf.planner, "plan_config"):
                continue

            llm_provider = agent_spec["llm_provider"]
            model = agent_spec["model"]
            cache_key = (llm_provider, model)

            if cache_key not in llm_cfg_cache:
                llm_config_path = f"{habitat_llm_dir}/conf/llm/{llm_provider}.yaml"
                if not os.path.exists(llm_config_path):
                    raise FileNotFoundError(
                        f"LLM provider config not found for '{llm_provider}' at {llm_config_path}"
                    )
                llm_cfg = OmegaConf.load(llm_config_path)
                if not hasattr(llm_cfg, "generation_params"):
                    llm_cfg.generation_params = {}
                llm_cfg.generation_params.model = model
                llm_cfg_cache[cache_key] = llm_cfg

            # Deep copy so agents do not share mutable OmegaConf state.
            copied_cfg = OmegaConf.create(
                OmegaConf.to_container(llm_cfg_cache[cache_key], resolve=True)
            )
            agent_conf.planner.plan_config.llm = copied_cfg


def ensure_benchmark_observation_config(config: DictConfig) -> None:
    """Populate benchmark defaults and normalize world-state visibility by run mode."""
    with open_dict(config):
        if not hasattr(config, "benchmark_observation_mode"):
            config.benchmark_observation_mode = "text"
        if not hasattr(config, "benchmark_run_mode"):
            config.benchmark_run_mode = "standard"
        run_mode = str(config.benchmark_run_mode).strip().lower()
        if hasattr(config, "world_model"):
            config.world_model.partial_obs = run_mode == "standard"
        config.agent_asymmetry = run_mode == "standard"
        if not hasattr(config, "benchmark_vision") or config.benchmark_vision is None:
            config.benchmark_vision = OmegaConf.create(
                {
                    "selector_prompt_name": "enacttom_frame_selector",
                    "selector_min_frames": 1,
                    "selector_max_frames": 5,
                    "selector_max_candidates": 12,
                    "image_format": "jpeg",
                }
            )


def apply_benchmark_prompt_configs(config: DictConfig) -> None:
    """Apply EnactToM benchmark prompt defaults with a shared acting-agent prompt."""
    import habitat_llm

    habitat_llm_dir = os.path.dirname(habitat_llm.__file__)
    agent_instruct = OmegaConf.load(f"{habitat_llm_dir}/conf/instruct/enacttom_agent.yaml")

    with open_dict(config):
        for agent_conf in config.evaluation.agents.values():
            if not hasattr(agent_conf, "planner") or not hasattr(agent_conf.planner, "plan_config"):
                continue
            agent_conf.planner.plan_config.instruct = OmegaConf.create(
                OmegaConf.to_container(agent_instruct, resolve=True)
            )


def load_tasks_from_file(task_file: str) -> Tuple[List[GeneratedTask], List[Dict]]:
    """Load tasks from a single JSON file.

    Supports two formats:
    - Bundle format: {"tasks": [task1, task2, ...]}
    - Single task format: {task_id, title, ...}

    Returns:
        Tuple of (tasks list, raw data list) - raw data includes golden_trajectory
    """
    with open(task_file) as f:
        data = json.load(f)

    tasks = []
    raw_data = []

    # Check if it's a bundle (has "tasks" array) or single task
    if "tasks" in data:
        # Bundle format
        for task_data in data["tasks"]:
            task = GeneratedTask.from_dict(task_data)
            tasks.append(task)
            raw_data.append(task_data)
    elif "task_id" in data:
        # Single task format
        task = GeneratedTask.from_dict(data)
        tasks.append(task)
        raw_data.append(data)

    return tasks, raw_data


def load_all_tasks(task_dir: Path) -> Tuple[List[GeneratedTask], List[Dict]]:
    """Load all tasks from a directory.

    Returns:
        Tuple of (tasks list, raw data list)
    """
    tasks = []
    raw_data = []

    # Find all JSON files in the directory
    json_files = sorted(task_dir.glob("*.json"))

    for task_file in json_files:
        try:
            file_tasks, file_raw = load_tasks_from_file(str(task_file))
            tasks.extend(file_tasks)
            raw_data.extend(file_raw)
        except Exception as e:
            cprint(f"Warning: Could not load {task_file.name}: {e}", "yellow")

    return tasks, raw_data


def run_single_task(
    config: DictConfig,
    env_interface: EnvironmentInterface,
    task: GeneratedTask,
    task_raw: Dict[str, Any],
    output_dir: str,
    default_model_spec: Dict[str, str],
    task_index: int = 0,
    total_tasks: int = 1,
) -> Dict[str, Any]:
    """Run benchmark on a single task.

    Returns:
        Results dict with success, steps, turns, etc.
    """
    from enacttom.runner import BenchmarkRunner
    from enacttom.runner.benchmark import task_to_instruction

    task_id = task.task_id
    prefix = f"[{task_index + 1}/{total_tasks}]" if total_tasks > 1 else ""

    cprint(f"\n{'=' * 60}", "blue")
    cprint(f"{prefix} TASK: {task.title}", "blue")
    cprint(f"{'=' * 60}", "blue")
    print(f"Task ID: {task_id}")
    print(f"Episode ID: {task.episode_id} (Scene: {task.scene_id})")
    if task.task:
        print(f"\n[Task]: {task.task}\n")
    print(f"Mechanics: {task.active_mechanics}")
    if task.mechanic_bindings:
        print(f"Mechanic bindings: {len(task.mechanic_bindings)} active")
        for b in task.mechanic_bindings:
            print(f"  - {b.mechanic_type}: {b.trigger_object} -> {b.target_object or 'self'}")

    # Validate num_agents matches config
    task_num_agents = task.num_agents
    config_num_agents = len(config.evaluation.agents)
    if task_num_agents != config_num_agents:
        cprint(f"SKIP: Task requires {task_num_agents} agents but config has {config_num_agents}", "yellow")
        return {
            "task_id": task_id,
            "title": task.title,
            "category": task.category,
            "skipped": True,
            "skip_reason": f"Agent count mismatch: task needs {task_num_agents}, config has {config_num_agents}",
            "success": False,
        }

    agent_model_mapping = {
        f"agent_{i}": {
            "model": default_model_spec["model"],
            "llm_provider": default_model_spec["llm_provider"],
        }
        for i in range(task_num_agents)
    }
    apply_agent_llm_configs(config, agent_model_mapping)
    apply_benchmark_prompt_configs(config)

    # Reset environment to the correct episode for this task
    if task.episode_id and task.episode_id != "unknown":
        cprint(f"Resetting environment to episode: {task.episode_id}", "blue")
        try:
            env_interface.reset_environment(episode_id=task.episode_id)
            cprint(f"Successfully loaded episode {task.episode_id}", "green")
        except (ValueError, IndexError) as e:
            cprint(f"SKIP: Could not load episode {task.episode_id}: {e}", "yellow")
            return {
                "task_id": task_id,
                "title": task.title,
                "category": task.category,
                "skipped": True,
                "skip_reason": f"Episode not found: {task.episode_id}",
                "success": False,
                "agent_model_mapping": agent_model_mapping,
            }

    # Create task-specific output directory
    task_output_dir = f"{output_dir}/{task_id}"
    Path(task_output_dir).mkdir(parents=True, exist_ok=True)
    api_usage_log = Path(task_output_dir) / "api_usage.jsonl"
    previous_api_usage_log = os.environ.get("ENACTTOM_API_USAGE_LOG")
    if api_usage_log.exists():
        api_usage_log.unlink()
    os.environ["ENACTTOM_API_USAGE_LOG"] = str(api_usage_log)

    try:
        # Create and setup benchmark runner
        runner = BenchmarkRunner(config)
        runner.setup(
            env_interface=env_interface,
            task=task,
            output_dir=task_output_dir,
        )

        # Build instruction
        run_mode = str(getattr(config, "benchmark_run_mode", "standard")).strip().lower()
        instruction = task_to_instruction(task, run_mode=run_mode)

        print(f"\nPer-agent instructions:")
        for agent_id, instr in instruction.items():
            print(f"\n--- {agent_id} ---")
            print(instr)

        # Print agent info
        cprint(f"\nAgents: {list(runner.agents.keys())}", "blue")
        for uid, agent in runner.agents.items():
            cprint(f"  agent_{uid} tools: {list(agent.tools.keys())}", "blue")

        # Get max steps from config
        max_steps = config.habitat.environment.get("max_episode_steps", 2000)

        # Calculate max turns as 4x golden trajectory length.
        golden_trajectory = task_raw.get("golden_trajectory", [])
        if "max_turns" in config:
            max_turns = config.max_turns
        else:
            max_turns = len(golden_trajectory) * 4

        cprint(f"\nMax simulation steps: {max_steps}", "blue")
        cprint(f"Max LLM turns: {max_turns} (golden trajectory: {len(golden_trajectory)} steps)", "blue")

        # Run benchmark
        results = {
            "task_id": task_id,
            "title": task.title,
            "category": task.category,
            "run_mode": run_mode,
            "skipped": False,
            "success": False,
            "steps": 0,
            "turns": 0,
            "error": None,
            "agent_model_mapping": agent_model_mapping,
        }

        try:
            cprint("Starting task execution with LLM planners...", "blue")
            run_results = runner.run(instruction=instruction, max_steps=max_steps, max_turns=max_turns)

            results["success"] = run_results.get("success", False)
            results["steps"] = run_results.get("steps", 0)
            results["turns"] = run_results.get("turns", 0)
            results["done"] = run_results.get("done", False)
            results["episode_over"] = run_results.get("episode_over", False)
            results["evaluation"] = run_results.get("evaluation", {})

            if results["success"]:
                cprint(f"\n✓ TASK PASSED: {task.title}", "green")
            else:
                cprint(f"\n✗ TASK FAILED: {task.title}", "red")

            print(f"Steps: {results['steps']}, Turns: {results['turns']}")

        except Exception as e:
            error_str = str(e)
            is_timeout = "Episode over" in error_str or "call reset before calling step" in error_str

            if is_timeout:
                cprint(f"\nTask timed out (max simulation steps reached)", "yellow")
                results["error"] = "timeout"
            else:
                cprint(f"Error during task execution: {e}", "red")
                import traceback
                traceback.print_exc()
                raise BenchmarkExecutionError(
                    f"Benchmark aborted for task '{task_id}': {e}"
                ) from e

        task_cost_summary = summarize_task_costs(Path(task_output_dir))
        if task_cost_summary.get("models"):
            results["api_cost_summary"] = task_cost_summary
        return results
    finally:
        if previous_api_usage_log is None:
            os.environ.pop("ENACTTOM_API_USAGE_LOG", None)
        else:
            os.environ["ENACTTOM_API_USAGE_LOG"] = previous_api_usage_log


def _build_category_stats(all_results: list) -> dict:
    """Build per-category aggregate statistics from benchmark results."""
    def _require_percent_complete(result: dict, category: str) -> float:
        """Require normalized progress from PDDL evaluation payload."""
        evaluation = result.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError(
                f"Missing evaluation payload for task '{result.get('task_id', 'unknown')}'"
                f" in category '{category}'"
            )

        progress = evaluation.get("percent_complete")
        if not isinstance(progress, (int, float)):
            raise ValueError(
                f"Missing numeric evaluation.percent_complete for task"
                f" '{result.get('task_id', 'unknown')}' in category '{category}'"
            )
        return float(progress)

    def _literal_tom_stats(results: list) -> dict:
        scored_task_count = 0
        fallback_score_sum = 0.0
        probe_count = 0
        supported_probe_count = 0
        passed_probe_count = 0

        for result in results:
            evaluation = result.get("evaluation", {})
            if not isinstance(evaluation, dict):
                continue

            probe_summary = evaluation.get("literal_tom_probe_summary", {})
            if isinstance(probe_summary, dict):
                probe_count += int(probe_summary.get("probe_count", 0) or 0)
                supported = int(probe_summary.get("supported_probe_count", 0) or 0)
                passed = int(probe_summary.get("passed_count", 0) or 0)
                supported_probe_count += supported
                passed_probe_count += passed
                if supported > 0:
                    scored_task_count += 1
                    continue

            score = evaluation.get("literal_tom_probe_score")
            if isinstance(score, (int, float)):
                scored_task_count += 1
                fallback_score_sum += float(score)

        score_pct = None
        if supported_probe_count > 0:
            score_pct = passed_probe_count / supported_probe_count * 100
        elif scored_task_count > 0:
            score_pct = fallback_score_sum / scored_task_count * 100

        return {
            "literal_tom_score": round(score_pct, 1) if score_pct is not None else None,
            "literal_tom_task_count": scored_task_count,
            "literal_tom_probe_count": probe_count,
            "literal_tom_supported_probe_count": supported_probe_count,
            "literal_tom_passed_probe_count": passed_probe_count,
        }

    # Group results by category (skip skipped tasks)
    by_category = {}
    for r in all_results:
        if r.get("skipped"):
            continue
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, []).append(r)

    stats = {}

    for cat, results in by_category.items():
        evals = [r.get("evaluation", {}) for r in results]
        total = len(results)
        passed = sum(1 for r in results if r.get("success"))
        timed_out = sum(1 for r in results if not r.get("done", True))
        avg_steps = sum(r.get("steps", 0) for r in results) / total
        avg_progress = sum(_require_percent_complete(r, cat) for r in results) / total

        cat_stats = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total * 100,
            "avg_progress": round(avg_progress, 3),
            "avg_steps": round(avg_steps, 1),
            "timed_out": timed_out,
        }
        cat_stats.update(_literal_tom_stats(results))

        if cat == "cooperative":
            # Nothing extra beyond the common stats
            pass

        elif cat == "mixed":
            main_successes = sum(1 for e in evals if e.get("main_goal_success"))
            avg_main_progress = sum(e.get("main_goal_progress", 0) for e in evals) / total

            # Per-agent subgoal stats
            all_agent_results = []
            for e in evals:
                agent_status = e.get("agent_subgoal_status", {})
                all_agent_results.extend(agent_status.values())

            agent_subgoal_rate = (
                sum(1 for v in all_agent_results if v) / len(all_agent_results)
                if all_agent_results else 0
            )

            cat_stats["main_goal_success_rate"] = round(main_successes / total * 100, 1)
            cat_stats["avg_main_goal_progress"] = round(avg_main_progress, 3)
            cat_stats["agent_subgoal_success_rate"] = round(agent_subgoal_rate * 100, 1)

        stats[cat] = cat_stats

    return stats


def _print_category_stats(category_stats: dict) -> None:
    """Print category-specific statistics to console."""
    if not category_stats:
        return

    for cat, stats in category_stats.items():
        cprint(f"\n--- {cat.upper()} ---", "cyan")
        cprint(f"  Tasks: {stats['total']}  |  Passed: {stats['passed']}  |  Pass rate: {stats['pass_rate']:.1f}%", "cyan")
        cprint(f"  Avg progress: {stats['avg_progress']:.1%}  |  Avg steps: {stats['avg_steps']:.0f}  |  Timed out: {stats['timed_out']}", "cyan")
        if stats.get("literal_tom_score") is not None:
            cprint(
                "  Literal ToM: "
                f"{stats['literal_tom_score']:.1f}%"
                f" ({stats.get('literal_tom_passed_probe_count', 0)}/"
                f"{stats.get('literal_tom_supported_probe_count', 0)} supported probes)",
                "cyan",
            )

        if cat == "mixed":
            cprint(f"  Main goal success: {stats['main_goal_success_rate']:.1f}%  |  Avg main progress: {stats['avg_main_goal_progress']:.1%}", "cyan")
            cprint(f"  Agent subgoal success rate: {stats['agent_subgoal_success_rate']:.1f}%", "cyan")


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    fix_config(config)
    config = setup_config(config, seed=47668090)
    ensure_benchmark_observation_config(config)

    # Get default model and provider from config (passed via +model=X +llm_provider=Y)
    model = expand_model_name(config.get("model", "gpt-5.2"))
    llm_provider = config.get("llm_provider", "") or detect_llm_provider(model) or "openai_chat"
    default_model_spec = {"model": model, "llm_provider": llm_provider}

    # Ensure save_video exists in config for runners.
    with open_dict(config):
        if not hasattr(config.evaluation, 'save_video'):
            config.evaluation.save_video = True

    cprint("\n" + "=" * 60, "blue")
    cprint("EnactToM Habitat Benchmark", "blue")
    cprint("=" * 60, "blue")
    cprint(f"LLM: {llm_provider} ({model})", "blue")
    cprint(f"Observation mode: {config.benchmark_observation_mode}", "blue")
    cprint(f"Benchmark mode: {config.benchmark_run_mode}", "blue")
    # Register Habitat components
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset
    dataset = EnactToMDatasetV0(config.habitat.dataset)
    cprint(f"Loaded dataset with {len(dataset.episodes)} episodes", "green")

    # Create environment interface
    cprint("Initializing Habitat environment...", "blue")
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    try:
        env_interface.initialize_perception_and_world_graph()
    except Exception as e:
        cprint(f"Warning: Failed to initialize world graph: {e}", "yellow")

    cprint("Environment initialized!", "green")

    # Determine which tasks to run
    task_file_arg = config.get("task", None)
    num_agents_filter = config.get("num_agents_filter", None)
    task_category_filter = config.get("task_category_filter", None)
    if task_category_filter:
        task_category_filter = str(task_category_filter).strip().lower()
        if task_category_filter not in ("cooperative", "mixed"):
            cprint(
                f"ERROR: Invalid task_category_filter '{task_category_filter}'. "
                "Expected one of cooperative|mixed.",
                "red",
            )
            sys.exit(1)
    task_dir = Path(config.get("task_dir", "data/enacttom/tasks"))

    if task_file_arg:
        # Single task mode: run only the specified task
        task_file = Path(task_file_arg)
        if not task_file.exists():
            cprint(f"ERROR: Task file not found: {task_file}", "red")
            sys.exit(1)
        cprint(f"Single task mode: {task_file}", "blue")
        tasks, raw_data = load_tasks_from_file(str(task_file))
    else:
        # All tasks mode: run all tasks in the directory
        if not task_dir.exists():
            cprint(f"ERROR: Task directory not found: {task_dir}", "red")
            cprint("Run task generation first: ./enacttom/run_enacttom.sh generate", "yellow")
            sys.exit(1)

        tasks, raw_data = load_all_tasks(task_dir)

        if not tasks:
            cprint(f"ERROR: No tasks found in {task_dir}", "red")
            cprint("Run task generation first: ./enacttom/run_enacttom.sh generate", "yellow")
            sys.exit(1)

        # Filter by agent count if specified
        if num_agents_filter:
            filtered_tasks = []
            filtered_raw = []
            for task, raw in zip(tasks, raw_data):
                if task.num_agents == num_agents_filter:
                    filtered_tasks.append(task)
                    filtered_raw.append(raw)
            tasks, raw_data = filtered_tasks, filtered_raw

            if not tasks:
                cprint(f"No tasks found with {num_agents_filter} agents", "yellow")
                return

            cprint(f"Running {len(tasks)} tasks with {num_agents_filter} agents", "blue")
        else:
            cprint(f"All tasks mode: {len(tasks)} tasks found", "blue")

    # Filter by category if specified
    if task_category_filter:
        filtered_tasks = []
        filtered_raw = []
        for task, raw in zip(tasks, raw_data):
            if str(getattr(task, "category", "cooperative")).lower() == task_category_filter:
                filtered_tasks.append(task)
                filtered_raw.append(raw)
        tasks, raw_data = filtered_tasks, filtered_raw

        if not tasks:
            cprint(f"No tasks found with category '{task_category_filter}'", "yellow")
            return

        cprint(f"Running {len(tasks)} task(s) with category '{task_category_filter}'", "blue")

    output_dir = config.paths.results_dir

    # Run all tasks
    all_results = []
    for i, (task, task_raw) in enumerate(zip(tasks, raw_data)):
        try:
            result = run_single_task(
                config=config,
                env_interface=env_interface,
                task=task,
                task_raw=task_raw,
                output_dir=output_dir,
                default_model_spec=default_model_spec,
                task_index=i,
                total_tasks=len(tasks),
            )
        except BenchmarkExecutionError as exc:
            cprint(f"FATAL benchmark error: {exc}", "red")
            try:
                env_interface.env.close()
            except Exception:
                pass
            sys.exit(1)
        all_results.append(result)

    # Close environment after all tasks are done
    try:
        env_interface.env.close()
    except Exception:
        pass

    # Build category-specific statistics
    category_stats = _build_category_stats(all_results)
    literal_tom_stats = {}
    if category_stats:
        total_supported = sum(
            stats.get("literal_tom_supported_probe_count", 0)
            for stats in category_stats.values()
        )
        total_passed_probes = sum(
            stats.get("literal_tom_passed_probe_count", 0)
            for stats in category_stats.values()
        )
        total_probe_count = sum(
            stats.get("literal_tom_probe_count", 0)
            for stats in category_stats.values()
        )
        total_literal_tasks = sum(
            stats.get("literal_tom_task_count", 0)
            for stats in category_stats.values()
        )
        literal_tom_stats = {
            "literal_tom_score": (
                round(total_passed_probes / total_supported * 100, 1)
                if total_supported > 0 else None
            ),
            "literal_tom_task_count": total_literal_tasks,
            "literal_tom_probe_count": total_probe_count,
            "literal_tom_supported_probe_count": total_supported,
            "literal_tom_passed_probe_count": total_passed_probes,
        }
    else:
        literal_tom_stats = {
            "literal_tom_score": None,
            "literal_tom_task_count": 0,
            "literal_tom_probe_count": 0,
            "literal_tom_supported_probe_count": 0,
            "literal_tom_passed_probe_count": 0,
        }

    # Print summary
    cprint("\n" + "=" * 60, "blue")
    cprint("BENCHMARK SUMMARY", "blue")
    cprint("=" * 60, "blue")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("success"))
    failed = sum(1 for r in all_results if not r.get("skipped") and not r.get("success"))
    skipped = sum(1 for r in all_results if r.get("skipped"))

    cprint(f"Total tasks: {total}", "blue")
    cprint(f"  Passed:  {passed}", "green" if passed > 0 else "blue")
    cprint(f"  Failed:  {failed}", "red" if failed > 0 else "blue")
    cprint(f"  Skipped: {skipped}", "yellow" if skipped > 0 else "blue")

    if total > 0:
        pass_rate = passed / (total - skipped) * 100 if (total - skipped) > 0 else 0
        cprint(f"\nPass rate: {pass_rate:.1f}%", "green" if pass_rate > 50 else "red")
        if literal_tom_stats.get("literal_tom_score") is not None:
            cprint(
                "Literal ToM: "
                f"{literal_tom_stats['literal_tom_score']:.1f}%"
                f" ({literal_tom_stats['literal_tom_passed_probe_count']}/"
                f"{literal_tom_stats['literal_tom_supported_probe_count']} supported probes)",
                "blue",
            )

    # Print category-specific stats
    _print_category_stats(category_stats)

    # Print per-task results
    print("\nPer-task results:")
    for r in all_results:
        status = "✓ PASS" if r.get("success") else ("SKIP" if r.get("skipped") else "✗ FAIL")
        color = "green" if r.get("success") else ("yellow" if r.get("skipped") else "red")
        reason = f" ({r.get('skip_reason', r.get('error', ''))})" if not r.get("success") else ""
        cprint(f"  [{status}] {r['task_id']}: {r['title']}{reason}", color)

    # Save summary to file
    summary_file = Path(output_dir) / "benchmark_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "model": model,
            "llm_provider": llm_provider,
            "benchmark_observation_mode": config.benchmark_observation_mode,
            "run_mode": config.benchmark_run_mode,
            "task_category_filter": task_category_filter,
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": pass_rate if total > 0 else 0,
            **literal_tom_stats,
            "category_stats": category_stats,
            "results": all_results,
        }, f, indent=2)

    cprint(f"\nResults saved to: {summary_file}", "blue")
    cprint("Benchmark complete!", "green")


if __name__ == "__main__":
    cprint("\nEnactToM Habitat Benchmark Runner", "blue")
    cprint("This script runs EnactToM tasks in Habitat with LLM planners.\n", "blue")

    if len(sys.argv) < 2:
        cprint("Usage: python run_habitat_benchmark.py --config-name <config>", "yellow")
        cprint("Example: python run_habitat_benchmark.py --config-name examples/enacttom_2_robots", "yellow")
        sys.exit(1)

    main()
