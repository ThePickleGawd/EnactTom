"""
Load a Habitat scene for task generation.

Requires GL context — always runs as a subprocess.

Usage:
    # CLI
    python -m enacttom.cli.new_scene 2 --working-dir DIR [--scene-id X] [--seed N]

    # Agent spawns subprocess:
    subprocess.run([sys.executable, "-m", "enacttom.cli.new_scene", ...])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Load a scene (random or specific)")
    parser.add_argument("num_agents", type=int, help="Number of agents (2-4)")
    parser.add_argument("--working-dir", required=True, help="Working directory for output")
    parser.add_argument("--config-name", default=None, help="Hydra config name (auto-detected from num_agents)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for scene selection")
    parser.add_argument("--scene-id", type=str, default=None, help="Specific scene ID to load")
    args = parser.parse_args()

    from enacttom.cli import failure, print_result, success

    if args.num_agents < 2 or args.num_agents > 4:
        print_result(failure(f"num_agents must be 2-4, got {args.num_agents}"))
        sys.exit(1)

    config_name = args.config_name or f"examples/enacttom_{args.num_agents}_robots"

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    from habitat_llm.utils.core import fix_config, setup_config

    from enacttom.task_gen.scene_loader import load_scene

    # Initialize Hydra config
    try:
        GlobalHydra.instance().clear()
        config_dir = str(project_root / "habitat_llm" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        config = compose(config_name=config_name)

        output_dir = f"{args.working_dir}/hydra_scene_{os.getpid()}"
        with open_dict(config):
            if "evaluation" in config:
                config.evaluation.output_dir = output_dir
            if "paths" in config:
                config.paths.results_dir = f"{output_dir}/results"
                config.paths.epi_result_file_path = f"{output_dir}/results/episode_result_log.csv"
                config.paths.run_result_file_path = f"{output_dir}/results/run_result_log.csv"
                config.paths.end_result_file_path = f"{output_dir}/results/end_result_log.csv"

        fix_config(config)
        config = setup_config(config, seed=args.seed or 47668090)
    except Exception as e:
        print_result(failure(f"Config error: {e}"))
        sys.exit(1)

    # Load scene
    try:
        scene_data = load_scene(config, seed=args.seed, scene_id=args.scene_id)

        # Save scene data to working directory
        scene_file = Path(args.working_dir) / "current_scene.json"
        scene_dict = scene_data.to_dict()
        with open(scene_file, "w") as f:
            json.dump(scene_dict, f, indent=2)

        print_result(success({
            "scene_data": scene_dict,
            "episode_id": scene_data.episode_id,
            "scene_id": scene_data.scene_id,
            "scene_file": str(scene_file),
            "rooms": len(scene_data.rooms),
            "furniture": len(scene_data.furniture),
            "objects": len(scene_data.objects),
        }))

    except Exception as e:
        import traceback

        print_result(failure(str(e), data={"traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
