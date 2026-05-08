#!/usr/bin/env python3
"""
Scene loader for task generation.

Loads a random Habitat episode and extracts the world graph,
providing accurate scene data for task generation.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class SceneData:
    """Complete scene data extracted from world graph."""

    # Episode info
    episode_id: str  # Dataset episode ID (e.g., "1940")
    scene_id: str    # HSSD scene ID (e.g., "102817140")

    # Scene inventory (from world graph)
    rooms: List[str] = field(default_factory=list)
    furniture: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    articulated_furniture: List[str] = field(default_factory=list)

    # Detailed info
    furniture_in_rooms: Dict[str, List[str]] = field(default_factory=dict)
    objects_on_furniture: Dict[str, List[str]] = field(default_factory=dict)

    # Agent spawn positions (calculated once, reused for all runs)
    agent_spawns: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "rooms": self.rooms,
            "furniture": self.furniture,
            "objects": self.objects,
            "articulated_furniture": self.articulated_furniture,
            "furniture_in_rooms": self.furniture_in_rooms,
            "objects_on_furniture": self.objects_on_furniture,
            "agent_spawns": self.agent_spawns,
        }

    # Compatibility: some downstream utilities treat scene_data like a dict.
    # Provide a minimal `.get()` to mirror dict semantics.
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self.to_dict().get(key, default)


    def to_scene_inventory(self) -> Dict[str, List[str]]:
        """Convert to scene_inventory format for compatibility."""
        return {
            "rooms": self.rooms,
            "furniture": self.furniture,
            "objects": self.objects,
            "articulated_furniture": self.articulated_furniture,
        }


def load_random_scene(config: "DictConfig", seed: Optional[int] = None) -> SceneData:
    """
    Load a random Habitat episode and extract scene data.

    Args:
        config: Hydra config (should be already fixed/setup)
        seed: Random seed for episode selection

    Returns:
        SceneData with complete scene information
    """
    return load_scene(config, seed=seed, scene_id=None)


def load_scene(
    config: "DictConfig",
    seed: Optional[int] = None,
    scene_id: Optional[str] = None,
) -> SceneData:
    """
    Load a Habitat episode and extract scene data.

    Args:
        config: Hydra config (should be already fixed/setup)
        seed: Random seed for episode selection (used if scene_id not provided)
        scene_id: Specific scene ID to load (e.g., "102817140"). If None, picks random.

    Returns:
        SceneData with complete scene information
    """
    from habitat_llm.agent.env import register_actions, register_measures, register_sensors
    from habitat_llm.agent.env.dataset import EnactToMDatasetV0
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface

    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Load dataset
    dataset = EnactToMDatasetV0(config.habitat.dataset)

    # Select episode
    if scene_id is not None:
        # Find episode with matching scene_id
        matching_episodes = [ep for ep in dataset.episodes if ep.scene_id == scene_id]
        if not matching_episodes:
            raise ValueError(f"No episode found with scene_id '{scene_id}'. Available scenes: {set(ep.scene_id for ep in dataset.episodes[:20])}...")
        # Pick first matching episode (or random if multiple)
        if seed is not None:
            random.seed(seed)
        selected_episode = random.choice(matching_episodes)
    else:
        # Random episode
        if seed is not None:
            random.seed(seed)
        selected_episode = random.choice(dataset.episodes)

    episode_id = selected_episode.episode_id
    actual_scene_id = selected_episode.scene_id

    # Create environment and load episode
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)
    env_interface.reset_environment(episode_id=episode_id)

    # Get number of agents from config
    num_agents = len(config.habitat.simulator.agents)

    # Extract world graph and agent spawns
    scene_data = extract_scene_data(env_interface, episode_id, actual_scene_id, num_agents)

    # Cleanup
    try:
        env_interface.env.close()
    except Exception:
        pass

    return scene_data


def extract_scene_data(
    env_interface: "EnvironmentInterface",
    episode_id: str,
    scene_id: str,
    num_agents: int = 2,
) -> SceneData:
    """
    Extract complete scene data from environment's world graph.

    Args:
        env_interface: Initialized EnvironmentInterface
        episode_id: The episode ID
        scene_id: The scene ID
        num_agents: Number of agents to extract spawn positions for

    Returns:
        SceneData with all scene information
    """
    # Get world graph (use agent 0's view, which should be complete)
    world_graph = env_interface.world_graph[0]

    # Extract agent spawn positions (these are calculated by the episode)
    agent_spawns = {}
    for agent_uid in range(num_agents):
        try:
            agent = env_interface.sim.agents_mgr[agent_uid].articulated_agent
            position = list(agent.base_pos)
            rotation = float(agent.base_rot)  # yaw angle
            agent_spawns[f"agent_{agent_uid}"] = {
                "position": position,
                "rotation": rotation,
            }
        except (IndexError, AttributeError):
            # Agent doesn't exist or can't get position
            pass

    # Extract rooms
    rooms = []
    for room in world_graph.get_all_rooms():
        room_name = room.name if hasattr(room, 'name') else str(room)
        rooms.append(room_name)

    # Extract furniture and track which room they're in
    furniture = []
    articulated = []
    furniture_in_rooms: Dict[str, List[str]] = {r: [] for r in rooms}

    # Get furniture-to-room mapping from world graph
    furniture_to_room = world_graph.get_furniture_to_room_map()

    for furn in world_graph.get_all_furnitures():
        furn_name = furn.name if hasattr(furn, 'name') else str(furn)
        furniture.append(furn_name)

        # Check if articulated (can open/close)
        if hasattr(furn, 'properties'):
            if furn.properties.get('is_articulated', False) or 'is_open' in furn.properties:
                articulated.append(furn_name)

        # Track room location using world graph mapping
        if furn in furniture_to_room:
            room_node = furniture_to_room[furn]
            room_name = room_node.name if hasattr(room_node, 'name') else str(room_node)
            if room_name in furniture_in_rooms:
                furniture_in_rooms[room_name].append(furn_name)

    # Extract objects and track what furniture they're on
    objects = []
    objects_on_furniture: Dict[str, List[str]] = {f: [] for f in furniture}

    # Get object-to-parent mapping from world graph
    object_furniture_pairs = world_graph.find_object_furniture_pairs()

    for obj in world_graph.get_all_objects():
        obj_name = obj.name if hasattr(obj, 'name') else str(obj)

        # Only include objects that have a known furniture parent in the world graph.
        # Objects without a parent resolve to "unknown" at runtime, which causes
        # agents to fail when navigating (e.g., Navigate[unknown]).
        if obj not in object_furniture_pairs:
            continue

        parent = object_furniture_pairs[obj]
        parent_name = parent.name if hasattr(parent, 'name') else str(parent)
        if parent_name not in objects_on_furniture:
            continue

        objects.append(obj_name)
        objects_on_furniture[parent_name].append(obj_name)

    return SceneData(
        episode_id=episode_id,
        scene_id=scene_id,
        rooms=rooms,
        furniture=furniture,
        objects=objects,
        articulated_furniture=articulated,
        furniture_in_rooms=furniture_in_rooms,
        objects_on_furniture=objects_on_furniture,
        agent_spawns=agent_spawns,
    )


def apply_agent_spawns(
    env_interface: "EnvironmentInterface",
    agent_spawns: Dict[str, Dict[str, Any]],
) -> None:
    """
    Apply cached spawn positions to agents.

    Args:
        env_interface: Initialized EnvironmentInterface
        agent_spawns: Dict mapping agent_X -> {"position": [x,y,z], "rotation": float}
    """
    import sys
    import numpy as np

    if not agent_spawns:
        return

    print(f"Applying cached spawn positions for {len(agent_spawns)} agents", file=sys.stderr)
    for agent_key, spawn in agent_spawns.items():
        if agent_key.startswith("_"):  # Skip comments
            continue
        agent_uid = int(agent_key.split("_")[1])
        try:
            agent = env_interface.sim.agents_mgr[agent_uid].articulated_agent
            # Convert list to numpy array (base_pos expects array, not Python list)
            agent.base_pos = np.array(spawn["position"])
            # Handle rotation as float (yaw) - ignore if list/quaternion
            rot = spawn.get("rotation", 0.0)
            if isinstance(rot, (int, float)):
                agent.base_rot = rot
            print(f"  {agent_key}: pos={spawn['position'][:2]}...", file=sys.stderr)
        except (IndexError, KeyError) as e:
            print(f"  Warning: Could not set spawn for {agent_key}: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Test scene loading
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from habitat_llm.utils import fix_config, setup_config

    GlobalHydra.instance().clear()
    config_dir = str(project_root / "habitat_llm" / "conf")
    initialize_config_dir(config_dir=config_dir, version_base=None)
    config = compose(config_name="examples/enacttom_2_robots")

    fix_config(config)
    config = setup_config(config, seed=47668090)

    scene_data = load_random_scene(config)

    print("\n=== Scene Data ===")
    print(f"Episode: {scene_data.episode_id}")
    print(f"Scene: {scene_data.scene_id}")
    print(f"Rooms: {scene_data.rooms[:5]}...")
    print(f"Furniture: {scene_data.furniture[:10]}...")
    print(f"Objects: {scene_data.objects[:10]}...")
    print(f"Articulated: {scene_data.articulated_furniture[:5]}...")
