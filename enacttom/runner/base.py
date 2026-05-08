"""
Base runner class for EnactToM benchmark modes.

This provides common setup for environment, GameStateManager, agents, tools,
video recording, and logging across benchmark and verification modes.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from enacttom.tracing import EventLog

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.agent import Agent


class EnactToMBaseRunner(ABC):
    """
    Base class for EnactToM runners.

    Handles common setup:
    - Environment initialization (sensors, actions, measures)
    - GameStateManager with mechanics
    - Agent creation with tool injection
    - Video recording (DebugVideoUtil)
    - Output directory structure
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the base runner.

        Args:
            config: Hydra configuration (should be already fixed/setup)
        """
        self.config = config

        # Will be set during setup
        self.env_interface: Optional["EnvironmentInterface"] = None
        self.game_manager = None
        self.world_adapter = None
        self.agents: Dict[int, "Agent"] = {}
        self.output_dir: str = ""

        # Video recording
        self._dvu = None
        self._per_agent_recorder = None
        self._fpv_recorder = None
        self._save_video_override: Optional[bool] = None

        # State tracking
        self._action_history: List[Dict[str, Any]] = []
        self._episode_done = False

        # Causal event log (ARE-style tracing)
        self.event_log = EventLog()

    def _is_vision_mode(self) -> bool:
        return str(getattr(self.config, "benchmark_observation_mode", "text")).lower() == "vision"

    def _append_textual_visual_summary(
        self,
        base_text: str,
        uid: int,
        snapshots: List[str],
        agents_passed: Optional[Dict[str, tuple]] = None,
    ) -> str:
        if self._is_vision_mode():
            return base_text
        summary = self._build_textual_visual_summary(uid, snapshots, agents_passed)
        if not summary:
            return base_text
        return f"{base_text}\n\n{summary}" if base_text else summary

    def _build_textual_visual_summary(
        self,
        uid: int,
        snapshots: List[str],
        agents_passed: Optional[Dict[str, tuple]] = None,
    ) -> str:
        """Build one compact per-turn visual summary for text-mode runs."""
        if self._is_vision_mode():
            return ""

        traversed_rooms: List[str] = []
        seen_objects: List[str] = []
        seen_agents: List[str] = []

        for snapshot in snapshots:
            match = re.match(r"\[Step \d+\]\s+([^:]+):\s+(.*)", snapshot.strip())
            if not match:
                continue

            room_name = match.group(1).strip()
            remainder = match.group(2).strip()
            if room_name and room_name not in traversed_rooms:
                traversed_rooms.append(room_name)

            parts = [part.strip() for part in remainder.split(".") if part.strip()]
            if parts:
                object_part = parts[0]
                if object_part and object_part.lower() != "no objects nearby":
                    for obj in [item.strip() for item in object_part.split(",")]:
                        if obj and obj not in seen_objects:
                            seen_objects.append(obj)

                for part in parts[1:]:
                    if re.match(r"agent_\d+\s+(nearby|in\s+\S+)", part):
                        if part not in seen_agents:
                            seen_agents.append(part)

        if agents_passed:
            for agent_name, (room_name, _) in agents_passed.items():
                descriptor = f"{agent_name} in {room_name}"
                if descriptor not in seen_agents:
                    seen_agents.append(descriptor)

        ended_in = None
        try:
            ended_in = self.env_interface.get_agent_room(uid)
        except Exception:
            ended_in = None

        lines: List[str] = []
        if traversed_rooms:
            lines.append(f"- Traversed: {' -> '.join(traversed_rooms)}")
        if seen_objects:
            lines.append(f"- Saw: {', '.join(seen_objects[:5])}")
        if seen_agents:
            lines.append(f"- Saw other agents: {', '.join(seen_agents[:3])}")
        if ended_in:
            lines.append(f"- Ended in: {ended_in}")
        elif not lines:
            lines.append("- No notable new visual observations")

        agent_id = f"agent_{uid}"
        return f"{agent_id}_VisualSummary:\n" + "\n".join(lines)

    @staticmethod
    def _inject_summary_into_planner_context(
        planner: Any,
        summary: str,
    ) -> None:
        """Append a late runner-generated summary into the planner prompt history."""
        if not summary or planner is None or not hasattr(planner, "curr_prompt"):
            return

        llm_cfg = getattr(getattr(planner, "planner_config", None), "llm", None)
        user_tag = getattr(llm_cfg, "user_tag", "")
        assistant_tag = getattr(llm_cfg, "assistant_tag", "")
        eot_tag = getattr(llm_cfg, "eot_tag", "")
        planning_mode = str(getattr(getattr(planner, "planner_config", None), "planning_mode", "")).lower()

        prompt_thought_suffix = f"{assistant_tag}Thought:"
        trace_thought_suffix = "Thought:"

        if planning_mode == "cot":
            if planner.curr_prompt.endswith(prompt_thought_suffix):
                planner.curr_prompt = planner.curr_prompt[: -len(prompt_thought_suffix)]
            if hasattr(planner, "trace") and planner.trace.endswith(trace_thought_suffix):
                planner.trace = planner.trace[: -len(trace_thought_suffix)]

        planner.curr_prompt += f"{user_tag}{summary}\n{eot_tag}"
        if hasattr(planner, "trace"):
            planner.trace += f"{summary}\n"

        if planning_mode == "cot":
            planner.curr_prompt += prompt_thought_suffix
            if hasattr(planner, "trace"):
                planner.trace += trace_thought_suffix

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        agent_actions: Optional[Dict[str, List[str]]] = None,
        save_video: Optional[bool] = None,
        message_targets: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Full setup sequence. Call before run().

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Optional task data with mechanics/bindings
            output_dir: Output directory for videos/logs
            agent_actions: Optional dict mapping agent_id to list of allowed actions.
                           If None, all actions are available to all agents.
                           Example: {"agent_0": ["Navigate", "Open", "Communicate"], "agent_1": ["Navigate"]}
            save_video: Whether to save video. If None, uses config.evaluation.save_video
            message_targets: Optional dict mapping agent_id to list of allowed recipient agent_ids.
                             If None, all agents can message anyone.
                             Example: {"agent_0": ["agent_1"]} — agent_0 can only message agent_1
        """
        self.env_interface = env_interface
        self.output_dir = output_dir or getattr(
            self.config.paths, 'results_dir', 'outputs/enacttom'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Store agent actions and message targets for tool setup
        self._agent_actions = agent_actions
        self._message_targets = message_targets
        self._save_video_override = save_video  # Override for video saving

        self._setup_game_manager(task_data)
        self._setup_agents()
        self._setup_tools()
        self._setup_video()
        self._setup_logging_dirs()

    def _setup_game_manager(self, task_data: Optional[Dict[str, Any]] = None) -> None:
        """Create GameStateManager and load task-declared mechanics."""
        from enacttom import GameStateManager, list_mechanics
        from enacttom.runner.world_adapter import HabitatWorldAdapter

        self.game_manager = GameStateManager(self.env_interface)
        self.world_adapter = HabitatWorldAdapter(self.env_interface, agent_uid=0)

        # If task_data is provided, respect its declared mechanics.
        if task_data is not None:
            # Task mode: use only the mechanics defined in the task
            self.game_manager.initialize_from_task(task_data)
            bindings = task_data.get("mechanic_bindings", task_data.get("mechanics", []))
            active = task_data.get("active_mechanics", [])
            if bindings or active:
                print(f"[EnactToMBaseRunner] Loaded mechanics: {active} ({len(bindings)} bindings)")
            else:
                print("[EnactToMBaseRunner] Task has no mechanics defined")
        else:
            # Backward-compatible fallback for direct runner use without a task.
            all_mechanics = list_mechanics()
            self.game_manager.initialize_from_task({
                "mechanics": [{"mechanic_type": m} for m in all_mechanics]
            })
            print(f"[EnactToMBaseRunner] Enabled all mechanics: {all_mechanics}")

        # Get entities from environment and set on game state
        entities = self.world_adapter.get_interactable_entities()
        state = self.game_manager.get_state()
        state.entities = entities
        self.game_manager.set_state(state)

        # Auto-bind only when a caller constructed the runner without task data.
        if task_data is None:
            state, bindings = self.game_manager.auto_bind_mechanics()
            if bindings:
                self._print_bindings(bindings)

    def _print_bindings(self, bindings: dict) -> None:
        """Pretty print auto-bound mechanics."""
        print("\n[EnactToMBaseRunner] Auto-bound mechanics:")

        # Mechanics
        for mech in ["inverse_state", "remote_control", "state_mirroring"]:
            if mech in bindings:
                info = bindings[mech]
                if mech == "inverse_state":
                    print(f"  • {mech}: {info.get('target')}")
                elif mech == "remote_control":
                    print(f"  • {mech}: {info.get('trigger')} → {info.get('target')}")
                elif mech == "state_mirroring":
                    pair = info.get('pair', [])
                    print(f"  • {mech}: {pair[0] if pair else '?'} ↔ {pair[1] if len(pair) > 1 else '?'}")

        print()

    def _setup_agents(self) -> None:
        """Create Agent instances from config."""
        from habitat_llm.agent import Agent

        # Agent configs are at config.agents.agent_X.config
        # evaluation.agents only has uid references
        if not hasattr(self.config, 'agents'):
            print("[EnactToMBaseRunner] Warning: No agents in config")
            return

        for agent_name, agent_entry in self.config.agents.items():
            if not hasattr(agent_entry, 'config'):
                continue

            # Get uid from agent name (e.g., "agent_0" -> 0) or from config
            uid = getattr(agent_entry, 'uid', None)
            if uid is None:
                uid = int(agent_name.split("_")[-1]) if "_" in agent_name else 0

            agent = Agent(
                uid=uid,
                agent_conf=agent_entry.config,
                env_interface=self.env_interface,
            )
            self.agents[uid] = agent
            print(f"[EnactToMBaseRunner] Created {agent_name} (uid={uid}) with tools: {list(agent.tools.keys())}")

    def _setup_tools(self) -> None:
        """Inject EnactToM tools and Communicate into agents based on allowed actions."""
        from enacttom.actions import get_enacttom_tools
        from enacttom.actions.tool_wrapper import wrap_habitat_tools
        from habitat_llm.tools.perception.communication_tool import CommunicationTool

        for uid, agent in self.agents.items():
            agent_id = f"agent_{uid}"

            # Get allowed actions for this agent (None means all allowed)
            allowed_actions = None
            if self._agent_actions is not None:
                allowed_actions = self._agent_actions.get(agent_id, [])

            def is_allowed(action_name: str) -> bool:
                """Check if action is allowed for this agent."""
                if allowed_actions is None:
                    return True  # No restrictions
                return action_name in allowed_actions

            # Add any registered EnactToM custom tools.
            enacttom_tools = get_enacttom_tools(agent_uid=uid)
            for tool_name, tool in enacttom_tools.items():
                if is_allowed(tool_name):
                    tool.set_environment(self.env_interface)
                    tool.set_game_manager(self.game_manager)
                    agent.tools[tool_name] = tool

            # Add Communicate tool if allowed
            if is_allowed("Communicate"):
                # Build description and allowed_targets based on message_targets
                agent_allowed_targets = None
                if self._message_targets and agent_id in self._message_targets:
                    target_agent_ids = self._message_targets[agent_id]
                    # Convert agent_id strings ("agent_1") to UIDs (1)
                    agent_allowed_targets = [int(aid.split("_")[-1]) for aid in target_agent_ids]
                    target_names = ", ".join(target_agent_ids)
                    comm_desc = (
                        f'Send a message to specific agents. You can ONLY message: {target_names}. '
                        f'Usage: Communicate["your message", {target_agent_ids[0]}] for a DM'
                        + (f' or Communicate["your message", all] to message all allowed recipients' if len(target_agent_ids) > 1 else '')
                        + '. The message MUST be in double quotes. Keep messages on a single line.'
                    )
                else:
                    comm_desc = 'Send a message to specific agents. Usage: Communicate["your message", agent_0] for a DM or Communicate["your message", all] to broadcast. The message MUST be in double quotes. Keep messages on a single line.'

                comm_config = OmegaConf.create({
                    "name": "Communicate",
                    "description": comm_desc,
                })
                comm_tool = CommunicationTool(comm_config)
                comm_tool.agent_uid = uid
                comm_tool.allowed_targets = agent_allowed_targets
                comm_tool.set_environment(self.env_interface)
                agent.tools["Communicate"] = comm_tool

            # Wrap Habitat tools with EnactToM condition checks (locks, etc.)
            # This also filters based on allowed_actions
            wrap_habitat_tools(agent, self.game_manager, allowed_actions)

            tools_added = list(agent.tools.keys())
            print(f"[EnactToMBaseRunner] Agent_{uid} tools: {tools_added}")

    def _setup_video(self) -> None:
        """Initialize video recording utilities."""
        # Check override first, then config, default to True
        if self._save_video_override is not None:
            save_video = self._save_video_override
        elif hasattr(self.config, 'evaluation') and hasattr(self.config.evaluation, 'save_video'):
            save_video = self.config.evaluation.save_video
        else:
            save_video = True

        if not save_video:
            return

        try:
            from habitat_llm.examples.example_utils import DebugVideoUtil, PerAgentThirdPersonRecorder

            video_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

            self._dvu = DebugVideoUtil(
                self.env_interface,
                video_dir,
                unique_postfix=True,
            )

            # Per-agent third-person videos (separate file per agent)
            agent_ids = [f"agent_{uid}" for uid in sorted(self.agents.keys())]
            self._per_agent_recorder = PerAgentThirdPersonRecorder(
                output_dir=self.output_dir,
                agent_ids=agent_ids,
            )

            print(f"[EnactToMBaseRunner] Video recording enabled: {video_dir}")
        except Exception as e:
            print(f"[EnactToMBaseRunner] Video setup failed: {e}")

    def _setup_logging_dirs(self) -> None:
        """Create output directory structure."""
        for subdir in ["prompts", "traces", "planner-log"]:
            path = os.path.join(self.output_dir, subdir)
            os.makedirs(path, exist_ok=True)

    def record_frame(
        self,
        observations: Dict[str, Any],
        actions: Optional[Dict] = None,
        turn: Optional[int] = None,
    ) -> None:
        """Record a video frame."""
        if self._dvu:
            try:
                self._dvu._store_for_video(observations, actions or {}, popup_images={}, turn=turn)
            except Exception:
                pass

        if self._per_agent_recorder:
            actions_dict = actions or {}
            for agent_id in self._per_agent_recorder.agent_ids:
                uid = int(agent_id.split("_")[-1])
                action_tuple = actions_dict.get(uid)
                try:
                    self._per_agent_recorder.record_frame(
                        agent_id, observations, action=action_tuple,
                    )
                except Exception:
                    pass

    def _sync_remote_effects_to_simulator(self, effects: List[str]) -> None:
        """
        Sync mechanic effects to the Habitat simulator.

        When mechanics like remote_control or state_mirroring trigger, they
        update game state but we also need to actually open/close objects
        in the simulator.

        Handles effects like:
        - remote_effect=cabinet_26.is_open=True
        - mirrored=drawer_2.is_open=True

        Args:
            effects: List of effect strings from mechanic result
        """
        from habitat.sims.habitat_simulator.sim_utilities import (
            get_ao_default_link,
            open_link,
            close_link,
        )

        sim = self.env_interface.sim
        aom = sim.get_articulated_object_manager()

        for effect in effects:
            # Handle both remote_effect= and mirrored= prefixes
            if effect.startswith("remote_effect="):
                rest = effect[len("remote_effect="):]
            elif effect.startswith("mirrored="):
                rest = effect[len("mirrored="):]
            else:
                continue

            # Parse "cabinet_26.is_open=True"
            try:
                obj_id, prop_value = rest.rsplit(".", 1)
                prop, value_str = prop_value.split("=")
                value = value_str.lower() == "true"
            except ValueError:
                continue

            if prop != "is_open":
                continue

            # Resolve object handle from world graph
            try:
                world_graph = getattr(self.env_interface, "full_world_graph", None)
                if world_graph is None:
                    for uid in self.env_interface.world_graph:
                        world_graph = self.env_interface.world_graph[uid]
                        break
                node = world_graph.get_node_from_name(obj_id)
                handle = node.sim_handle
            except (ValueError, AttributeError):
                handle = obj_id

            # Get the articulated object
            ao = aom.get_object_by_handle(handle)
            if ao is None:
                continue

            # Get default link and open/close it
            default_link = get_ao_default_link(ao, compute_if_not_found=True)
            if default_link is None:
                continue

            if value:
                open_link(ao, default_link)
            else:
                close_link(ao, default_link)

    def execute_action(
        self,
        uid: int,
        action_name: str,
        target: str,
    ) -> Dict[str, Any]:
        """
        Execute an action via GameStateManager and agent tools.

        Routes all actions through GameStateManager first to apply mechanics,
        then executes in Habitat via agent tools.

        Args:
            uid: Agent UID
            action_name: Name of action (Navigate, Open, Pick, etc.)
            target: Target entity name

        Returns:
            Dict with success, observation, and optional surprise info

        Order of operations:
            1. Check mechanics for blocking/transformation (doesn't modify state)
            2. If blocked by mechanic, return immediately
            3. Execute in Habitat (physical action)
            4. If Habitat fails (too far, occluded), return failure WITHOUT applying state
            5. If Habitat succeeds, apply mechanics to game state
        """
        import torch
        from enacttom.mechanics.handlers import apply_mechanics

        agent_id = f"agent_{uid}"
        target = target or ""

        # 1. Check mechanics (doesn't modify state yet)
        mech_result = apply_mechanics(
            action_name, agent_id, target, self.game_manager.get_state()
        )

        # 2. If mechanic blocked the action, return early WITHOUT executing in Habitat
        if mech_result.blocked:
            # Log blocked action
            self.event_log.log_action(
                step=self.get_sim_steps(),
                agent_id=agent_id,
                action=action_name,
                target=target,
                result=mech_result.observation,
                success=False,
            )
            return {
                "success": False,
                "observation": mech_result.observation,
                "surprise": mech_result.surprise_trigger,
                "blocked": True,
            }

        # Get actual action to execute (may be transformed by mechanic)
        actual_action = mech_result.actual_action or action_name
        actual_target = mech_result.actual_target or target
        mechanic_observation = mech_result.observation if mech_result.applies else None

        # 3. Execute via agent tools in Habitat
        if uid not in self.agents:
            return {
                "success": False,
                "observation": f"No agent with uid {uid}",
            }

        agent = self.agents[uid]

        if actual_action not in agent.tools:
            return {
                "success": False,
                "observation": f"Tool '{actual_action}' not available",
            }

        obs = self.env_interface.get_observations()
        low_level_action, response = agent.process_high_level_action(
            actual_action, actual_target, obs
        )

        if low_level_action is None:
            obs_text = response or f"Executed {actual_action}[{actual_target}]"
        else:
            # Execute motor skill
            tool = agent.tools[actual_action]
            skill_steps = 0
            max_skill_steps = 1500  # ~50 seconds at 30Hz, matches benchmark runner
            action_completed = False
            action_failed = False
            obs_text = ""
            surroundings_snapshots: List[str] = []
            agents_passed: Dict[str, tuple] = {}  # agent_name -> (room, first_step)

            while skill_steps < max_skill_steps:
                try:
                    raw_obs, reward, done, info = self.env_interface.step({uid: low_level_action})
                except AssertionError as e:
                    if "Episode over" in str(e):
                        self._episode_done = True
                        break
                    raise

                parsed_obs = self.env_interface.parse_observations(raw_obs)
                self.record_frame(parsed_obs, {uid: (actual_action, actual_target)})
                skill_steps += 1

                # Capture surroundings every 30 frames
                if skill_steps % 30 == 0:
                    snapshot = self._get_surroundings_description(uid, skill_steps)
                    surroundings_snapshots.append(snapshot)
                    surroundings_snapshots = surroundings_snapshots[-3:]
                    # Track agents encountered in same room (first sighting only)
                    for agent_name, room in self._get_nearby_agents(uid):
                        if agent_name not in agents_passed:
                            agents_passed[agent_name] = (room, skill_steps)

                if done:
                    self._episode_done = True
                    break

                # Check if skill is done
                if hasattr(tool, 'skill') and hasattr(tool.skill, '_is_skill_done'):
                    is_done = tool.skill._is_skill_done(
                        raw_obs, None, None, torch.ones(1, 1), 0
                    )
                    if is_done:
                        action_completed = True
                        if not obs_text:
                            obs_text = "Successful execution!"
                        break

                low_level_action, response = agent.process_high_level_action(
                    actual_action, actual_target, raw_obs
                )
                response_text = (response or "").strip()
                if response_text:
                    obs_text = response_text
                    if self._is_action_failure_text(response_text):
                        action_failed = True
                    elif self._is_action_success_text(response_text):
                        action_completed = True
                    else:
                        action_failed = True
                    break

                if low_level_action is None:
                    action_failed = True
                    obs_text = "Action ended without explicit success signal."
                    break

            if not action_completed and not action_failed:
                if self._episode_done:
                    action_failed = True
                    obs_text = "Action interrupted because the episode ended."
                elif skill_steps >= max_skill_steps:
                    action_failed = True
                    obs_text = f"Action timed out after {max_skill_steps} simulator steps."

            # Append surroundings observations collected during motor skill
            obs_text = self._append_textual_visual_summary(
                obs_text,
                uid,
                surroundings_snapshots,
                agents_passed,
            )

            if action_failed:
                # Log failed action
                self.event_log.log_action(
                    step=self.get_sim_steps(),
                    agent_id=agent_id,
                    action=action_name,
                    target=target,
                    result=obs_text,
                    success=False,
                )
                return {
                    "success": False,
                    "observation": obs_text,
                }

            if not obs_text:
                obs_text = "Successful execution!"

        # 4. Check if Habitat action failed (e.g., "too far", "occluded")
        habitat_failed = self._is_action_failure_text(obs_text)

        if habitat_failed:
            # Log failed action
            self.event_log.log_action(
                step=self.get_sim_steps(),
                agent_id=agent_id,
                action=action_name,
                target=target,
                result=obs_text,
                success=False,
            )
            # Habitat action failed - don't apply mechanics, return failure
            return {
                "success": False,
                "observation": obs_text,
            }

        postcondition_error = self._check_action_postcondition(uid, actual_action, actual_target)
        if postcondition_error:
            obs_text = f"{obs_text} Postcondition check failed: {postcondition_error}".strip()
            self.event_log.log_action(
                step=self.get_sim_steps(),
                agent_id=agent_id,
                action=action_name,
                target=target,
                result=obs_text,
                success=False,
            )
            return {
                "success": False,
                "observation": obs_text,
            }

        # Log the successful action
        action_event = self.event_log.log_action(
            step=self.get_sim_steps(),
            agent_id=agent_id,
            action=action_name,
            target=target,
            result=obs_text,
            success=True,
        )

        if mech_result.applies:
            state, result = self.game_manager.apply_action(action_name, agent_id, target)
            # Sync any remote effects to the actual simulator
            self._sync_remote_effects_to_simulator(result.effects)
            # Log mechanic effect
            self.event_log.log_mechanic(
                step=self.get_sim_steps(),
                mechanic=mech_result.mechanic_type or "unknown",
                trigger=target,
                effect=mechanic_observation or result.observation,
                caused_by=action_event.event_id,
            )
            # Append mechanic observation to Habitat observation
            if mechanic_observation:
                obs_text = f"{obs_text} {mechanic_observation}"
            surprise_trigger = mech_result.surprise_trigger or result.surprise_trigger
        else:
            surprise_trigger = None

        return {
            "success": True,
            "observation": obs_text,
            "surprise": surprise_trigger,
        }

    @staticmethod
    def _is_action_failure_text(text: str) -> bool:
        """Heuristic detection of action failures from skill/tool responses."""
        lower = (text or "").lower()
        return any(
            fail_phrase in lower
            for fail_phrase in ["too far", "occluded", "failed to", "unexpected failure", "cannot"]
        )

    @staticmethod
    def _is_action_success_text(text: str) -> bool:
        """Detect explicit success signals from skill/tool responses."""
        lower = (text or "").lower()
        return (
            "successful execution" in lower
            or "was a success" in lower
            or "inside you find" in lower
        )

    @staticmethod
    def _parse_place_target(target: Optional[str]) -> Optional[Dict[str, str]]:
        """Parse Place target string into object/relation/receptacle parts."""
        if not target:
            return None
        parts = [p.strip() for p in str(target).split(",")]
        if len(parts) < 3:
            return None
        return {
            "object": parts[0],
            "relation": parts[1].lower(),
            "receptacle": parts[2],
        }

    def _build_action_postcondition(
        self,
        uid: int,
        action_name: str,
        target: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Build a predicate that should hold after a successful action."""
        if action_name == "Open" and target:
            return {"entity": target, "property": "is_open"}
        if action_name == "Close" and target:
            return {"entity": target, "property": "is_closed"}
        if action_name == "Pick" and target:
            return {"entity": target, "property": "is_held_by", "target": f"agent_{uid}"}
        if action_name == "Place":
            parsed = self._parse_place_target(target)
            if not parsed:
                return None
            relation = parsed["relation"]
            if relation in ("on", "on_top"):
                prop = "is_on_top"
            elif relation in ("within", "inside", "in"):
                prop = "is_inside"
            else:
                return None
            return {
                "entity": parsed["object"],
                "property": prop,
                "target": parsed["receptacle"],
            }
        return None

    def _check_action_postcondition(
        self,
        uid: int,
        action_name: str,
        target: Optional[str],
    ) -> Optional[str]:
        """
        Verify action postcondition to avoid reporting false action successes.

        Returns an error string when postcondition is not satisfied.
        """
        proposition = self._build_action_postcondition(uid, action_name, target)
        if proposition is None:
            return None

        result = self.evaluate_task(
            {
                "description": f"postcondition::{action_name}",
                "required_states": [proposition],
            }
        )
        if result.get("success", False):
            return None

        failures = result.get("failure_explanations", [])
        if failures:
            return failures[0]
        return "Action postcondition did not hold after execution."

    def execute_parsed_action(self, uid: int, action_str: str) -> Dict[str, Any]:
        """
        Execute an action from a string like "Navigate[kitchen_1]".

        Args:
            uid: Agent UID
            action_str: Action string in format "ActionName[target]"

        Returns:
            Result dict from execute_action
        """
        # Allow empty brackets for actions like Wait[]
        match = re.match(r'(\w+)\[([^\]]*)\]', action_str)
        if not match:
            return {
                "success": False,
                "observation": f"Invalid action format: {action_str}. Use ActionName[target]",
            }

        action_name, target = match.groups()
        # Handle empty target for actions like Wait[]
        target = target if target else None
        return self.execute_action(uid, action_name, target)

    def execute_actions_concurrent(
        self,
        actions: Dict[int, tuple],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Execute multiple agents' actions concurrently.

        All agents execute their skills together, stepping the environment
        with combined low-level actions until all skills complete.

        Args:
            actions: Dict mapping agent uid to (action_name, target) tuple

        Returns:
            Dict mapping agent uid to result dict
        """
        import torch
        from enacttom.mechanics.handlers import apply_mechanics

        results: Dict[int, Dict[str, Any]] = {}
        agent_states: Dict[int, Dict[str, Any]] = {}

        # Phase 1: Setup all agents' actions (check mechanics, initialize skills)
        for uid, (action_name, target) in actions.items():
            agent_id = f"agent_{uid}"
            target = target or ""

            # Check mechanics
            mech_result = apply_mechanics(
                action_name, agent_id, target, self.game_manager.get_state()
            )

            if mech_result.blocked:
                results[uid] = {
                    "success": False,
                    "observation": mech_result.observation,
                    "blocked": True,
                }
                continue

            if uid not in self.agents:
                results[uid] = {"success": False, "observation": f"No agent with uid {uid}"}
                continue

            agent = self.agents[uid]
            actual_action = mech_result.actual_action or action_name
            actual_target = mech_result.actual_target or target

            if actual_action not in agent.tools:
                results[uid] = {"success": False, "observation": f"Tool '{actual_action}' not available"}
                continue

            obs = self.env_interface.get_observations()
            low_level_action, response = agent.process_high_level_action(
                actual_action, actual_target, obs
            )

            if low_level_action is None:
                # Instant action (no motor skill needed)
                obs_text = response or f"Executed {actual_action}[{actual_target}]"
                if self._is_action_failure_text(obs_text):
                    results[uid] = {"success": False, "observation": obs_text}
                else:
                    if mech_result.applies:
                        _, mechanic_result = self.game_manager.apply_action(
                            action_name, agent_id, target
                        )
                        self._sync_remote_effects_to_simulator(mechanic_result.effects)
                        if mech_result.observation:
                            obs_text = f"{obs_text} {mech_result.observation}"
                    results[uid] = {
                        "success": True,
                        "observation": obs_text,
                        "skill_steps": 0,
                    }
                continue

            # Skill-based action - track explicit terminal status
            state = {
                "agent": agent,
                "tool": agent.tools[actual_action],
                "action_name": actual_action,
                "target": actual_target,
                "low_level_action": low_level_action,
                "mech_result": mech_result,
                "done": False,
                "completed": False,
                "failed": False,
                "response": "",
                "skill_steps": 0,
            }

            initial_response = (response or "").strip()
            if initial_response:
                state["done"] = True
                state["response"] = initial_response
                if self._is_action_failure_text(initial_response):
                    state["failed"] = True
                elif self._is_action_success_text(initial_response):
                    state["completed"] = True
                else:
                    state["failed"] = True
                state["low_level_action"] = None

            agent_states[uid] = state

        # Phase 2: Execute all skills concurrently
        max_skill_steps = 1500  # ~50 seconds at 30Hz, matches benchmark runner
        total_steps = 0
        surroundings: Dict[int, List[str]] = {}
        agents_passed: Dict[int, Dict[str, tuple]] = {}  # uid -> {agent_name -> (room, step)}

        while agent_states and total_steps < max_skill_steps:
            if all(state["done"] for state in agent_states.values()):
                break

            # Collect low-level actions from all active agents
            combined_actions: Dict[int, Any] = {}
            for uid, state in agent_states.items():
                if not state["done"] and state["low_level_action"] is not None:
                    combined_actions[uid] = state["low_level_action"]

            if not combined_actions:
                break

            # Step environment with ALL agents at once
            try:
                raw_obs, reward, done, info = self.env_interface.step(combined_actions)
            except AssertionError as e:
                if "Episode over" in str(e):
                    self._episode_done = True
                    break
                raise

            parsed_obs = self.env_interface.parse_observations(raw_obs)
            action_tuples = {
                uid: (state["action_name"], state["target"])
                for uid, state in agent_states.items()
            }
            self.record_frame(parsed_obs, action_tuples)
            total_steps += 1

            if done:
                self._episode_done = True
                break

            # Get next low-level actions for each active agent
            for uid, state in agent_states.items():
                if state["done"]:
                    continue

                state["skill_steps"] += 1

                # Capture surroundings every 30 frames
                if state["skill_steps"] % 30 == 0:
                    snapshot = self._get_surroundings_description(uid, state["skill_steps"])
                    surroundings.setdefault(uid, []).append(snapshot)
                    surroundings[uid] = surroundings[uid][-3:]
                    # Track agents encountered in same room (first sighting only)
                    ap = agents_passed.setdefault(uid, {})
                    for agent_name, room in self._get_nearby_agents(uid):
                        if agent_name not in ap:
                            ap[agent_name] = (room, state["skill_steps"])

                tool = state["tool"]

                # Check if skill is done
                if hasattr(tool, 'skill') and hasattr(tool.skill, '_is_skill_done'):
                    is_done = tool.skill._is_skill_done(
                        raw_obs, None, None, torch.ones(1, 1), 0
                    )
                    if is_done:
                        state["done"] = True
                        state["completed"] = True
                        if not state["response"]:
                            state["response"] = "Successful execution!"
                        continue

                low_level_action, response = state["agent"].process_high_level_action(
                    state["action_name"], state["target"], raw_obs
                )

                response_text = (response or "").strip()
                if response_text:
                    state["done"] = True
                    state["response"] = response_text
                    if self._is_action_failure_text(response_text):
                        state["failed"] = True
                    elif self._is_action_success_text(response_text):
                        state["completed"] = True
                    else:
                        state["failed"] = True
                    state["low_level_action"] = None
                    continue

                if low_level_action is None:
                    state["done"] = True
                    state["failed"] = True
                    state["response"] = "Action ended without explicit success signal."
                    state["low_level_action"] = None
                else:
                    state["low_level_action"] = low_level_action

        # Phase 3: Collect results and apply mechanics
        for uid, state in agent_states.items():
            obs_text = (state["response"] or "").strip()
            obs_text = self._append_textual_visual_summary(
                obs_text,
                uid,
                surroundings.get(uid, []),
                agents_passed.get(uid),
            )
            mech_result = state["mech_result"]

            # Handle non-terminated actions as explicit failures.
            if not state["done"]:
                state["failed"] = True
                if self._episode_done:
                    obs_text = "Action interrupted because the episode ended."
                else:
                    obs_text = f"Action timed out after {max_skill_steps} simulator steps."

            if state["failed"] or not state["completed"]:
                if not obs_text:
                    obs_text = "Action failed without an explicit error message."
                results[uid] = {"success": False, "observation": obs_text}
                continue

            postcondition_error = self._check_action_postcondition(
                uid,
                state["action_name"],
                state["target"],
            )
            if postcondition_error:
                obs_text = f"{obs_text} Postcondition check failed: {postcondition_error}".strip()
                results[uid] = {"success": False, "observation": obs_text}
                continue

            # Apply mechanic state changes
            action_name = state["action_name"]
            if mech_result.applies:
                agent_id = f"agent_{uid}"
                target = state["target"]
                _, mechanic_result = self.game_manager.apply_action(action_name, agent_id, target)
                self._sync_remote_effects_to_simulator(mechanic_result.effects)
                if mech_result.observation:
                    obs_text = f"{obs_text} {mech_result.observation}"

            if not obs_text:
                obs_text = "Successful execution!"

            results[uid] = {
                "success": True,
                "observation": obs_text,
                "skill_steps": state["skill_steps"],
            }

        return results

    def save_video(self, suffix: str) -> Optional[str]:
        """Save recorded video with given suffix."""
        video_dir = os.path.join(self.output_dir, "videos")

        if self._dvu and self._dvu.frames:
            try:
                self._dvu._make_video(play=False, postfix=suffix)
            except Exception as e:
                print(f"[EnactToMBaseRunner] Failed to save combined video: {e}")

        if self._per_agent_recorder:
            try:
                self._per_agent_recorder.save_individual_videos(postfix=suffix)
            except Exception as e:
                print(f"[EnactToMBaseRunner] Failed to save per-agent videos: {e}")

        if (self._dvu and self._dvu.frames) or self._per_agent_recorder:
            return video_dir
        return None

    def save_planner_log(self, data: Dict[str, Any]) -> str:
        """Save planner log JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, "planner-log", f"planner-log-{timestamp}.json")

        # Include event trace summary
        data["event_trace"] = self.event_log.get_summary()

        with open(log_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[EnactToMBaseRunner] Saved planner log: {log_file}")
        return log_file

    def save_event_log(self, suffix: str = "") -> str:
        """
        Save the full causal event log.

        Args:
            suffix: Optional suffix for the filename

        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_dir = os.path.join(self.output_dir, "traces")
        os.makedirs(trace_dir, exist_ok=True)

        filename = f"event-trace-{timestamp}"
        if suffix:
            filename = f"{filename}-{suffix}"
        filepath = os.path.join(trace_dir, f"{filename}.json")

        self.event_log.save(filepath)
        print(f"[EnactToMBaseRunner] Saved event trace: {filepath}")

        # Also save narrative version for human reading
        narrative_path = os.path.join(trace_dir, f"{filename}-narrative.txt")
        with open(narrative_path, "w") as f:
            f.write(self.event_log.to_narrative())
        print(f"[EnactToMBaseRunner] Saved event narrative: {narrative_path}")

        return filepath

    def get_causal_chain(self, event_id: str) -> list:
        """
        Get the causal chain leading to an event.

        Useful for debugging "why did this happen?"

        Args:
            event_id: Event ID to trace back from

        Returns:
            List of events in chronological order
        """
        return self.event_log.get_causal_chain(event_id)

    def save_prompts(self, prompts: Dict[str, str], traces: Optional[Dict[str, str]] = None) -> None:
        """
        Save per-agent prompts and traces.

        Args:
            prompts: Dict mapping agent_id -> prompt text
            traces: Optional dict mapping agent_id -> trace text
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for agent_id, prompt in prompts.items():
            # Extract uid from agent_id (e.g., "agent_0" -> "0")
            uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id

            prompt_dir = os.path.join(self.output_dir, "prompts", uid)
            os.makedirs(prompt_dir, exist_ok=True)

            prompt_file = os.path.join(prompt_dir, f"prompt-{timestamp}-{uid}.txt")
            with open(prompt_file, "w") as f:
                f.write(prompt)

        if traces:
            for agent_id, trace in traces.items():
                uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id

                trace_dir = os.path.join(self.output_dir, "traces", uid)
                os.makedirs(trace_dir, exist_ok=True)

                trace_file = os.path.join(trace_dir, f"trace-{timestamp}-{uid}.txt")
                with open(trace_file, "w") as f:
                    f.write(trace)

    def cleanup(self) -> None:
        """Close environment and release resources."""
        if self.env_interface:
            try:
                self.env_interface.env.close()
            except Exception:
                pass

    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from environment."""
        return self.env_interface.get_observations()

    def get_sim_steps(self) -> int:
        """
        Get actual simulation step count from Habitat environment.

        This is the true physics step count (up to max_steps like 20k),
        not the number of LLM turns/actions.
        """
        try:
            # Chain: env_interface.env (GymHabitatEnv) -> env (HabGymWrapper)
            # -> env (RLTaskEnv) -> _env (Env with _elapsed_steps)
            return self.env_interface.env.env.env._env._elapsed_steps
        except AttributeError:
            # Fallback if structure differs
            return 0

    def get_world_graph(self) -> Dict[int, Any]:
        """Get the current per-agent world graph."""
        world_graph = {}
        for uid in self.agents.keys():
            try:
                world_graph[uid] = self.env_interface.world_graph[uid]
            except Exception:
                world_graph[uid] = None
        return world_graph

    def _get_surroundings_description(self, uid: int, skill_step: int) -> str:
        """
        Build a concise one-line surroundings description from the agent's world graph.

        Args:
            uid: Agent UID
            skill_step: Current motor skill step number

        Returns:
            Formatted string like "[Step 30] kitchen_0: apple_1 (on counter_3). Agent_1 in bedroom_0."
        """
        from habitat_llm.world_model.entity import Room, Object, SpotRobot

        try:
            wg = self.env_interface.world_graph[uid]
        except (KeyError, AttributeError, TypeError):
            return f"[Step {skill_step}] (surroundings unavailable)"

        # Find agent's current room
        agent_name = f"agent_{uid}"
        try:
            agent_node = wg.get_node_from_name(agent_name)
            room_neighbors = wg.get_neighbors_of_type(agent_node, Room)
            current_room = room_neighbors[0].name if room_neighbors else "unknown"
        except (ValueError, IndexError, AttributeError):
            current_room = "unknown"

        # Find objects in the current room (on furniture)
        object_parts = []
        try:
            for obj in wg.get_all_objects():
                try:
                    obj_room = wg.get_room_for_entity(obj)
                except ValueError:
                    continue
                if obj_room and obj_room.name == current_room:
                    furniture = wg.find_furniture_for_object(obj)
                    if furniture:
                        object_parts.append(f"{obj.name} (on {furniture.name})")
                    else:
                        object_parts.append(obj.name)
        except (AttributeError, TypeError):
            pass

        objects_str = ", ".join(object_parts[:5]) if object_parts else "no objects nearby"

        # Find other agents and their rooms
        other_agents_parts = []
        try:
            for node in wg.get_all_nodes_of_type(SpotRobot) or []:
                if node.name == agent_name:
                    continue
                other_rooms = wg.get_neighbors_of_type(node, Room)
                other_room = other_rooms[0].name if other_rooms else "unknown"
                if other_room == current_room:
                    other_agents_parts.append(f"{node.name} nearby")
                else:
                    other_agents_parts.append(f"{node.name} in {other_room}")
        except (AttributeError, TypeError):
            pass

        other_agents_str = ". ".join(other_agents_parts)

        line = f"[Step {skill_step}] {current_room}: {objects_str}."
        if other_agents_str:
            line += f" {other_agents_str}."
        return line

    def _get_nearby_agents(self, uid: int) -> List[tuple]:
        """
        Return other agents that are in the same room as the given agent.

        Returns:
            List of (agent_name, room_name) tuples for co-located agents.
        """
        from habitat_llm.world_model.entity import Room, SpotRobot

        try:
            wg = self.env_interface.world_graph[uid]
        except (KeyError, AttributeError, TypeError):
            return []

        agent_name = f"agent_{uid}"
        try:
            agent_node = wg.get_node_from_name(agent_name)
            room_neighbors = wg.get_neighbors_of_type(agent_node, Room)
            current_room = room_neighbors[0].name if room_neighbors else None
        except (ValueError, IndexError, AttributeError):
            return []

        if not current_room:
            return []

        nearby = []
        try:
            for node in wg.get_all_nodes_of_type(SpotRobot) or []:
                if node.name == agent_name:
                    continue
                other_rooms = wg.get_neighbors_of_type(node, Room)
                if other_rooms and other_rooms[0].name == current_room:
                    nearby.append((node.name, current_room))
        except (AttributeError, TypeError):
            pass

        return nearby

    def evaluate_task(
        self,
        success_condition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion using Habitat predicates plus EnactToM overlay predicates.

        Handles both:
        - Simulator predicates (is_on_top, is_inside, etc.) via evaluation.py
        - Overlay predicates (such as is_unlocked) via GameStateManager

        Args:
            success_condition: Task's success_condition dict. If None, returns empty result.

        Returns:
            Dict with percent_complete, success, failure_explanations
        """
        if success_condition is None:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": ["No success_condition defined"],
            }

        try:
            from enacttom.evaluation import evaluate_task
            from enacttom.state.manager import GameStateManager
            sim = self.env_interface.sim

            # Get world graph for name-to-handle resolution (prefer full observability)
            world_graph = getattr(self.env_interface, "full_world_graph", None)
            if not world_graph and hasattr(self.env_interface, "world_graph"):
                for uid in self.agents.keys():
                    if uid in self.env_interface.world_graph:
                        world_graph = self.env_interface.world_graph[uid]
                        break

            required_states = success_condition.get("required_states", [])

            # Split predicates: game state vs simulator
            game_state_predicates = GameStateManager.GAME_STATE_PREDICATES
            simulator_conditions = []
            game_state_results = {}
            failure_explanations = []

            for i, prop in enumerate(required_states):
                prop_id = prop.get("prop_id", f"prop_{i}")
                property_name = prop.get("property", "")

                if property_name in game_state_predicates and self.game_manager:
                    # Check via GameStateManager
                    result = self.game_manager._check_game_state_predicate(prop)
                    if result is None:
                        # Shouldn't happen, but fallback
                        game_state_results[prop_id] = False
                        failure_explanations.append(f"Could not evaluate {prop_id}")
                    else:
                        game_state_results[prop_id] = result
                        if not result:
                            entity = prop.get("entity", "")
                            target = prop.get("target", prop.get("value", ""))
                            failure_explanations.append(
                                f"{entity} does not have {property_name.replace('has_', '')} {target}"
                            )
                else:
                    # Delegate to simulator evaluation - preserve original prop_id
                    prop_with_id = dict(prop)
                    prop_with_id["prop_id"] = prop_id
                    simulator_conditions.append(prop_with_id)

            # Evaluate simulator conditions
            proposition_status = dict(game_state_results)
            if simulator_conditions:
                sim_condition = {
                    "description": success_condition.get("description", ""),
                    "required_states": simulator_conditions,
                }
                result = evaluate_task(sim_condition, sim, world_graph=world_graph)
                proposition_status.update(result.proposition_status)
                failure_explanations.extend(result.failure_explanations)

            # Calculate overall success
            total = len(required_states) if required_states else 1
            satisfied = sum(1 for v in proposition_status.values() if v)
            percent_complete = satisfied / total if total > 0 else 1.0

            return {
                "percent_complete": percent_complete,
                "success": percent_complete == 1.0,
                "failure_explanations": failure_explanations,
                "proposition_status": proposition_status,
            }
        except Exception as e:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": [f"Evaluation error: {str(e)}"],
            }

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Main execution loop. Implemented by subclasses.

        Returns:
            Dict with results (steps, history, etc.)
        """
        pass
