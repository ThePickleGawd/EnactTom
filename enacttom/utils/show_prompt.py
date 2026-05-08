#!/usr/bin/env python3
"""Utility to show the full prompt for a task (what the LLM actually sees)."""

import json
import sys
from pathlib import Path


# Load the prompt template
PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "habitat_llm" / "conf" / "instruct" / "enacttom_no_visual.yaml"


def load_prompt_template() -> str:
    """Load the prompt template from YAML."""
    import yaml
    with open(PROMPT_TEMPLATE_PATH) as f:
        config = yaml.safe_load(f)
    return config["prompt"]


def get_tool_descriptions(actions: list) -> str:
    """Get tool descriptions for the given actions."""
    from enacttom.actions.registry import STANDARD_ACTIONS
    # Import to register custom actions
    import enacttom.actions.custom_actions  # noqa: F401
    from enacttom.actions.registry import _ACTION_REGISTRY

    lines = []
    for action in actions:
        if action in STANDARD_ACTIONS:
            lines.append(f"- {STANDARD_ACTIONS[action]}")
        elif action in _ACTION_REGISTRY:
            # Custom action from registry
            cls = _ACTION_REGISTRY[action]
            desc = getattr(cls, "action_description", f"{action}: Custom action")
            lines.append(f"- {desc}")
        else:
            lines.append(f"- {action}: Unknown action")
    return "\n".join(lines)


def build_full_prompt(template: str, agent_id: str, agent_num: int, instruction: str, actions: list) -> str:
    """Build the full prompt by filling in the template."""
    tool_descriptions = get_tool_descriptions(actions)

    # For OpenAI/Claude, tags are empty strings
    # The conversation structure is handled by the API's message format
    prompt = template.format(
        system_tag="",
        user_tag="",
        assistant_tag="",
        eot_tag="",
        id=agent_num,
        tool_descriptions=tool_descriptions,
        input=instruction,
    )

    return prompt


def show_prompt(task_path: str) -> None:
    """Load a task and print the full per-agent prompts."""
    from enacttom.task_gen import GeneratedTask
    from enacttom.runner.benchmark import task_to_instruction

    # Load task
    with open(task_path) as f:
        task_data = json.load(f)

    task = GeneratedTask.from_dict(task_data)
    instructions = task_to_instruction(task)

    # Load prompt template
    template = load_prompt_template()

    print(f"Task: {task.title}")
    print(f"=" * 80)

    for agent_id, instruction in instructions.items():
        # Extract agent number from agent_id (e.g., "agent_0" -> 0)
        agent_num = int(agent_id.split("_")[1])
        actions = task.agent_actions.get(agent_id, [])

        full_prompt = build_full_prompt(template, agent_id, agent_num, instruction, actions)

        print(f"\n{'=' * 80}")
        print(f"FULL PROMPT FOR {agent_id.upper()}")
        print(f"{'=' * 80}\n")
        print(full_prompt)
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m enacttom.utils.show_prompt <task.json>")
        sys.exit(1)

    show_prompt(sys.argv[1])
