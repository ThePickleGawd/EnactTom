"""
PDDL → Human-readable description.

Derives tom_level, tom_reasoning, and task descriptions directly from PDDL.
No redundant storage — these are always computed from the PDDL source of truth.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from enacttom.pddl.dsl import Formula, Literal, And, Or, Not, Knows, Believes
from enacttom.pddl.problem_pddl import parse_problem_pddl
from enacttom.pddl.tom_verifier import compute_tom_depth, explain_tom_depth

if TYPE_CHECKING:
    from enacttom.task_gen.task_generator import GeneratedTask


# Map from PDDL predicate names to natural language templates
_PREDICATE_NL = {
    "is_open": "{entity} is open",
    "is_closed": "{entity} is closed",
    "is_on_top": "{entity} is on top of {target}",
    "is_inside": "{entity} is inside {target}",
    "is_in_room": "{entity} is in {target}",
    "is_on_floor": "{entity} is on the floor",
    "is_next_to": "{entity} is next to {target}",
    "is_clean": "{entity} is clean",
    "is_dirty": "{entity} is dirty",
    "is_filled": "{entity} is filled",
    "is_empty": "{entity} is empty",
    "is_powered_on": "{entity} is powered on",
    "is_powered_off": "{entity} is powered off",
    "is_held_by": "{entity} is held by {target}",
    "is_unlocked": "{entity} is unlocked",
}


def _literal_to_nl(lit: Literal) -> str:
    """Convert a single literal to natural language."""
    template = _PREDICATE_NL.get(lit.predicate)
    if not template:
        # Fallback: just format the predicate and args
        args_str = ", ".join(lit.args)
        text = f"{lit.predicate}({args_str})"
    else:
        entity = lit.args[0] if lit.args else "?"
        target = lit.args[1] if len(lit.args) > 1 else "?"
        text = template.format(entity=entity, target=target)

    # Format object IDs for readability: cabinet_27 -> "cabinet 27"
    text = _format_object_id(text)

    if lit.negated:
        text = f"NOT: {text}"
    return text


def _format_object_id(text: str) -> str:
    """Make object IDs more readable in descriptions."""
    import re
    # Match patterns like cabinet_27, table_13 and format as "cabinet 27"
    def _replace(m):
        name = m.group(1).replace("_", " ")
        return f"{name} {m.group(2)}"
    return re.sub(r'\b([a-z]+(?:_[a-z]+)*)_(\d+)\b', _replace, text)


def goal_to_natural_language(goal: Formula) -> str:
    """Convert a PDDL goal formula to a natural language description."""
    if isinstance(goal, Literal):
        return _literal_to_nl(goal)

    if isinstance(goal, And):
        parts = [goal_to_natural_language(op) for op in goal.operands]
        if len(parts) == 1:
            return parts[0]
        return "Complete all of: " + "; ".join(parts)

    if isinstance(goal, Or):
        parts = [goal_to_natural_language(op) for op in goal.operands]
        return "Complete any of: " + " OR ".join(parts)

    if isinstance(goal, Not):
        inner = goal_to_natural_language(goal.operand)
        return f"NOT: {inner}"

    if isinstance(goal, Knows):
        inner_nl = goal_to_natural_language(goal.inner)
        return f"{_format_object_id(goal.agent)} knows that: {inner_nl}"

    if isinstance(goal, Believes):
        inner_nl = goal_to_natural_language(goal.inner)
        return f"{_format_object_id(goal.agent)} believes that: {inner_nl}"

    return goal.to_pddl()


def describe_task(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate tom_level, tom_reasoning, and task description from PDDL.

    This is the primary API for deriving human-readable fields.
    No redundant storage — always computed from the PDDL source of truth.

    Args:
        task: The generated task with problem_pddl
        scene_data: Optional scene data for context

    Returns:
        Dict with tom_level, tom_reasoning, and optional generated description
    """
    result: Dict[str, Any] = {}

    # Compute ToM depth
    tom_info = explain_tom_depth(task, scene_data)
    result["tom_level"] = tom_info["tom_level"]
    result["tom_reasoning"] = tom_info["tom_reasoning"]
    result["information_gaps"] = tom_info["information_gaps"]
    result["communication_required"] = tom_info["communication_required"]

    # Generate description from problem_pddl goal.
    problem_pddl = getattr(task, "problem_pddl", None)
    if isinstance(problem_pddl, str) and problem_pddl.strip():
        parsed_problem = parse_problem_pddl(problem_pddl)
        result["generated_description"] = goal_to_natural_language(parsed_problem.goal_formula)

    return result
