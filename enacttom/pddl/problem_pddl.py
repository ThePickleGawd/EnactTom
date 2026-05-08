"""
Utilities for inline task-level PDDL problem strings.

`problem_pddl` is stored inside task JSON as the single authoritative
problem specification. This module parses that string into the existing
DSL dataclasses used by the solver/checker stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

from enacttom.pddl.dsl import (
    And,
    Believes,
    EpistemicFormula,
    Formula,
    Knows,
    Literal,
    Not,
    Problem,
    parse_goal_string,
)


_ProblemInitExpr = Union[Literal, Knows, Believes]


@dataclass(frozen=True)
class ParsedProblemPDDL:
    """Parsed inline problem PDDL payload."""

    problem_name: str
    domain_name: str
    objects: Dict[str, str]
    init_literals: List[Literal]
    epistemic_init: List[Union[Knows, Believes]]
    goal_formula: Formula
    goal_pddl: str
    owners: Dict[str, str] = None  # literal PDDL string -> owner ID

    def __post_init__(self):
        if self.owners is None:
            object.__setattr__(self, 'owners', {})

    def to_problem(self) -> Problem:
        """Convert to DSL `Problem` dataclass."""
        return Problem(
            name=self.problem_name,
            domain_name=self.domain_name,
            objects=self.objects,
            init=self.init_literals,
            goal=self.goal_formula,
            epistemic_init=self.epistemic_init,
        )


def parse_problem_pddl(problem_pddl: str) -> ParsedProblemPDDL:
    """
    Parse an inline PDDL problem definition from `task.json`.

    Supported sections:
    - `(:domain ...)`
    - `(:objects ...)`
    - `(:init ...)`
    - `(:goal ...)`

    Numeric fluents / assignments are intentionally unsupported in v1.
    """
    raw = _strip_comments(problem_pddl or "").strip()
    if not raw:
        raise ValueError("problem_pddl is empty")

    problem_name = _extract_problem_name(raw)
    domain_name = _extract_domain_name(raw)
    try:
        objects_text = _extract_section(raw, "objects")
    except ValueError:
        objects_text = ""  # :objects is optional
    init_text = _extract_section(raw, "init")
    goal_text = _extract_goal(raw)

    objects = _parse_objects_block(objects_text)
    init_literals, epistemic_init = _parse_init_block(init_text)
    goal_formula = parse_goal_string(goal_text)

    # Parse optional :goal-owners section
    owners = _parse_goal_owners(raw)

    return ParsedProblemPDDL(
        problem_name=problem_name,
        domain_name=domain_name,
        objects=objects,
        init_literals=init_literals,
        epistemic_init=epistemic_init,
        goal_formula=goal_formula,
        goal_pddl=goal_text,
        owners=owners,
    )


def extract_goal_from_problem_pddl(problem_pddl: str) -> str:
    """Extract raw `:goal` expression from an inline problem string."""
    raw = _strip_comments(problem_pddl or "").strip()
    if not raw:
        raise ValueError("problem_pddl is empty")
    return _extract_goal(raw)


def replace_goal_in_problem_pddl(problem_pddl: str, new_goal_pddl: str) -> str:
    """Replace the `:goal` formula in an inline problem string."""
    raw = _strip_comments(problem_pddl or "").strip()
    if not raw:
        raise ValueError("problem_pddl is empty")

    idx = _find_goal_idx(raw)
    if idx < 0:
        raise ValueError("problem_pddl is missing (:goal ...)")

    pos = idx + len("(:goal")
    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    if pos >= len(raw) or raw[pos] != "(":
        raise ValueError("(:goal ...) must contain a parenthesized formula")

    end = _find_matching_paren(raw, pos)
    return raw[:pos] + new_goal_pddl.strip() + raw[end + 1 :]


def replace_init_in_problem_pddl(problem_pddl: str, new_init_pddl: str) -> str:
    """Replace the `:init` block in an inline problem string."""
    raw = _strip_comments(problem_pddl or "").strip()
    if not raw:
        raise ValueError("problem_pddl is empty")

    lower = raw.lower()
    needle = "(:init"
    idx = lower.find(needle)
    if idx < 0:
        raise ValueError("problem_pddl is missing (:init ...)")

    pos = idx + len(needle)
    end = _find_matching_paren(raw, idx)
    return raw[:pos] + "\n    " + new_init_pddl.strip() + "\n  " + raw[end:]


def normalize_problem_pddl(problem_pddl: str) -> str:
    """Normalize compatibility-only init facts in inline problem PDDL.

    The current planner domain does not define ``is_openable``. Older prompts
    and model priors sometimes emit it as a redundant init fact, which makes
    Fast Downward reject the problem even though the rest of the task is valid.
    Drop that compatibility alias from ``:init`` before planner-facing use.
    """
    parsed = parse_problem_pddl(problem_pddl)
    filtered_init = [
        lit for lit in parsed.init_literals
        if lit.predicate != "is_openable"
    ]
    if len(filtered_init) == len(parsed.init_literals):
        return problem_pddl

    init_exprs = [lit.to_pddl() for lit in filtered_init]
    init_exprs.extend(expr.to_pddl() for expr in parsed.epistemic_init)
    return replace_init_in_problem_pddl(problem_pddl, "\n    ".join(init_exprs))


def collect_object_ids_from_formula(formula: Formula) -> Set[str]:
    """Collect grounded object IDs referenced in a formula."""
    out: Set[str] = set()

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            for arg in node.args:
                if not arg.startswith("?"):
                    out.add(arg)
            return
        if isinstance(node, EpistemicFormula):
            _walk(node.inner)
            return
        if hasattr(node, "operands"):
            for op in getattr(node, "operands", []) or []:
                _walk(op)
            return
        if isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)

    _walk(formula)
    return out


def collect_object_ids_from_init(
    init_literals: List[Literal],
    epistemic_init: Optional[List[Union[Knows, Believes]]] = None,
) -> Set[str]:
    """Collect grounded object IDs referenced in an init block."""
    out: Set[str] = set()

    for lit in init_literals:
        for arg in lit.args:
            if not arg.startswith("?"):
                out.add(arg)

    for expr in epistemic_init or []:
        out.update(collect_object_ids_from_formula(expr))

    return out


def build_object_room_map_from_problem(parsed_problem: ParsedProblemPDDL) -> Dict[str, str]:
    """Build entity -> room mapping from explicit init facts only.

    Rooms map to themselves so observability checks work for room-valued facts
    such as ``(agent_in_room agent_1 kitchen_1)``.
    """
    room_map: Dict[str, str] = {}

    for obj_id, obj_type in parsed_problem.objects.items():
        if obj_type == "room":
            room_map[obj_id] = obj_id

    for lit in parsed_problem.init_literals:
        if lit.negated:
            continue
        if lit.predicate == "is_in_room" and len(lit.args) == 2:
            obj_id, room_id = lit.args
            room_map[obj_id] = room_id

    return room_map


def validate_problem_pddl_self_contained(
    parsed_problem: ParsedProblemPDDL,
    *,
    num_agents: Optional[int] = None,
    require_room_grounding: bool = True,
) -> List[str]:
    """Validate raw problem PDDL as a self-contained proof artifact."""
    errors: List[str] = []
    declared = set(parsed_problem.objects.keys())

    init_refs = collect_object_ids_from_init(
        parsed_problem.init_literals, parsed_problem.epistemic_init
    )
    goal_refs = collect_object_ids_from_formula(parsed_problem.goal_formula)

    undeclared_init = sorted(ref for ref in init_refs if ref not in declared)
    undeclared_goal = sorted(ref for ref in goal_refs if ref not in declared)

    if undeclared_init:
        errors.append(
            "problem_pddl :init references undeclared objects: "
            + ", ".join(undeclared_init)
        )
    if undeclared_goal:
        errors.append(
            "problem_pddl :goal references undeclared objects: "
            + ", ".join(undeclared_goal)
        )

    if num_agents is not None:
        expected_agents = {f"agent_{i}" for i in range(num_agents)}
        declared_agents = {
            name for name, typ in parsed_problem.objects.items() if typ == "agent"
        }
        if declared_agents != expected_agents:
            errors.append(
                "problem_pddl :objects must declare exactly the benchmark agents: "
                f"expected {sorted(expected_agents)}, got {sorted(declared_agents)}"
            )

    if require_room_grounding:
        room_map = build_object_room_map_from_problem(parsed_problem)
        room_typed_objects = {
            name for name, typ in parsed_problem.objects.items() if typ in {"object", "furniture"}
        }

        relevant_room_objects = set()
        for obj_id in init_refs | goal_refs:
            if obj_id in room_typed_objects:
                relevant_room_objects.add(obj_id)

        missing_room_grounding = sorted(
            obj_id for obj_id in relevant_room_objects if obj_id not in room_map
        )
        if missing_room_grounding:
            errors.append(
                "problem_pddl must ground room membership with "
                "`(is_in_room <object> <room>)` for all goal/mechanic-relevant "
                "objects and furniture. Missing: "
                + ", ".join(missing_room_grounding)
            )

        expected_agents = {name for name, typ in parsed_problem.objects.items() if typ == "agent"}
        agent_room_grounded = {
            lit.args[0]
            for lit in parsed_problem.init_literals
            if not lit.negated and lit.predicate == "agent_in_room" and len(lit.args) == 2
        }
        missing_agent_rooms = sorted(expected_agents - agent_room_grounded)
        if missing_agent_rooms:
            errors.append(
                "problem_pddl must ground all agents with "
                "`(agent_in_room <agent> <room>)` in :init. Missing: "
                + ", ".join(missing_agent_rooms)
            )

    return errors


def _strip_comments(text: str) -> str:
    # PDDL line comments start with ';'
    return re.sub(r";[^\n]*", "", text)


def _extract_problem_name(text: str) -> str:
    m = re.search(r"\(\s*define\s+\(\s*problem\s+([^\s\)]+)\s*\)", text, flags=re.IGNORECASE)
    if not m:
        return "task_problem"
    return m.group(1)


def _extract_domain_name(text: str) -> str:
    m = re.search(r"\(\s*:domain\s+([^\s\)]+)\s*\)", text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("problem_pddl is missing (:domain ...)")
    return m.group(1)


def _extract_section(text: str, section: str) -> str:
    """
    Extract section payload from `(:<section> ...)`.

    Returns the raw content inside the section (excluding wrapper parens).
    """
    lower = text.lower()
    needle = f"(:{section.lower()}"
    idx = lower.find(needle)
    if idx < 0:
        raise ValueError(f"problem_pddl is missing (:{section} ...)")

    start = idx + len(needle)
    end = _find_matching_paren(text, idx)
    return text[start:end].strip()


def _find_goal_idx(text: str) -> int:
    """Find the index of ``(:goal`` that is NOT ``(:goal-owners``.

    PDDL section keywords are always followed by whitespace or ``(``,
    so we match ``(:goal`` only when the next char is whitespace or ``(``.
    This avoids the prefix collision with ``(:goal-owners``.
    """
    m = re.search(r"\(:goal(?=[\s(])", text, re.IGNORECASE)
    return m.start() if m else -1


def _extract_goal(text: str) -> str:
    idx = _find_goal_idx(text)
    if idx < 0:
        raise ValueError("problem_pddl is missing (:goal ...)")

    pos = idx + len("(:goal")
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "(":
        raise ValueError("(:goal ...) must contain a parenthesized formula")

    end = _find_matching_paren(text, pos)
    return text[pos : end + 1].strip()


def _find_matching_paren(text: str, open_idx: int) -> int:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        raise ValueError("Internal parser error: expected '(' at open index")

    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
            if depth < 0:
                break
    raise ValueError("Unbalanced parentheses in problem_pddl")


def _split_top_level_s_exprs(text: str) -> List[str]:
    exprs: List[str] = []
    depth = 0
    start: Optional[int] = None

    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                exprs.append(text[start : i + 1].strip())
                start = None
            if depth < 0:
                raise ValueError("Unbalanced parentheses while parsing init block")

    if depth != 0:
        raise ValueError("Unbalanced parentheses while parsing init block")

    return [e for e in exprs if e]


def _parse_objects_block(text: str) -> Dict[str, str]:
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return {}

    objects: Dict[str, str] = {}
    pending: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-":
            if not pending:
                raise ValueError("Invalid :objects block: '-' without preceding object names")
            if i + 1 >= len(tokens):
                raise ValueError("Invalid :objects block: missing type after '-'")
            typ = tokens[i + 1]
            for name in pending:
                objects[name] = typ
            pending = []
            i += 2
            continue
        pending.append(tok)
        i += 1

    # Untyped tails default to object.
    for name in pending:
        objects[name] = "object"

    return objects


def _parse_init_expr(expr: str) -> _ProblemInitExpr:
    parsed = parse_goal_string(expr)
    if isinstance(parsed, Literal):
        return parsed
    if isinstance(parsed, (Knows, Believes)):
        return parsed
    if isinstance(parsed, Not) and isinstance(parsed.operand, Literal):
        inner = parsed.operand
        return Literal(predicate=inner.predicate, args=inner.args, negated=not inner.negated)
    raise ValueError(f"Unsupported init expression in problem_pddl: {expr}")


def _parse_init_block(text: str) -> Tuple[List[Literal], List[Union[Knows, Believes]]]:
    literals: List[Literal] = []
    epistemic: List[Union[Knows, Believes]] = []

    for expr in _split_top_level_s_exprs(text):
        parsed = _parse_init_expr(expr)
        if isinstance(parsed, Literal):
            literals.append(parsed)
        else:
            epistemic.append(parsed)

    return literals, epistemic


def _parse_goal_owners(text: str) -> Dict[str, str]:
    """Parse optional (:goal-owners ...) section from problem PDDL.

    Format::

        (:goal-owners
          (agent_0 (is_inside trophy_1 cabinet_10))
          (agent_1 (is_inside trophy_1 cabinet_20)))

    Also accepts the explicit personal wrapper::

        (:goal-owners
          (personal agent_0 (is_open cabinet_10)))

    Returns mapping from PDDL literal string to owner ID.
    """
    lower = text.lower()
    needle = "(:goal-owners"
    idx = lower.find(needle)
    if idx < 0:
        return {}

    start = idx + len(needle)
    end = _find_matching_paren(text, idx)
    body = text[start:end].strip()

    owners: Dict[str, str] = {}
    for entry in _split_top_level_s_exprs(body):
        # Canonical form is (agent_id formula). Also tolerate
        # (personal agent_id formula).
        # Strip outer parens
        inner = entry.strip()
        if inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1].strip()

        owner_id = None
        formula_str = None

        wrapped_parts = inner.split(None, 2)
        if len(wrapped_parts) >= 3 and wrapped_parts[0] == "personal":
            owner_id = wrapped_parts[1]
            formula_str = wrapped_parts[2].strip()
        elif len(wrapped_parts) >= 2 and wrapped_parts[0] == "shared":
            # Shared goals belong in :goal, not :goal-owners.
            continue
        else:
            parts = inner.split(None, 1)
            if len(parts) == 2:
                owner_id = parts[0]
                formula_str = parts[1].strip()

        if not owner_id or not formula_str:
            continue
        if not owner_id.startswith("agent_"):
            continue

        # Normalize the formula via parse+serialize for consistent keys.
        # If formula is compound (and A B C), decompose into individual
        # literals so each one maps to the owner separately.
        try:
            formula = parse_goal_string(formula_str)
            if isinstance(formula, And):
                for operand in formula.operands:
                    owners[operand.to_pddl()] = owner_id
            else:
                owners[formula.to_pddl()] = owner_id
        except ValueError:
            # Ignore malformed owner formulas so later validation can reject
            # the task without canonicalization crashing first.
            continue

    return owners


def strip_goal_owners_pddl(pddl_str: str) -> str:
    """Remove (:goal-owners ...) section from a PDDL string.

    Used before passing to planners that don't understand this extension.
    """
    lower = pddl_str.lower()
    needle = "(:goal-owners"
    idx = lower.find(needle)
    if idx < 0:
        return pddl_str

    end = _find_matching_paren(pddl_str, idx)
    return pddl_str[:idx] + pddl_str[end + 1:]
