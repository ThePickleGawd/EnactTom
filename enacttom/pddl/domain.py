"""
Shared EnactToM PDDL domain definition.

Encodes all EnactToM actions, predicates, and types with conditional effects
for mechanics (inverse_state, state_mirroring, remote_control, etc.).
"""

from __future__ import annotations

from typing import Optional, Set

from enacttom.pddl.dsl import (
    Type,
    Predicate,
    Param,
    Action,
    Effect,
    ForallEffect,
    Literal,
    And,
    Formula,
    Not,
    Domain,
)


# ---------------------------------------------------------------------------
# Type hierarchy
# ---------------------------------------------------------------------------

ENACTTOM_TYPES = [
    Type("agent"),
    Type("object"),
    Type("furniture", parent="object"),
    Type("room"),
]


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

ENACTTOM_PREDICATES = [
    # Spatial / relational
    Predicate("is_on_top", (Param("x", "object"), Param("y", "furniture"))),
    Predicate("is_inside", (Param("x", "object"), Param("y", "furniture"))),
    Predicate("is_in_room", (Param("x", "object"), Param("r", "room"))),
    Predicate("is_on_floor", (Param("x", "object"),)),
    Predicate("is_next_to", (Param("x", "object"), Param("y", "object"))),

    # Unary state
    Predicate("is_open", (Param("f", "furniture"),)),
    Predicate("is_closed", (Param("f", "furniture"),)),
    Predicate("is_clean", (Param("x", "object"),)),
    Predicate("is_dirty", (Param("x", "object"),)),
    Predicate("is_filled", (Param("x", "object"),)),
    Predicate("is_empty", (Param("x", "object"),)),
    Predicate("is_powered_on", (Param("x", "object"),)),
    Predicate("is_powered_off", (Param("x", "object"),)),
    Predicate("is_unlocked", (Param("f", "furniture"),)),
    Predicate("is_locked", (Param("f", "furniture"),)),

    # Agent predicates
    Predicate("is_held_by", (Param("x", "object"), Param("a", "agent"))),
    Predicate("agent_in_room", (Param("a", "agent"), Param("r", "room"))),

    # Mechanic predicates
    Predicate("is_inverse", (Param("f", "furniture"),)),
    Predicate("mirrors", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("mirrors_closed", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls_unlocked", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls_closed", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("controls_locks", (Param("f1", "furniture"), Param("f2", "furniture"))),
    Predicate("is_restricted", (Param("a", "agent"), Param("r", "room"))),
    Predicate("can_communicate", (Param("from", "agent"), Param("to", "agent"))),
]


# ---------------------------------------------------------------------------
# Predicate descriptions (for prompt generation)
# ---------------------------------------------------------------------------

# Maps predicate name to a one-line description for the LLM prompt.
# Grouped by comment headers in ENACTTOM_PREDICATES above.
_PREDICATE_DESCRIPTIONS = {
    "is_on_top": "object is on top of furniture",
    "is_inside": "object is inside furniture (container)",
    "is_in_room": "object is located in room",
    "is_on_floor": "object is on the floor",
    "is_next_to": "object is adjacent to another object",
    "is_open": "furniture is open",
    "is_closed": "furniture is closed",
    "is_clean": "object is clean",
    "is_dirty": "object is dirty",
    "is_filled": "object is filled with liquid",
    "is_empty": "object is empty",
    "is_powered_on": "object is powered on",
    "is_powered_off": "object is powered off",
    "is_unlocked": "furniture is unlocked",
    "is_locked": "furniture is locked",
    "is_held_by": "object is held by agent",
    "agent_in_room": "agent is in room",
    "is_inverse": "(mechanic) furniture has inverted open/close",
    "mirrors": "(mechanic) furniture1 state mirrors furniture2",
    "mirrors_closed": "(mechanic) furniture1 open/close toggles furniture2 closed/open state",
    "controls": "(mechanic) furniture1 remotely controls furniture2",
    "controls_unlocked": "(mechanic) furniture1 remotely controls furniture2 unlocked/locked state",
    "controls_closed": "(mechanic) furniture1 remotely controls furniture2 closed/open state",
    "controls_locks": "(mechanic) furniture1 remotely controls furniture2 locked/unlocked state",
    "is_restricted": "(mechanic) agent cannot enter room",
    "can_communicate": "(mechanic) agent can send messages to another agent",
}

_PREDICATE_GROUPS = [
    ("Spatial / Relational", ["is_on_top", "is_inside", "is_in_room", "is_on_floor", "is_next_to"]),
    ("Unary State", ["is_open", "is_closed", "is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off", "is_unlocked", "is_locked"]),
    ("Agent", ["is_held_by", "agent_in_room"]),
    ("Mechanic (init-only, do NOT use in pddl_goal)", ["is_inverse", "mirrors", "mirrors_closed", "controls", "controls_unlocked", "controls_closed", "controls_locks", "is_restricted", "can_communicate"]),
]

INIT_ONLY_PREDICATES = {
    "is_inverse",
    "mirrors",
    "mirrors_closed",
    "controls",
    "controls_unlocked",
    "controls_closed",
    "controls_locks",
    "is_restricted",
    "can_communicate",
}


def validate_goal_formula_allowed(formula: Formula) -> list[str]:
    """Reject init-only mechanic predicates anywhere in goal space."""
    from enacttom.pddl.dsl import And, Believes, Knows, Literal, Not, Or

    errors: list[str] = []

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            if node.predicate in INIT_ONLY_PREDICATES:
                errors.append(
                    f"Predicate '{node.predicate}' is init-only and cannot appear in pddl_goal: {node.to_pddl()}"
                )
            return
        if isinstance(node, (Knows, Believes)):
            _walk(node.inner)
            return
        if isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)
            return
        if isinstance(node, (And, Or)):
            for operand in node.operands:
                _walk(operand)

    _walk(formula)
    return errors


def get_predicates_for_prompt(allowed_predicates: Optional[Set[str]] = None) -> str:
    """
    Generate predicate signatures for the LLM system prompt.

    Dynamically derived from ENACTTOM_PREDICATES — never hardcoded.
    """
    pred_map = {p.name: p for p in ENACTTOM_PREDICATES}
    lines = []
    for group_name, pred_names in _PREDICATE_GROUPS:
        group_lines = []
        for name in pred_names:
            if allowed_predicates is not None and name not in allowed_predicates:
                continue
            pred = pred_map.get(name)
            if not pred:
                continue
            params_str = " ".join(f"{p.name}:{p.type}" for p in pred.params)
            desc = _PREDICATE_DESCRIPTIONS.get(name, "")
            group_lines.append(f"- `({name} {params_str})` — {desc}")
        if group_lines:
            lines.append(f"### {group_name}")
            lines.extend(group_lines)
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ENACTTOM_ACTIONS = [
    # Open: with conditional effects for mechanics
    Action(
        name="open",
        params=[Param("a", "agent"), Param("f", "furniture"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?f", "?r")),
            Literal("is_closed", ("?f",)),
            Not(operand=Literal("is_locked", ("?f",))),
        )),
        effects=[
            Effect(Literal("is_open", ("?f",))),
            Effect(Literal("is_closed", ("?f",), negated=True)),
            # Conditional: inverse mechanic — undo the open, leave it closed
            Effect(
                Literal("is_closed", ("?f",)),
                condition=Literal("is_inverse", ("?f",)),
            ),
            Effect(
                Literal("is_open", ("?f",), negated=True),
                condition=Literal("is_inverse", ("?f",)),
            ),
            # Conditional: state mirroring propagates (forall quantified)
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors_closed", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            # Conditional: remote control triggers unlock (forall quantified)
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_unlocked", ("?f", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_closed", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_locks", ("?f", "?g")),
                effect=Literal("is_locked", ("?g",)),
                negative_effect=Literal("is_unlocked", ("?g",)),
            ),
        ],
        observability="full",
    ),

    # Close
    Action(
        name="close",
        params=[Param("a", "agent"), Param("f", "furniture"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?f", "?r")),
            Literal("is_open", ("?f",)),
        )),
        effects=[
            Effect(Literal("is_closed", ("?f",))),
            Effect(Literal("is_open", ("?f",), negated=True)),
            # Conditional: inverse mechanic — undo the close, leave it open
            Effect(
                Literal("is_open", ("?f",)),
                condition=Literal("is_inverse", ("?f",)),
            ),
            Effect(
                Literal("is_closed", ("?f",), negated=True),
                condition=Literal("is_inverse", ("?f",)),
            ),
            # Conditional: state mirroring propagates (forall quantified)
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("mirrors_closed", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls", ("?f", "?g")),
                effect=Literal("is_closed", ("?g",)),
                negative_effect=Literal("is_open", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_unlocked", ("?f", "?g")),
                effect=Literal("is_locked", ("?g",)),
                negative_effect=Literal("is_unlocked", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_closed", ("?f", "?g")),
                effect=Literal("is_open", ("?g",)),
                negative_effect=Literal("is_closed", ("?g",)),
            ),
            ForallEffect(
                variable=Param("g", "furniture"),
                condition=Literal("controls_locks", ("?f", "?g")),
                effect=Literal("is_unlocked", ("?g",)),
                negative_effect=Literal("is_locked", ("?g",)),
            ),
        ],
        observability="full",
    ),

    # Navigate: move agent to a room, remove from old room
    Action(
        name="navigate",
        params=[Param("a", "agent"), Param("r", "room")],
        preconditions=Not(operand=Literal("is_restricted", ("?a", "?r"))),
        effects=[
            Effect(Literal("agent_in_room", ("?a", "?r"))),
            # Remove agent from any previous room
            ForallEffect(
                variable=Param("old", "room"),
                condition=Literal("agent_in_room", ("?a", "?old")),
                effect=Literal("agent_in_room", ("?a", "?r")),
                negative_effect=Literal("agent_in_room", ("?a", "?old")),
            ),
        ],
        observability="full",
    ),

    # Pick
    Action(
        name="pick",
        params=[Param("a", "agent"), Param("x", "object"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?x", "?r")),
        )),
        effects=[
            Effect(Literal("is_held_by", ("?x", "?a"))),
        ],
        observability="full",
    ),

    # Place
    Action(
        name="place",
        params=[Param("a", "agent"), Param("x", "object"), Param("f", "furniture"), Param("r", "room")],
        preconditions=And(operands=(
            Literal("is_held_by", ("?x", "?a")),
            Literal("agent_in_room", ("?a", "?r")),
            Literal("is_in_room", ("?f", "?r")),
        )),
        effects=[
            Effect(Literal("is_held_by", ("?x", "?a"), negated=True)),
            # Domain-level abstraction: runtime Place can realize either
            # on-top or within placement depending on relation argument.
            # We expose both so solvability checks remain aligned with
            # task-level goals authored in problem_pddl.
            Effect(Literal("is_on_top", ("?x", "?f"))),
            Effect(Literal("is_inside", ("?x", "?f"))),
            Effect(Literal("is_in_room", ("?x", "?r"))),
        ],
        observability="full",
    ),

    # Communicate: epistemic effect — transfers knowledge
    Action(
        name="communicate",
        params=[Param("from", "agent"), Param("to", "agent")],
        preconditions=Literal("can_communicate", ("?from", "?to")),
        effects=[],  # Epistemic effects handled by the solver
        observability="full",
    ),

    # Wait: no-op
    Action(
        name="wait",
        params=[Param("a", "agent")],
        preconditions=None,
        effects=[],
        observability="full",
    ),

]


# ---------------------------------------------------------------------------
# Domain singleton
# ---------------------------------------------------------------------------

ENACTTOM_DOMAIN = Domain(
    name="enacttom",
    types=ENACTTOM_TYPES,
    predicates=ENACTTOM_PREDICATES,
    actions=ENACTTOM_ACTIONS,
)
