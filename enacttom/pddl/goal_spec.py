"""
Unified goal specification for EnactToM tasks.

Consolidates three fragmented task JSON fields (pddl_goal, pddl_ordering,
pddl_owners) into a single ``goals`` array format::

    {
      "goals": [
        {"id": 0, "pddl": "(K agent_0 (is_on_top laptop_0 table_29))", "after": []},
        {"id": 1, "pddl": "(is_on_top spoon_2 couch_15)", "after": [0]},
        {"id": 2, "pddl": "(is_open cabinet_39)", "after": [0], "owner": "agent_0"}
      ]
    }

Each goal entry carries its own ordering dependencies (``after``) and optional
ownership, eliminating the need for separate ordering/owner dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from enacttom.pddl.dsl import (
    And,
    Believes,
    Domain,
    EpistemicFormula,
    Formula,
    Knows,
    parse_goal_string,
    validate_goal_predicates,
)
from enacttom.pddl.domain import validate_goal_formula_allowed


# ---------------------------------------------------------------------------
# GoalEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoalEntry:
    """A single goal within a GoalSpec.

    Attributes:
        id: Unique index for this goal within the spec.
        pddl: The PDDL formula string (e.g. ``"(is_open cabinet_27)"``).
        after: IDs of prerequisite goals that must be satisfied before this
            one. An empty tuple means no ordering constraints.
        owner: Agent responsible for this goal, or ``None`` for shared goals.
    """

    id: int
    pddl: str
    after: Tuple[int, ...] = ()
    owner: Optional[str] = None


# ---------------------------------------------------------------------------
# GoalSpec
# ---------------------------------------------------------------------------

class GoalSpec:
    """Unified goal specification that replaces the legacy triple of
    ``pddl_goal`` / ``pddl_ordering`` / ``pddl_owners``.

    Attributes:
        entries: The ordered tuple of :class:`GoalEntry` instances.
    """

    __slots__ = ("entries",)

    def __init__(self, entries: Tuple[GoalEntry, ...]) -> None:
        self.entries: Tuple[GoalEntry, ...] = entries

    # -- Construction -------------------------------------------------------

    @classmethod
    def from_legacy(
        cls,
        pddl_goal: str,
        ordering: List[Dict[str, str]],
        owners: Dict[str, str],
    ) -> GoalSpec:
        """Build a GoalSpec from the three legacy task JSON fields.

        Args:
            pddl_goal: A PDDL goal string, typically an ``(and ...)``
                conjunction of literals or epistemic formulas.
            ordering: List of ``{"before": "<pddl>", "after": "<pddl>"}``
                dicts that encode temporal ordering between conjuncts.
                ``"after"`` means the second goal must be completed *after*
                the first.
            owners: Mapping from PDDL formula string to owner identifier
                (e.g. ``{"(is_open cabinet_39)": "agent_0"}``).

        Returns:
            A new :class:`GoalSpec`.
        """
        formula = parse_goal_string(pddl_goal)
        conjuncts = formula.flatten()

        # Map each conjunct's PDDL string to its index.
        pddl_to_idx: Dict[str, int] = {}
        for idx, conjunct in enumerate(conjuncts):
            pddl_to_idx[conjunct.to_pddl()] = idx

        # Build ``after`` sets from ordering constraints.
        after_map: Dict[int, Set[int]] = {i: set() for i in range(len(conjuncts))}
        for constraint in ordering:
            before_str = constraint.get("before", "").strip()
            after_str = constraint.get("after", "").strip()
            before_idx = pddl_to_idx.get(before_str)
            after_idx = pddl_to_idx.get(after_str)
            if before_idx is not None and after_idx is not None:
                # "after" goal depends on "before" goal completing first.
                after_map[after_idx].add(before_idx)

        # Build entries.
        entries: List[GoalEntry] = []
        for idx, conjunct in enumerate(conjuncts):
            pddl_str = conjunct.to_pddl()
            owner = owners.get(pddl_str)
            entries.append(
                GoalEntry(
                    id=idx,
                    pddl=pddl_str,
                    after=tuple(sorted(after_map[idx])),
                    owner=owner,
                )
            )

        return cls(entries=tuple(entries))

    @classmethod
    def from_goals_array(cls, goals: List[Dict[str, Any]]) -> GoalSpec:
        """Build a GoalSpec from the new ``goals`` array format.

        Args:
            goals: List of dicts, each containing:
                - ``id`` (int): unique index
                - ``pddl`` (str): the PDDL formula string
                - ``after`` (list[int]): prerequisite goal IDs
                - ``owner`` (str, optional): owner identifier

        Returns:
            A new :class:`GoalSpec`.

        Raises:
            ValueError: On duplicate IDs, invalid ``after`` references,
                or cycles in the dependency graph.
        """
        entries: List[GoalEntry] = []
        seen_ids: Set[int] = set()

        for item in goals:
            gid = int(item["id"])
            if gid in seen_ids:
                raise ValueError(f"Duplicate goal ID: {gid}")
            seen_ids.add(gid)

            entries.append(
                GoalEntry(
                    id=gid,
                    pddl=str(item["pddl"]),
                    after=tuple(int(x) for x in item.get("after", [])),
                    owner=item.get("owner"),
                )
            )

        result = cls(entries=tuple(entries))

        # Validate after-references.
        for entry in entries:
            for dep in entry.after:
                if dep not in seen_ids:
                    raise ValueError(
                        f"Goal {entry.id} references non-existent "
                        f"prerequisite ID {dep}"
                    )

        # Validate no cycles.
        if cls._detect_cycle(result.entries):
            raise ValueError(
                "Cycle detected in goal dependency graph (after references)"
            )

        return result

    # -- Serialization ------------------------------------------------------

    def to_goals_array(self) -> List[Dict[str, Any]]:
        """Serialize to the JSON-friendly ``goals`` array format.

        Returns:
            A list of dicts suitable for embedding in a task JSON file.
        """
        result: List[Dict[str, Any]] = []
        for entry in self.entries:
            item: Dict[str, Any] = {
                "id": entry.id,
                "pddl": entry.pddl,
                "after": list(entry.after),
            }
            if entry.owner is not None:
                item["owner"] = entry.owner
            result.append(item)
        return result

    def to_formula(self) -> Formula:
        """Build a Formula AST from the entries.

        Returns:
            An :class:`And` conjunction of all individual goal formulas.
            If there is exactly one entry, returns its parsed formula
            directly (``And`` with a single operand delegates via
            ``to_pddl()`` anyway, but this keeps the tree clean).
        """
        parsed = tuple(parse_goal_string(e.pddl) for e in self.entries)
        if len(parsed) == 1:
            return parsed[0]
        return And(operands=parsed)

    def to_pddl_string(self) -> str:
        """Return the full PDDL goal string.

        Equivalent to ``self.to_formula().to_pddl()``.
        """
        return self.to_formula().to_pddl()

    # -- Validation ---------------------------------------------------------

    def validate(
        self,
        domain: Domain,
        valid_agents: Set[str],
    ) -> List[str]:
        """Run comprehensive validation on this goal spec.

        Checks performed:
            1. Each goal's PDDL string parses without error.
            2. All predicates exist in *domain* and have the correct arity.
            3. Epistemic wrappers (K/B) reference agents in *valid_agents*.
            4. All ``after`` references point to valid goal IDs.
            5. No cycles in the ``after`` dependency graph.

        Args:
            domain: The PDDL domain containing predicate definitions.
            valid_agents: Set of recognized agent identifiers
                (e.g. ``{"agent_0", "agent_1"}``).

        Returns:
            A list of human-readable error strings. An empty list means
            the spec is valid.
        """
        errors: List[str] = []
        valid_ids = {e.id for e in self.entries}

        for entry in self.entries:
            # Parse check.
            try:
                formula = parse_goal_string(entry.pddl)
            except ValueError as exc:
                errors.append(
                    f"Goal {entry.id}: failed to parse PDDL "
                    f"'{entry.pddl}': {exc}"
                )
                continue

            # Predicate arity check.
            pred_errors = validate_goal_predicates(formula, domain)
            for pe in pred_errors:
                errors.append(f"Goal {entry.id}: {pe}")
            for pe in validate_goal_formula_allowed(formula):
                errors.append(f"Goal {entry.id}: {pe}")

            # Epistemic agent check.
            _check_epistemic_agents(formula, valid_agents, entry.id, errors)

            # After-reference check.
            for dep in entry.after:
                if dep not in valid_ids:
                    errors.append(
                        f"Goal {entry.id}: 'after' references "
                        f"non-existent goal ID {dep}"
                    )

        # Cycle check.
        if GoalSpec._detect_cycle(self.entries):
            errors.append("Cycle detected in goal dependency graph")

        return errors

    # -- Queries ------------------------------------------------------------

    def get_prerequisites(self, goal_id: int) -> Set[int]:
        """Return the set of prerequisite goal IDs for *goal_id*.

        Args:
            goal_id: The ID of the goal to query.

        Returns:
            Set of goal IDs that must be completed before *goal_id*.

        Raises:
            KeyError: If *goal_id* is not found in this spec.
        """
        for entry in self.entries:
            if entry.id == goal_id:
                return set(entry.after)
        raise KeyError(f"Goal ID {goal_id} not found in this GoalSpec")

    def get_entries_by_owner(self, owner: str) -> List[GoalEntry]:
        """Return all entries assigned to *owner*.

        Args:
            owner: The owner identifier to filter by (e.g. ``"agent_0"``).

        Returns:
            List of matching :class:`GoalEntry` instances, preserving order.
        """
        return [e for e in self.entries if e.owner == owner]

    def get_required_entries(self) -> List[GoalEntry]:
        """Return entries with no owner (shared/cooperative goals).

        These are the shared goals all agents must help achieve.

        Returns:
            List of :class:`GoalEntry` instances where ``owner is None``.
        """
        return [e for e in self.entries if e.owner is None]

    @property
    def has_epistemic_goals(self) -> bool:
        """True if any entry contains an epistemic operator (K or B)."""
        return any("(K " in e.pddl or "(B " in e.pddl for e in self.entries)

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _detect_cycle(entries: Tuple[GoalEntry, ...]) -> bool:
        """Check for cycles in the ``after`` dependency graph using DFS.

        Args:
            entries: The goal entries to check.

        Returns:
            ``True`` if a cycle exists, ``False`` otherwise.
        """
        # Build adjacency: goal -> set of goals it depends on (after).
        # We detect a cycle by looking for back-edges in a DFS.
        adjacency: Dict[int, Tuple[int, ...]] = {
            e.id: e.after for e in entries
        }
        valid_ids = set(adjacency.keys())

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[int, int] = {gid: WHITE for gid in valid_ids}

        def _dfs(node: int) -> bool:
            color[node] = GRAY
            for neighbor in adjacency.get(node, ()):
                if neighbor not in color:
                    # Reference to unknown node; skip (caught elsewhere).
                    continue
                if color[neighbor] == GRAY:
                    return True  # Back edge -> cycle.
                if color[neighbor] == WHITE and _dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for gid in valid_ids:
            if color[gid] == WHITE:
                if _dfs(gid):
                    return True
        return False

    # -- Dunder methods -----------------------------------------------------

    def __repr__(self) -> str:
        return f"GoalSpec(entries={self.entries!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GoalSpec):
            return NotImplemented
        return self.entries == other.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _check_epistemic_agents(
    formula: Formula,
    valid_agents: Set[str],
    goal_id: int,
    errors: List[str],
) -> None:
    """Recursively check that epistemic agent references are valid.

    Args:
        formula: The parsed formula tree to walk.
        valid_agents: Set of recognized agent identifiers.
        goal_id: The goal ID (for error messages).
        errors: Accumulator list for error strings.
    """
    if isinstance(formula, Knows):
        if formula.agent not in valid_agents:
            errors.append(
                f"Goal {goal_id}: K() references unknown agent "
                f"'{formula.agent}' (valid: {sorted(valid_agents)})"
            )
        _check_epistemic_agents(formula.inner, valid_agents, goal_id, errors)
    elif isinstance(formula, Believes):
        if formula.agent not in valid_agents:
            errors.append(
                f"Goal {goal_id}: B() references unknown agent "
                f"'{formula.agent}' (valid: {sorted(valid_agents)})"
            )
        _check_epistemic_agents(formula.inner, valid_agents, goal_id, errors)
    elif isinstance(formula, And):
        for op in formula.operands:
            _check_epistemic_agents(op, valid_agents, goal_id, errors)
    elif isinstance(formula, EpistemicFormula):
        # Catch-all for future epistemic types.
        if hasattr(formula, "inner") and formula.inner is not None:
            _check_epistemic_agents(
                formula.inner, valid_agents, goal_id, errors
            )
