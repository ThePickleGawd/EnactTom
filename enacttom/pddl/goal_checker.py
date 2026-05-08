"""
PDDL goal checker — replaces DAGProgress for runtime evaluation.

Evaluates PDDL goal formulas against simulator state with latching behavior
(once a conjunct is satisfied, it stays satisfied).  Or-branched goals use
live-state evaluation for the branch alternatives.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from enacttom.pddl.dsl import (
    Formula, Literal, And, Or, Not, Knows, Believes, EpistemicFormula,
    parse_goal_string,
)


class PDDLGoalChecker:
    """
    Evaluates PDDL goal formula against simulator state.

    Replaces DAGProgress with PDDL-native goal checking.

    For cooperative/mixed tasks:
        Conjuncts are latched: once completed, they stay completed.
        Supports ordering constraints via pddl_ordering.

    For Or-branched goals:
        Each Or operand is a separate branch. Branches use live-state
        evaluation; the goal is complete when any branch is fully satisfied.

    Runtime benchmark evaluation uses the functional (non-epistemic) projection
    of the task goal by default. Design-time PDDL verification still uses the
    full original goal elsewhere in the pipeline.
    """

    def __init__(
        self,
        goal: Formula,
        ordering: Optional[List[Dict[str, str]]] = None,
        owners: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            goal: Parsed PDDL goal formula
            ordering: List of {"before": "(pred ...)", "after": "(pred ...)"} constraints
            owners: Map from goal literal string to owner (for example, "agent_0")
                   Literals not in this map default to required (cooperative).
        """
        self.goal = goal

        # Detect Or-branching.
        self._is_or_goal = isinstance(goal, Or)

        if self._is_or_goal:
            # Each Or operand is a branch.
            # Flatten each branch's conjuncts separately.
            self._branches: List[List[Formula]] = []
            self._branch_ranges: List[Tuple[int, int]] = []  # (start, end) in self.conjuncts
            self.conjuncts: List[Formula] = []

            for operand in goal.operands:
                branch_conjuncts = operand.flatten()
                start = len(self.conjuncts)
                self.conjuncts.extend(branch_conjuncts)
                end = len(self.conjuncts)
                self._branches.append(branch_conjuncts)
                self._branch_ranges.append((start, end))
        else:
            self.conjuncts: List[Formula] = goal.flatten()
            self._branches = []
            self._branch_ranges = []

        self.completed: Set[int] = set()  # indices into self.conjuncts

        # Build ordering constraints: index -> set of prerequisite indices
        self._prerequisites: Dict[int, Set[int]] = {}
        if not self._is_or_goal:
            self._build_ordering(ordering or [])

        # Owner map: index -> owner string
        self._owners: Dict[int, str] = {}
        if owners:
            conjunct_strs = [c.to_pddl() for c in self.conjuncts]
            for literal_str, owner in owners.items():
                matched = False
                for idx, cs in enumerate(conjunct_strs):
                    if cs == literal_str:
                        self._owners[idx] = owner
                        matched = True
                        break
                if not matched:
                    # Supplementary goal (e.g. mixed-task personal objective
                    # not in the main :goal block).  Parse and append.
                    formula = parse_goal_string(literal_str)
                    new_idx = len(self.conjuncts)
                    self.conjuncts.append(formula)
                    self._owners[new_idx] = owner

    def _build_ordering(self, ordering: List[Dict[str, str]]) -> None:
        """Build prerequisite map from ordering constraints."""
        conjunct_strs = [c.to_pddl() for c in self.conjuncts]

        for constraint in ordering:
            before_str = constraint.get("before", "")
            after_str = constraint.get("after", "")

            before_idx = None
            after_idx = None
            for idx, cs in enumerate(conjunct_strs):
                if cs == before_str:
                    before_idx = idx
                if cs == after_str:
                    after_idx = idx

            if before_idx is not None and after_idx is not None:
                self._prerequisites.setdefault(after_idx, set()).add(before_idx)

    def _evaluate_conjunct(self, conjunct: Formula, check_predicate: Callable) -> bool:
        """Evaluate a single conjunct against runtime predicate checks."""
        return conjunct.evaluate(check_predicate)

    def update(self, check_predicate: Callable) -> Dict[str, Any]:
        """
        Check goal conjuncts against current state.

        For cooperative/mixed: latching (once completed, stays completed).
        For Or-goals: live-state evaluation (no latching).

        Args:
            check_predicate: Function(predicate_name, args_tuple) -> bool

        Returns:
            Dict with percent_complete, all_complete, newly_completed,
            and for Or-goals: winning_branch
        """
        if self._is_or_goal:
            return self._update_or_goal(check_predicate)
        return self._update_cooperative(check_predicate)

    def _update_cooperative(self, check_predicate: Callable) -> Dict[str, Any]:
        """Cooperative/mixed update with latching semantics."""
        newly_completed = []

        for idx, conjunct in enumerate(self.conjuncts):
            if idx in self.completed:
                continue

            # Check ordering prerequisites
            prereqs = self._prerequisites.get(idx, set())
            if not prereqs.issubset(self.completed):
                continue

            satisfied = self._evaluate_conjunct(conjunct, check_predicate)

            if satisfied:
                self.completed.add(idx)
                newly_completed.append(conjunct.to_pddl())

        total = len(self.conjuncts) if self.conjuncts else 1
        return {
            "completed": [self.conjuncts[i].to_pddl() for i in self.completed],
            "newly_completed": newly_completed,
            "percent_complete": len(self.completed) / total,
            "all_complete": len(self.completed) == len(self.conjuncts),
        }

    def _update_or_goal(self, check_predicate: Callable) -> Dict[str, Any]:
        """Or-goal update: live-state evaluation (no latching).

        Evaluates all conjuncts fresh each call, overwrites self.completed
        with current live state. Reports progress as best branch's progress.
        """
        prev_completed = set(self.completed)
        self.completed.clear()
        newly_completed = []

        # Evaluate all conjuncts against live state
        for idx, conjunct in enumerate(self.conjuncts):
            satisfied = self._evaluate_conjunct(conjunct, check_predicate)
            if satisfied:
                self.completed.add(idx)
                if idx not in prev_completed:
                    newly_completed.append(conjunct.to_pddl())

        # Find best branch progress and winning branch
        best_progress = 0.0
        winning_branch = None

        for branch_idx, (start, end) in enumerate(self._branch_ranges):
            branch_size = end - start
            if branch_size == 0:
                continue
            branch_done = sum(1 for i in range(start, end) if i in self.completed)
            progress = branch_done / branch_size
            best_progress = max(best_progress, progress)
            if branch_done == branch_size:
                winning_branch = branch_idx

        return {
            "completed": [self.conjuncts[i].to_pddl() for i in self.completed],
            "newly_completed": newly_completed,
            "percent_complete": best_progress,
            "all_complete": winning_branch is not None,
            "winning_branch": winning_branch,
        }

    # -------------------------------------------------------------------------
    # Branch-aware query methods
    # -------------------------------------------------------------------------

    @property
    def is_or_goal(self) -> bool:
        """True if this goal has Or-branching."""
        return self._is_or_goal

    @property
    def num_branches(self) -> int:
        """Number of Or branches (0 for cooperative goals)."""
        return len(self._branches)

    def get_branch_conjuncts(self, branch_idx: int) -> List[Formula]:
        """Get conjuncts for a specific Or branch."""
        if not self._is_or_goal or branch_idx >= len(self._branches):
            return []
        return list(self._branches[branch_idx])

    def get_branch_conjunct_indices(self, branch_idx: int) -> List[int]:
        """Get conjunct indices for a specific branch."""
        if not self._is_or_goal or branch_idx >= len(self._branch_ranges):
            return []
        start, end = self._branch_ranges[branch_idx]
        return list(range(start, end))

    def is_branch_complete(self, branch_idx: int) -> bool:
        """Check if all conjuncts in a branch are satisfied."""
        if not self._is_or_goal or branch_idx >= len(self._branch_ranges):
            return False
        start, end = self._branch_ranges[branch_idx]
        return all(i in self.completed for i in range(start, end))

    def get_branch_progress(self, branch_idx: int) -> float:
        """Get completion fraction for a specific branch."""
        if not self._is_or_goal or branch_idx >= len(self._branch_ranges):
            return 0.0
        start, end = self._branch_ranges[branch_idx]
        branch_size = end - start
        if branch_size == 0:
            return 1.0
        branch_done = sum(1 for i in range(start, end) if i in self.completed)
        return branch_done / branch_size

    # -------------------------------------------------------------------------
    # Owner queries
    # -------------------------------------------------------------------------

    def get_required_conjuncts(self) -> List[Formula]:
        """Get conjuncts that are required for task success (no owner or required=True)."""
        return [
            c for idx, c in enumerate(self.conjuncts)
            if idx not in self._owners
        ]

    def get_agent_conjuncts(self, agent_id: str) -> List[Formula]:
        """Get conjuncts owned by a specific agent."""
        return [
            c for idx, c in enumerate(self.conjuncts)
            if self._owners.get(idx) == agent_id
        ]

    def get_owner(self, idx: int) -> Optional[str]:
        """Get the owner of a specific conjunct index."""
        return self._owners.get(idx)

    def is_conjunct_completed(self, idx: int) -> bool:
        """Check if a specific conjunct index is completed."""
        return idx in self.completed

    def to_propositions(self) -> List[Dict[str, Any]]:
        """
        Convert PDDL goal conjuncts to evaluation.py proposition format.

        Returns list of {"entity": ..., "property": ..., "target": ..., "required": ...}
        Epistemic conjuncts extract the inner literal for proposition format.
        """
        props = []
        for idx, conjunct in enumerate(self.conjuncts):
            literal = conjunct
            # Unwrap epistemic layers to get the inner literal
            while isinstance(literal, EpistemicFormula):
                literal = literal.inner
            if isinstance(literal, Literal):
                prop = literal.to_proposition()
                # Add ownership/required info
                owner = self._owners.get(idx)
                if owner:
                    prop["required"] = owner
                else:
                    prop["required"] = True
                props.append(prop)
        return props

    def reset(self) -> None:
        """Reset all progress."""
        self.completed.clear()

    @classmethod
    def from_task_data(
        cls,
        task_data: Dict[str, Any],
        *,
        functional_only: bool = True,
    ) -> Optional["PDDLGoalChecker"]:
        """Create from raw task JSON data containing problem_pddl."""
        problem_pddl = task_data.get("problem_pddl")
        if not isinstance(problem_pddl, str) or not problem_pddl.strip():
            return None

        from enacttom.pddl.problem_pddl import parse_problem_pddl
        from enacttom.pddl.runtime_projection import project_runtime_from_parsed_problem

        parsed = parse_problem_pddl(problem_pddl)
        goal = parsed.goal_formula
        owners = parsed.owners or {}
        if functional_only:
            projection = project_runtime_from_parsed_problem(parsed)
            if projection.functional_goal is None:
                return None
            goal = projection.functional_goal
            owners = projection.functional_owners
        return cls(
            goal=goal,
            ordering=[],
            owners=owners,
        )
