"""
PDDL solver wrapper.

Provides an interface to solve epistemic PDDL problems.
Currently implements a lightweight depth-bounded search suitable for
EnactToM's bounded belief depth (max 3).

For production use with larger state spaces, this can be extended to
call PDKB (QuMuLab/pdkb-planning) or Fast Downward as external solvers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from enacttom.pddl.dsl import Domain, Problem, Literal, And, Or, Not, Knows, Believes, EpistemicFormula, Formula, collect_leaf_literals
from enacttom.pddl.epistemic import ObservabilityModel


@dataclass
class SolverResult:
    """Result of solving a PDDL problem."""
    solvable: bool
    plan: Optional[List[str]] = None
    belief_depth: int = 0
    solve_time: float = 0.0
    error: Optional[str] = None
    trivial_k_goals: List[str] = field(default_factory=list)


class PDKBSolver:
    """
    Lightweight epistemic PDDL solver for EnactToM tasks.

    Uses a conservative approach: checks if the goal is achievable
    given the domain actions and observability constraints, without
    full state-space search.

    For EnactToM's purposes, we need to verify:
    1. All goal predicates reference valid objects
    2. There exists a sequence of actions that achieves each goal conjunct
    3. Observability constraints don't make the problem unsolvable
    """

    def solve(
        self,
        domain: Domain,
        problem: Problem,
        observability: Optional[ObservabilityModel] = None,
        max_belief_depth: int = 3,
    ) -> SolverResult:
        """
        Check if the problem is solvable at the given belief depth.

        This is a structural solvability check, not a full planner.
        It verifies that:
        - All goal predicates are achievable by domain actions
        - Object references in goals are valid
        - Observability constraints don't create impossible requirements
        """
        start = time.time()

        if not problem.goal:
            return SolverResult(
                solvable=True,
                belief_depth=0,
                solve_time=time.time() - start,
            )

        # Extract leaf literals from goal (unwrap K/B/Or/Not wrappers)
        goal_literals = collect_leaf_literals(problem.goal)

        # Check 1: All goal predicates must be achievable
        domain_predicate_names = {p.name for p in domain.predicates}

        for literal in goal_literals:
            if literal.predicate not in domain_predicate_names:
                return SolverResult(
                    solvable=False,
                    belief_depth=0,
                    solve_time=time.time() - start,
                    error=f"Goal predicate '{literal.predicate}' not in domain",
                )

        # Check 2: All object references in goals must exist in problem
        valid_objects = set(problem.objects.keys())
        for literal in goal_literals:
            for arg in literal.args:
                if arg.startswith("?"):
                    continue  # Variable, not ground
                if arg not in valid_objects:
                    return SolverResult(
                        solvable=False,
                        belief_depth=0,
                        solve_time=time.time() - start,
                        error=f"Goal references unknown object '{arg}'",
                    )

        # Check 3: For each goal literal, verify it is reachable.
        # Dynamic predicates (with positive effects) are considered potentially
        # achievable. Static predicates must already hold exactly in init.
        positive_effect_predicates = set()
        for action in domain.actions:
            for effect in action.effects:
                if hasattr(effect, 'literal'):
                    if not effect.literal.negated:
                        positive_effect_predicates.add(effect.literal.predicate)
                elif hasattr(effect, 'effect'):
                    # ForallEffect: the .effect field is the positive effect
                    positive_effect_predicates.add(effect.effect.predicate)

        init_positive_literals = {
            (l.predicate, l.args)
            for l in problem.init
            if not l.negated
        }

        for literal in goal_literals:
            lit_key = (literal.predicate, literal.args)
            if literal.predicate in positive_effect_predicates:
                # Dynamic predicate; conservative structural check only.
                continue

            if literal.negated:
                # Static negated literal is impossible only if opposite literal
                # is fixed true in init.
                if lit_key in init_positive_literals:
                    return SolverResult(
                        solvable=False,
                        belief_depth=0,
                        solve_time=time.time() - start,
                        error=(
                            "Static goal literal is unsatisfiable: "
                            f"not {literal.predicate}{literal.args} but "
                            "the positive literal is fixed in init"
                        ),
                    )
                continue

            if lit_key not in init_positive_literals:
                return SolverResult(
                    solvable=False,
                    belief_depth=0,
                    solve_time=time.time() - start,
                    error=(
                        f"No action can achieve literal "
                        f"'{literal.predicate}{literal.args}'"
                    ),
                )

        # Check 4: Epistemic requirements
        belief_depth, trivial_goals = self._compute_min_belief_depth(
            problem, observability, max_belief_depth
        )
        if belief_depth > max_belief_depth:
            return SolverResult(
                solvable=False,
                belief_depth=belief_depth,
                solve_time=time.time() - start,
                error=(
                    f"Task requires belief depth {belief_depth}, "
                    f"which exceeds allowed depth {max_belief_depth}"
                ),
                trivial_k_goals=trivial_goals,
            )

        return SolverResult(
            solvable=True,
            belief_depth=belief_depth,
            solve_time=time.time() - start,
            trivial_k_goals=trivial_goals,
        )

    def _compute_min_belief_depth(
        self,
        problem: Problem,
        observability: Optional[ObservabilityModel],
        max_depth: int,
    ) -> Tuple[int, List[str]]:
        """
        Compute minimum belief depth needed.

        Returns (depth, trivial_k_goals) where trivial_k_goals lists
        K()/B() goals that are trivially satisfied because the agent
        can directly observe the fact.

        Depth 0: No epistemic reasoning needed (all agents see everything)
        Depth 1: Agents must reason about others' knowledge
        Depth 2: Agents must reason about what others think they know
        Depth 3: Third-order nesting
        """
        trivial_goals: List[str] = []

        if not observability or not observability.has_information_asymmetry():
            return 0, trivial_goals

        # Count syntactic nesting depth of epistemic formulas in the goal
        syntactic_depth = _max_epistemic_depth(problem.goal) if problem.goal else 0
        if syntactic_depth == 0:
            return 0, trivial_goals

        # If scene data is available (object_rooms populated), do semantic check
        if syntactic_depth > 0 and observability.object_rooms:
            semantic_depth, trivial_goals = _compute_non_trivial_depth(
                problem.goal, observability
            )
            return min(semantic_depth, max_depth), trivial_goals

        return min(syntactic_depth, max_depth), trivial_goals

    def check_communication_budget(
        self,
        problem: Problem,
        observability: ObservabilityModel,
    ) -> Optional[str]:
        """
        Check if message limits can support K() goal requirements.

        Returns None if OK, or a warning string if budget is insufficient.
        """
        if not observability.message_limits:
            return None  # No limits, no problem

        if not problem.goal:
            return None

        # Collect non-trivial K() goals that require communication
        required_transfers: Dict[str, int] = {}  # receiver_agent -> count of K() goals needing info

        for conjunct in _collect_epistemic_goals(problem.goal):
            receiver = conjunct.agent
            inner = conjunct.inner

            if not isinstance(inner, Literal):
                continue

            # Check if receiver can directly observe this fact
            if observability.is_fact_observable_by(receiver, inner.predicate, inner.args):
                continue  # Trivial — no communication needed

            # This K() goal requires someone to communicate to receiver
            required_transfers[receiver] = required_transfers.get(receiver, 0) + 1

        if not required_transfers:
            return None

        # For each receiver that needs info, check if any informer has budget
        # Conservative: assume each K() goal needs at least 1 message from some sender
        warnings = []
        all_agents = set()
        for agent in observability.message_limits:
            all_agents.add(agent)
        for agent in observability.restricted_rooms:
            all_agents.add(agent)

        for receiver, needed_facts in required_transfers.items():
            # Find potential informers (agents who CAN observe the facts)
            # and check their combined message budget
            total_budget = 0
            for agent in all_agents:
                if agent == receiver:
                    continue
                agent_targets = observability.message_targets.get(agent)
                if agent_targets is not None and receiver not in agent_targets:
                    continue
                limit = observability.message_limits.get(agent)
                if limit is not None:
                    total_budget += limit
                else:
                    total_budget = needed_facts  # Unlimited sender available
                    break

            if total_budget < needed_facts:
                warnings.append(
                    f"{receiver} needs {needed_facts} K() fact(s) communicated "
                    f"but available senders have combined budget of {total_budget} message(s)"
                )

        if warnings:
            return "Communication budget may be insufficient: " + "; ".join(warnings)
        return None


def _collect_epistemic_goals(formula: Optional[Formula]) -> List:
    """Recursively collect all Knows/Believes nodes from a formula tree."""
    if formula is None:
        return []
    if isinstance(formula, (Knows, Believes)):
        return [formula]
    if isinstance(formula, (And, Or)):
        result = []
        for op in formula.operands:
            result.extend(_collect_epistemic_goals(op))
        return result
    if isinstance(formula, Not) and formula.operand is not None:
        return _collect_epistemic_goals(formula.operand)
    return []


def _max_epistemic_depth(formula: Optional[Formula]) -> int:
    """Compute the maximum nesting depth of epistemic operators in a formula."""
    if formula is None:
        return 0
    if isinstance(formula, (Knows, Believes)):
        return 1 + _max_epistemic_depth(formula.inner)
    if isinstance(formula, (And, Or)):
        return max((_max_epistemic_depth(op) for op in formula.operands), default=0)
    if hasattr(formula, 'operand'):
        return _max_epistemic_depth(formula.operand)
    return 0


def _compute_non_trivial_depth(
    formula: Optional[Formula],
    observability: ObservabilityModel,
) -> Tuple[int, List[str]]:
    """
    Compute epistemic depth considering only non-trivial K()/B() goals.

    A K(agent, phi) is trivial if agent can directly observe all entities
    in phi. Only non-trivial epistemic operators contribute to depth.

    Returns (depth, list of trivial goal PDDL strings).
    """
    if formula is None:
        return 0, []

    trivial_goals: List[str] = []

    if isinstance(formula, (Knows, Believes)):
        agent = formula.agent
        inner = formula.inner

        if isinstance(inner, Literal) and observability.is_k_goal_trivial(agent, inner):
            trivial_goals.append(formula.to_pddl())
            # Trivial depth-1 K() — doesn't add depth, but still recurse inner
            inner_depth, inner_trivial = _compute_non_trivial_depth(inner, observability)
            trivial_goals.extend(inner_trivial)
            return inner_depth, trivial_goals

        # Non-trivial K() — adds 1 to depth
        inner_depth, inner_trivial = _compute_non_trivial_depth(inner, observability)
        trivial_goals.extend(inner_trivial)
        return 1 + inner_depth, trivial_goals

    if isinstance(formula, (And, Or)):
        max_depth = 0
        for op in formula.operands:
            d, t = _compute_non_trivial_depth(op, observability)
            trivial_goals.extend(t)
            max_depth = max(max_depth, d)
        return max_depth, trivial_goals

    if isinstance(formula, Not) and formula.operand is not None:
        return _compute_non_trivial_depth(formula.operand, observability)

    return 0, trivial_goals
