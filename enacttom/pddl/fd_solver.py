"""
Fast Downward solver wrapping unified-planning.

Performs real state-space search for PDDL solvability, unlike PDKBSolver's
structural checks.

When epistemic goals (K/B) are present AND observability data is available,
uses epistemic compilation to verify both physical and epistemic solvability
in a single FD call. Otherwise falls back to stripping K/B and running FD
on the physical goal only.

Falls back to PDKBSolver when unified-planning is not installed.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from typing import Dict, List, Optional, Set, Tuple

from enacttom.pddl.dsl import (
    And,
    Believes,
    Domain,
    EpistemicFormula,
    Formula,
    Knows,
    Literal,
    Not,
    Or,
    Problem,
)
from enacttom.pddl.epistemic import ObservabilityModel
from enacttom.pddl.solver import PDKBSolver, SolverResult

logger = logging.getLogger(__name__)

try:
    from unified_planning.io import PDDLReader
    from unified_planning.shortcuts import OneshotPlanner, get_environment
    get_environment().credits_stream = None
    HAS_UP = True
except ImportError:
    HAS_UP = False

# NOTE: In some benchmark harness environments unified-planning may be partially
# available or misconfigured, leading to opaque parse errors. We treat the
# strict backend as unavailable unless explicitly enabled.
FORCE_DISABLE_UP = False



# ---------------------------------------------------------------------------
# Epistemic stripping
# ---------------------------------------------------------------------------

def _strip_epistemic(formula: Formula) -> Formula:
    """Unwrap K()/B() to get physical goal for classical planner."""
    if isinstance(formula, (Knows, Believes)):
        return _strip_epistemic(formula.inner)
    if isinstance(formula, And):
        return And(tuple(_strip_epistemic(op) for op in formula.operands))
    if isinstance(formula, Or):
        return Or(tuple(_strip_epistemic(op) for op in formula.operands))
    if isinstance(formula, Not) and formula.operand is not None:
        return Not(operand=_strip_epistemic(formula.operand))
    return formula  # Literal unchanged


def _has_epistemic_goals(formula: Formula) -> bool:
    """Check if a formula contains any K()/B() operators."""
    if isinstance(formula, (Knows, Believes)):
        return True
    if isinstance(formula, (And, Or)):
        return any(_has_epistemic_goals(op) for op in formula.operands)
    if isinstance(formula, Not) and formula.operand is not None:
        return _has_epistemic_goals(formula.operand)
    return False


def _collect_epistemic_leaf_literals(formula: Formula) -> List[Literal]:
    """Collect leaf literals that appear under at least one K()/B() wrapper."""
    literals: List[Literal] = []

    def _walk(node: Formula, inside_epistemic: bool) -> None:
        if isinstance(node, Literal):
            if inside_epistemic:
                literals.append(node)
            return
        if isinstance(node, (Knows, Believes)):
            _walk(node.inner, True)
            return
        if isinstance(node, (And, Or)):
            for op in node.operands:
                _walk(op, inside_epistemic)
            return
        if isinstance(node, Not) and node.operand is not None:
            _walk(node.operand, inside_epistemic)

    _walk(formula, False)
    return literals


def _epistemic_grounding_errors(
    problem: Problem,
    observability: Optional[ObservabilityModel],
) -> List[str]:
    """Return missing grounding needed for strict epistemic proof."""
    if observability is None or problem.goal is None:
        return ["observability model is unavailable"]

    grounded_object_rooms = observability.object_rooms or {}
    agent_room_grounding = {
        lit.args[0]
        for lit in problem.init
        if lit.predicate == "agent_in_room" and len(lit.args) == 2 and not lit.negated
    }

    errors: List[str] = []
    seen: Set[str] = set()
    for literal in _collect_epistemic_leaf_literals(problem.goal):
        for arg in literal.args:
            if arg.startswith("?"):
                continue
            obj_type = problem.objects.get(arg)
            if obj_type is None:
                key = f"undeclared:{arg}"
                if key not in seen:
                    errors.append(f"{literal.to_pddl()} references undeclared object '{arg}'")
                    seen.add(key)
                continue
            if obj_type == "room":
                continue
            if obj_type == "agent":
                if arg not in agent_room_grounding:
                    key = f"agent:{arg}"
                    if key not in seen:
                        errors.append(
                            f"{literal.to_pddl()} requires `(agent_in_room {arg} <room>)` in :init"
                        )
                        seen.add(key)
                continue
            if arg not in grounded_object_rooms:
                key = f"object:{arg}"
                if key not in seen:
                    errors.append(
                        f"{literal.to_pddl()} requires `(is_in_room {arg} <room>)` in :init"
                    )
                    seen.add(key)
    return errors


def _diagnose_unsolvable(problem: Problem, physical_goal: Formula) -> List[str]:
    """Diagnose why a physical goal is unsolvable by checking init completeness.

    Returns a list of actionable hints about missing init facts.
    """
    hints: List[str] = []
    init_preds: Dict[str, Set[Tuple[str, ...]]] = {}
    for lit in problem.init:
        if not lit.negated:
            init_preds.setdefault(lit.predicate, set()).add(lit.args)

    declared_agents = sorted(
        name for name, typ in problem.objects.items() if typ == "agent"
    )
    declared_rooms = {
        name for name, typ in problem.objects.items() if typ == "room"
    }

    # Check agent grounding
    grounded_agents = {
        args[0]
        for args in init_preds.get("agent_in_room", set())
        if len(args) == 2
    }
    for agent in declared_agents:
        if agent not in grounded_agents:
            hints.append(
                f"Missing agent position: add (agent_in_room {agent} <room>) to :init"
            )

    # Check object/furniture room grounding
    grounded_objects = {
        args[0]
        for args in init_preds.get("is_in_room", set())
        if len(args) == 2
    }

    # Collect objects referenced in the goal
    goal_refs: Set[str] = set()

    def _collect_refs(f: Formula) -> None:
        if isinstance(f, Literal):
            for arg in f.args:
                if not arg.startswith("?"):
                    goal_refs.add(arg)
        elif isinstance(f, (And, Or)):
            for op in f.operands:
                _collect_refs(op)
        elif isinstance(f, Not) and f.operand is not None:
            _collect_refs(f.operand)

    _collect_refs(physical_goal)

    for obj_id in sorted(goal_refs):
        obj_type = problem.objects.get(obj_id)
        if obj_type in ("agent", "room") or obj_type is None:
            continue
        if obj_id not in grounded_objects:
            hints.append(
                f"Missing room grounding: add (is_in_room {obj_id} <room>) to :init"
            )

    # Check if goal references predicates with no matching init facts
    def _check_goal_predicates(f: Formula) -> None:
        if isinstance(f, Literal):
            if not f.negated:
                # For positive goal literals, check if there's any init support
                pred = f.predicate
                if pred in ("is_on_top", "is_on_floor", "is_inside"):
                    # Object placement — check if object has a current position
                    if len(f.args) >= 1:
                        obj = f.args[0]
                        has_pos = any(
                            obj in (a[0] if len(a) >= 1 else "")
                            for p in ("is_on_top", "is_on_floor", "is_inside")
                            for a in init_preds.get(p, set())
                        )
                        if not has_pos and obj in problem.objects:
                            hints.append(
                                f"Object '{obj}' has no initial position in :init. "
                                f"Add (is_on_top {obj} <furniture>) or similar."
                            )
        elif isinstance(f, (And, Or)):
            for op in f.operands:
                _check_goal_predicates(op)
        elif isinstance(f, Not) and f.operand is not None:
            _check_goal_predicates(f.operand)

    _check_goal_predicates(physical_goal)

    return hints


def _deduplicate_conjuncts(formula: Formula) -> Formula:
    """Remove duplicate conjuncts from an And formula."""
    if not isinstance(formula, And):
        return formula
    seen: Set[str] = set()
    unique = []
    for op in formula.operands:
        key = op.to_pddl()
        if key not in seen:
            seen.add(key)
            unique.append(op)
    if len(unique) == 1:
        return unique[0]
    return And(tuple(unique))


# ---------------------------------------------------------------------------
# PDDL serialization for unified-planning
# ---------------------------------------------------------------------------

def _problem_to_pddl(problem: Problem, domain_pddl: str) -> str:
    """Serialize a Problem to a PDDL problem string for classical planning.

    Strips :goal-owners section if present (not standard PDDL).
    """
    obj_lines = []
    by_type: Dict[str, List[str]] = {}
    for name, typ in problem.objects.items():
        by_type.setdefault(typ, []).append(name)
    for typ, names in sorted(by_type.items()):
        obj_lines.append(f"    {' '.join(sorted(names))} - {typ}")
    objects_str = "\n".join(obj_lines)

    init_str = "\n    ".join(l.to_pddl() for l in problem.init if not l.negated)

    goal_str = problem.goal.to_pddl() if problem.goal else "()"

    return (
        f"(define (problem {problem.name})\n"
        f"  (:domain {problem.domain_name})\n"
        f"  (:objects\n{objects_str}\n  )\n"
        f"  (:init\n    {init_str}\n  )\n"
        f"  (:goal {goal_str})\n"
        f")"
    )


def _strip_goal_owners_from_pddl(pddl_str: str) -> str:
    """Remove (:goal-owners ...) section from a PDDL string."""
    return re.sub(r'\(\s*:goal-owners\s.*?\)\s*\)', '', pddl_str, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# FastDownwardSolver
# ---------------------------------------------------------------------------

def _validate_pddl_grounding(domain_pddl: str, problem_pddl: str) -> List[str]:
    """Check that all ground names in domain actions are declared.

    Domain-level actions with :parameters () (0-ary, like epistemic
    observe/inform actions) can only reference :constants, not problem
    :objects. This pre-check catches the issue before unified-planning
    crashes with an opaque "does not correspond" error.

    Returns list of error strings (empty = valid).
    """
    errors: List[str] = []

    # Collect declared :constants from domain
    const_names: Set[str] = set()
    const_match = re.search(r'\(:constants\s(.*?)\)', domain_pddl, re.DOTALL)
    if const_match:
        # Parse "name1 name2 - type" lines
        for token in re.findall(r'(\b[a-zA-Z_]\w*)\b', const_match.group(1)):
            if token not in ("agent", "room", "furniture", "object"):
                const_names.add(token)

    # Collect declared :objects from problem
    obj_names: Set[str] = set()
    obj_match = re.search(r'\(:objects\s(.*?)\)', problem_pddl, re.DOTALL)
    if obj_match:
        for token in re.findall(r'(\b[a-zA-Z_]\w*)\b', obj_match.group(1)):
            if token not in ("agent", "room", "furniture", "object"):
                obj_names.add(token)

    all_declared = const_names | obj_names

    # Collect predicate names from domain (so we don't flag them as missing objects)
    pred_names: Set[str] = set()
    pred_match = re.search(r'\(:predicates\s(.*?)\)\s*(?:\(:)', domain_pddl, re.DOTALL)
    if pred_match:
        pred_names = set(re.findall(r'\((\w+)', pred_match.group(1)))

    # Find 0-ary actions (no parameters) — these are the epistemic compiler's actions
    zero_ary_actions = re.findall(
        r'\(:action\s+(\w+)\s+:parameters\s*\(\s*\)\s+:precondition\s+(.*?)\s+:effect\s+(.*?)\)',
        domain_pddl,
        re.DOTALL,
    )

    for action_name, precond, effect in zero_ary_actions:
        # Extract all identifiers from precondition and effect
        for section in (precond, effect):
            tokens = re.findall(r'\b([a-zA-Z_]\w*)\b', section)
            for token in tokens:
                # Skip PDDL keywords, predicate names, and action-internal names
                if token in ("and", "or", "not", "when", "forall"):
                    continue
                if token in pred_names:
                    continue
                # Skip 0-ary predicates (knows_*, msg_tok_*)
                if token.startswith(("knows_", "msg_tok_")):
                    continue
                if token not in all_declared:
                    errors.append(
                        f"Action '{action_name}' references undeclared name "
                        f"'{token}' (not in :constants or :objects)"
                    )

    # Deduplicate
    return list(dict.fromkeys(errors))


class FastDownwardSolver:
    """
    Real state-space PDDL solver using Fast Downward via unified-planning.

    For epistemic goals: strips K()/B() to get a physical goal, solves that
    with a classical planner, then validates epistemic requirements separately.

    Falls back to PDKBSolver if unified-planning is not installed.
    """

    def __init__(self):
        self._fallback = PDKBSolver()

    def solve(
        self,
        domain: Domain,
        problem: Problem,
        observability: Optional[ObservabilityModel] = None,
        max_belief_depth: int = 3,
        timeout: float = 30.0,
        strict: bool = False,
    ) -> SolverResult:
        """
        Solve a PDDL problem using Fast Downward.

        When epistemic goals are present and observability data is available,
        uses epistemic compilation to verify both physical and epistemic
        solvability in a single FD call. Otherwise strips K/B and checks
        the physical goal only.

        Args:
            domain: The PDDL domain.
            problem: The PDDL problem instance.
            observability: Epistemic observability model.
            max_belief_depth: Max epistemic nesting depth.
            timeout: Planner timeout in seconds.

        Returns:
            SolverResult with solvability, plan, and belief depth.
        """
        if (not HAS_UP) or FORCE_DISABLE_UP:
            if strict:
                return SolverResult(
                    solvable=False,
                    solve_time=0.0,
                    error=(
                        "Strict proof backend unavailable: unified-planning / "
                        "Fast Downward is not installed"
                    ),
                )
            logger.warning(
                "unified-planning not installed; falling back to PDKBSolver "
                "(structural checks only, no real state-space search). "
                "Install with: pip install unified-planning up-fast-downward"
            )
            return self._fallback.solve(domain, problem, observability, max_belief_depth)

        start = time.time()

        if not problem.goal:
            return SolverResult(
                solvable=True,
                belief_depth=0,
                solve_time=time.time() - start,
            )

        # Epistemic compilation path: unified FD call for physical + epistemic
        has_epistemic = _has_epistemic_goals(problem.goal)
        grounding_errors = _epistemic_grounding_errors(problem, observability) if has_epistemic else []
        if has_epistemic and strict and grounding_errors:
            error_lines = [
                "Strict epistemic proof requires explicit observability grounding.",
                "Missing facts in :init:",
            ]
            for err in grounding_errors:
                error_lines.append(f"  - {err}")
            error_lines.append(
                "Fix: add the missing (agent_in_room ...) and (is_in_room ...) "
                "facts to the problem_pddl :init section, or provide scene data."
            )
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                error="\n".join(error_lines),
            )
        if has_epistemic and observability and not grounding_errors:
            return self._solve_epistemic(
                domain, problem, observability, max_belief_depth, timeout, start, strict
            )

        # Non-epistemic or no observability data: strip K/B and solve physical
        return self._solve_physical(
            domain, problem, observability, max_belief_depth, timeout, start, strict
        )

    def _solve_epistemic(
        self,
        domain: Domain,
        problem: Problem,
        observability: ObservabilityModel,
        max_belief_depth: int,
        timeout: float,
        start: float,
        strict: bool,
    ) -> SolverResult:
        """Epistemic compilation path — single FD call for physical + epistemic."""
        from enacttom.pddl.epistemic_compiler import compile_epistemic

        try:
            compilation = compile_epistemic(
                problem.goal, domain, problem, observability
            )
        except Exception as e:
            logger.error("Epistemic compilation error: %s", e)
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                error=f"Epistemic compilation error: {e}",
            )

        if compilation.belief_depth > max_belief_depth:
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                belief_depth=compilation.belief_depth,
                error=(
                    f"Task requires belief depth {compilation.belief_depth}, "
                    f"which exceeds allowed depth {max_belief_depth}"
                ),
                trivial_k_goals=compilation.trivial_k_goals,
            )

        try:
            result = self._run_planner(
                compilation.domain_pddl, compilation.problem_pddl, timeout
            )
        except Exception as e:
            err_str = str(e)
            if "expression false" in err_str.lower():
                # Contradictory goal (e.g. physical (not X) + epistemic K(a, X))
                # Do NOT fall back — the problem is genuinely unsolvable.
                logger.error(
                    "Fast Downward detected contradictory goal (expression false): %s", e
                )
                return SolverResult(
                    solvable=False,
                    solve_time=time.time() - start,
                    error=(
                        "Contradictory goal: epistemic K() fact conflicts with "
                        "a negated physical goal. Ensure K() inner facts are "
                        "consistent with physical goal literals."
                    ),
                )
            logger.error(
                "Fast Downward planner error (epistemic): %s", e
            )
            if strict:
                return SolverResult(
                    solvable=False,
                    solve_time=time.time() - start,
                    error=f"Strict proof backend failed: {e}",
                )
            return self._fallback.solve(
                domain, problem, observability, max_belief_depth=3
            )

        if not result["solvable"]:
            base_error = result.get(
                "error",
                "No plan found — goal is unreachable "
                "(physical or epistemic requirements unsatisfiable)",
            )
            hints = _diagnose_unsolvable(problem, _strip_epistemic(problem.goal))
            if hints:
                base_error += "\nActionable fixes:\n" + "\n".join(f"  - {h}" for h in hints)
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                error=base_error,
            )

        return SolverResult(
            solvable=True,
            plan=result.get("plan"),
            belief_depth=compilation.belief_depth,
            solve_time=time.time() - start,
            trivial_k_goals=compilation.trivial_k_goals,
        )

    def _solve_physical(
        self,
        domain: Domain,
        problem: Problem,
        observability: Optional[ObservabilityModel],
        max_belief_depth: int,
        timeout: float,
        start: float,
        strict: bool,
    ) -> SolverResult:
        """Strip K/B and solve the physical goal only (fallback path)."""
        physical_goal = _strip_epistemic(problem.goal)
        physical_goal = _deduplicate_conjuncts(physical_goal)

        planning_problem = Problem(
            name=problem.name,
            domain_name=problem.domain_name,
            objects=problem.objects,
            init=problem.init,
            goal=physical_goal,
        )

        domain_pddl = domain.to_planning_pddl()
        problem_pddl = _problem_to_pddl(planning_problem, domain_pddl)

        try:
            result = self._run_planner(domain_pddl, problem_pddl, timeout)
        except Exception as e:
            logger.error("Fast Downward planner error: %s", e)
            if strict:
                return SolverResult(
                    solvable=False,
                    solve_time=time.time() - start,
                    error=f"Strict proof backend failed: {e}",
                )
            return self._fallback.solve(
                domain, problem, observability, max_belief_depth
            )

        if not result["solvable"]:
            base_error = result.get("error", "No plan found — goal is unreachable from init state")
            hints = _diagnose_unsolvable(problem, physical_goal)
            if hints:
                base_error += "\nActionable fixes:\n" + "\n".join(f"  - {h}" for h in hints)
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                error=base_error,
            )

        # Run epistemic checks (belief depth, trivial K, budget)
        epistemic_result = self._fallback._compute_min_belief_depth(
            problem, observability, max_belief_depth
        )
        belief_depth, trivial_goals = epistemic_result
        if belief_depth > max_belief_depth:
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                belief_depth=belief_depth,
                error=(
                    f"Task requires belief depth {belief_depth}, "
                    f"which exceeds allowed depth {max_belief_depth}"
                ),
                trivial_k_goals=trivial_goals,
            )

        return SolverResult(
            solvable=True,
            plan=result.get("plan"),
            belief_depth=belief_depth,
            solve_time=time.time() - start,
            trivial_k_goals=trivial_goals,
        )

    def _run_planner(
        self,
        domain_pddl: str,
        problem_pddl: str,
        timeout: float,
    ) -> dict:
        """Run Fast Downward via unified-planning.

        Returns dict with 'solvable' (bool), 'plan' (list of str), 'error' (str).
        """
        # Pre-validate: check that domain actions don't reference undeclared names.
        grounding_errors = _validate_pddl_grounding(domain_pddl, problem_pddl)
        if grounding_errors:
            raise RuntimeError(
                "PDDL grounding error (names in domain actions not declared "
                f"as :constants or :objects): {'; '.join(grounding_errors)}"
            )

        reader = PDDLReader()

        # Write temp files for PDDLReader
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pddl', prefix='fd_domain_', delete=False
        ) as df:
            df.write(domain_pddl)
            domain_path = df.name

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pddl', prefix='fd_problem_', delete=False
        ) as pf:
            pf.write(problem_pddl)
            problem_path = pf.name

        try:
            up_problem = reader.parse_problem(domain_path, problem_path)
        except Exception as e:
            # Dump PDDL for debugging before deleting temp files
            logger.debug("PDDL parse failure — problem PDDL:\n%s", problem_pddl[:500])
            logger.debug("PDDL parse failure — domain PDDL (first 200 chars):\n%s", domain_pddl[:200])
            os.unlink(domain_path)
            os.unlink(problem_path)
            raise RuntimeError(f"PDDL parse error: {e}") from e

        os.unlink(domain_path)
        os.unlink(problem_path)

        # Fast Downward writes auxiliary files like output.sas in the process
        # cwd by default. Isolate each solve so concurrent workers do not race.
        with tempfile.TemporaryDirectory(prefix="fd_run_") as planner_cwd:
            old_cwd = os.getcwd()
            try:
                os.chdir(planner_cwd)
                with OneshotPlanner(name="fast-downward") as planner:
                    up_result = planner.solve(up_problem, timeout=timeout)
            finally:
                os.chdir(old_cwd)

        if up_result.status in (
            up_result.status.__class__.SOLVED_SATISFICING,
            up_result.status.__class__.SOLVED_OPTIMALLY,
        ):
            plan_steps = []
            if up_result.plan:
                for action in up_result.plan.actions:
                    plan_steps.append(str(action))
            return {"solvable": True, "plan": plan_steps}

        return {
            "solvable": False,
            "error": f"Planner status: {up_result.status.name}",
        }

    def check_communication_budget(
        self,
        problem: Problem,
        observability: ObservabilityModel,
    ) -> Optional[str]:
        """Delegate to PDKBSolver's budget check."""
        return self._fallback.check_communication_budget(problem, observability)
