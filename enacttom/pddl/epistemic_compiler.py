"""
Epistemic compilation for classical PDDL solvers.

Compiles K()/B() epistemic goals into classical PDDL predicates and
grounded actions so that a single Fast Downward call verifies both
physical and epistemic solvability.

For each unique leaf fact referenced in K() goals, builds a knowledge
propagation network across all agents:
  - ``knows_<agent>_<hash>`` 0-ary predicates for EVERY agent
  - Observe actions for agents who can directly see the fact
  - Inform actions for every (sender, receiver) pair with can_communicate,
    preconditioned on sender's knowledge predicate + budget token

FD discovers relay chains automatically:
  observe → knows_a0 → inform → knows_a1 → relay → knows_a2

Nested K: K(a0, K(a2, phi)) adds a second-layer knowledge predicate for a0
knowing that a2 knows, achievable via communication.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KGoalNode:
    """A single K()/B() goal extracted from the formula tree."""
    agent: str              # who needs to know
    inner: Formula          # what they need to know (may itself be K/B)
    fact_id: str            # deterministic predicate name (md5-based)
    depth: int              # nesting level (1, 2, 3)
    trivial: bool           # agent can directly observe all entities
    inner_k_fact_id: Optional[str] = None  # for nested K: the inner K's fact_id


@dataclass
class EpistemicCompilation:
    """Result of compiling epistemic goals into classical PDDL."""
    domain_pddl: str        # augmented domain string
    problem_pddl: str       # augmented problem string
    classical_goal: Formula  # goal with K/B replaced by knowledge predicates
    belief_depth: int        # max non-trivial nesting depth
    trivial_k_goals: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_epistemic(
    goal: Formula,
    domain: Domain,
    problem: Problem,
    observability: ObservabilityModel,
) -> EpistemicCompilation:
    """
    Compile epistemic goals into classical PDDL.

    Transforms K()/B() goals into 0-ary knowledge predicates with
    observe/inform actions, producing an augmented domain+problem that
    a classical planner can solve.

    Args:
        goal: The original goal formula (may contain K/B).
        domain: The PDDL domain.
        problem: The PDDL problem (objects, init, goal).
        observability: Scene-derived observability model.

    Returns:
        EpistemicCompilation with augmented domain/problem PDDL strings.
    """
    # Collect all K/B nodes from the goal
    k_goals = _collect_k_goals(goal, observability)

    if not k_goals:
        # No epistemic goals — return domain/problem as-is
        domain_pddl = domain.to_planning_pddl()
        problem_pddl = _problem_to_classical_pddl(problem)
        return EpistemicCompilation(
            domain_pddl=domain_pddl,
            problem_pddl=problem_pddl,
            classical_goal=goal,
            belief_depth=0,
        )

    # Build the knowledge predicate map: original K/B pddl -> KGoalNode
    k_goal_map: Dict[str, KGoalNode] = {}
    for kg in k_goals:
        key = _k_goal_key(kg)
        k_goal_map[key] = kg

    # All agents in the problem
    all_agents = sorted(
        name for name, typ in problem.objects.items() if typ == "agent"
    )

    # Communication graph from init
    can_comm: Set[Tuple[str, str]] = set()
    for lit in problem.init:
        if lit.predicate == "can_communicate" and len(lit.args) == 2:
            can_comm.add((lit.args[0], lit.args[1]))

    # Compute belief depth (max non-trivial depth)
    belief_depth = max(
        (kg.depth for kg in k_goals if not kg.trivial),
        default=0,
    )

    # Collect trivial K goal pddl strings
    trivial_k_goals = [
        _k_goal_pddl(kg) for kg in k_goals if kg.trivial
    ]

    # --- Build knowledge propagation network ---

    # Collect unique leaf facts (the physical formulas inside K/B goals)
    # and the fact_hash used by each K-goal targeting that leaf.
    leaf_facts = _collect_leaf_facts(k_goals)

    # Build predicates: knows_<agent>_<fact_hash> for ALL agents per fact
    extra_preds = _build_knowledge_predicates_network(
        k_goals, leaf_facts, all_agents
    )

    # Build observe actions: for each fact, any agent who can observe it
    extra_actions = _build_observe_actions_network(
        leaf_facts, all_agents, observability
    )

    # Build inform actions: for every (sender, receiver) with can_communicate
    extra_actions.extend(
        _build_inform_actions_network(
            leaf_facts, k_goals, all_agents, can_comm, observability
        )
    )

    # Build nested K actions (communication for outer layer)
    extra_actions.extend(
        _build_nested_k_actions(k_goals, all_agents, can_comm, observability)
    )

    # Build budget tokens for init
    budget_tokens = _build_budget_tokens(observability)

    # Build extra predicates for budget tokens
    budget_preds = _build_budget_predicates(observability)
    extra_preds.extend(budget_preds)

    # Replace K/B in goal with classical predicates
    classical_goal = _replace_epistemic_in_goal(goal, k_goal_map)

    # Collect grounded objects referenced in extra actions — these must be
    # declared as :constants in the domain for unified-planning's PDDLReader.
    action_constants = _collect_action_constants(extra_actions, problem)

    # Build augmented domain PDDL
    base_domain_pddl = domain.to_planning_pddl()
    augmented_domain = _build_augmented_domain_pddl(
        base_domain_pddl, extra_preds, extra_actions, action_constants
    )

    # Build augmented problem PDDL (exclude constants from :objects)
    augmented_problem = _build_augmented_problem_pddl(
        problem, classical_goal, budget_tokens, action_constants
    )

    return EpistemicCompilation(
        domain_pddl=augmented_domain,
        problem_pddl=augmented_problem,
        classical_goal=classical_goal,
        belief_depth=belief_depth,
        trivial_k_goals=trivial_k_goals,
    )


# ---------------------------------------------------------------------------
# K/B goal collection
# ---------------------------------------------------------------------------

def _fact_id(agent: str, formula: Formula) -> str:
    """Deterministic short ID for a knowledge predicate.

    Uses md5 of ``agent:formula.to_pddl()`` truncated to 8 hex chars.
    """
    key = f"{agent}:{formula.to_pddl()}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _knowledge_fact_id(formula: EpistemicFormula) -> str:
    """Return the fact_id suffix used for the given epistemic formula."""
    if isinstance(formula.inner, (Knows, Believes)):
        return _fact_id(formula.agent, formula.inner)
    return _leaf_fact_hash(formula.inner)


def _epistemic_nesting_depth(formula: Formula) -> int:
    """Count the number of K/B layers in a formula."""
    if isinstance(formula, (Knows, Believes)):
        return 1 + _epistemic_nesting_depth(formula.inner)
    return 0


def _collect_k_goals(
    formula: Formula,
    obs: ObservabilityModel,
) -> List[KGoalNode]:
    """Walk formula tree and collect all K()/B() nodes as KGoalNodes.

    Each node's ``depth`` is its actual epistemic nesting depth (number of
    K/B layers from itself down), NOT its position from the formula root.
    E.g. K(a0, K(a2, phi)) has depth=2; the inner K(a2, phi) has depth=1.

    For depth-1 K goals, fact_id is the leaf-fact hash (shared across agents).
    For nested K goals, fact_id is agent-specific (uses _fact_id).
    """
    results: List[KGoalNode] = []

    if isinstance(formula, (Knows, Believes)):
        agent = formula.agent
        inner = formula.inner

        # Actual epistemic nesting depth: 1 (this K) + inner K layers
        nesting_depth = 1 + _epistemic_nesting_depth(inner)

        # Check if this K goal is trivial (agent can observe all entities)
        trivial = _is_formula_observable_by(agent, inner, obs)

        # Compute fact_id:
        # - For depth-1 K(agent, literal): use leaf hash (agent-independent)
        #   so all agents share the same predicate suffix for the same fact.
        # - For nested K(agent, K(...)): use agent-specific hash since the
        #   outer knowledge is about another agent's knowledge.
        inner_k_fid = None
        if isinstance(inner, (Knows, Believes)):
            # Nested: inner fact_id for the inner K goal
            inner_k_fid = _knowledge_fact_id(inner)
            # Outer fact_id is agent-specific
            fid = _fact_id(agent, inner)
        else:
            # Depth-1: use leaf hash
            fid = _leaf_fact_hash(inner)

        node = KGoalNode(
            agent=agent,
            inner=inner,
            fact_id=fid,
            depth=nesting_depth,
            trivial=trivial,
            inner_k_fact_id=inner_k_fid,
        )
        results.append(node)

        # Recurse into inner formula for nested K/B
        results.extend(_collect_k_goals(inner, obs))

    elif isinstance(formula, And):
        for op in formula.operands:
            results.extend(_collect_k_goals(op, obs))

    elif isinstance(formula, Or):
        for op in formula.operands:
            results.extend(_collect_k_goals(op, obs))

    elif isinstance(formula, Not) and formula.operand is not None:
        results.extend(_collect_k_goals(formula.operand, obs))

    return results


def _is_formula_observable_by(
    agent: str,
    formula: Formula,
    obs: ObservabilityModel,
) -> bool:
    """Check if agent can observe ALL entities referenced in a formula.

    For nested K: K(a0, K(a1, phi)) — a0 can "observe" the inner K
    only if a0 can observe phi AND a1 can also observe phi (then a0
    can infer a1 knows). This is handled by the inference action
    generation, not here. Here we only check leaf-level observability.
    """
    if isinstance(formula, (Knows, Believes)):
        # For nested K, the outer agent needs communication, not direct observation
        return False

    if isinstance(formula, Literal):
        return obs.is_fact_observable_by(agent, formula.predicate, formula.args)

    if isinstance(formula, And):
        return all(_is_formula_observable_by(agent, op, obs) for op in formula.operands)

    if isinstance(formula, Or):
        # Conservative: agent must be able to observe all branches
        return all(_is_formula_observable_by(agent, op, obs) for op in formula.operands)

    if isinstance(formula, Not) and formula.operand is not None:
        return _is_formula_observable_by(agent, formula.operand, obs)

    return True


# ---------------------------------------------------------------------------
# Leaf fact collection
# ---------------------------------------------------------------------------

def _collect_leaf_facts(
    k_goals: List[KGoalNode],
) -> Dict[str, Formula]:
    """Collect unique leaf (physical) facts from K goals.

    Returns mapping: fact_hash -> physical formula.
    The fact_hash is derived from the physical formula only (agent-independent)
    so that all agents share the same hash for the same fact.
    """
    facts: Dict[str, Formula] = {}
    for kg in k_goals:
        leaf = _get_leaf_formula(kg.inner)
        if leaf is not None:
            fhash = _leaf_fact_hash(leaf)
            facts[fhash] = leaf
    return facts


def _get_leaf_formula(formula: Formula) -> Optional[Formula]:
    """Unwrap K/B to get the innermost physical formula."""
    if isinstance(formula, (Knows, Believes)):
        return _get_leaf_formula(formula.inner)
    return formula


def _leaf_fact_hash(formula: Formula) -> str:
    """Deterministic hash for a physical formula (agent-independent)."""
    return hashlib.md5(formula.to_pddl().encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Knowledge predicates (network: all agents per leaf fact)
# ---------------------------------------------------------------------------

def _build_knowledge_predicates_network(
    k_goals: List[KGoalNode],
    leaf_facts: Dict[str, Formula],
    all_agents: List[str],
) -> List[str]:
    """Build 0-ary predicates for the knowledge propagation network.

    For each unique leaf fact, creates knows_<agent>_<hash> for ALL agents
    (not just goal targets). Also creates predicates for nested K goals.
    """
    seen: Set[str] = set()
    preds: List[str] = []

    # Leaf-fact knowledge predicates for all agents
    for fhash in sorted(leaf_facts):
        for agent in all_agents:
            pred_name = f"knows_{agent}_{fhash}"
            if pred_name not in seen:
                seen.add(pred_name)
                preds.append(f"({pred_name})")

    # Nested K predicates (outer-layer: a0 knows that a2 knows)
    for kg in k_goals:
        pred_name = f"knows_{kg.agent}_{kg.fact_id}"
        if pred_name not in seen:
            seen.add(pred_name)
            preds.append(f"({pred_name})")

    return preds


# ---------------------------------------------------------------------------
# Observe actions (network: any agent who can see a fact)
# ---------------------------------------------------------------------------

def _build_observe_actions_network(
    leaf_facts: Dict[str, Formula],
    all_agents: List[str],
    obs: ObservabilityModel,
) -> List[str]:
    """Build observe actions for every agent who can see each leaf fact."""
    actions: List[str] = []
    used_action_names: set[str] = set()

    for fhash, formula in sorted(leaf_facts.items()):
        precond = formula.to_pddl()
        for agent in all_agents:
            if _is_formula_observable_by(agent, formula, obs):
                pred_name = f"knows_{agent}_{fhash}"
                action = (
                    f"(:action observe_{pred_name}\n"
                    f"  :parameters ()\n"
                    f"  :precondition {precond}\n"
                    f"  :effect ({pred_name}))"
                )
                actions.append(action)

    return actions


# ---------------------------------------------------------------------------
# Inform actions (network: relay via any sender who knows)
# ---------------------------------------------------------------------------

def _build_inform_actions_network(
    leaf_facts: Dict[str, Formula],
    k_goals: List[KGoalNode],
    all_agents: List[str],
    can_comm: Set[Tuple[str, str]],
    obs: ObservabilityModel,
) -> List[str]:
    """Build inform actions for knowledge relay across all agent pairs.

    For each fact and each (sender, receiver) with can_communicate:
    - Precondition: sender knows the fact + budget token
    - Effect: receiver knows the fact + consume token
    - Optional effect: sender knows receiver now knows the fact

    FD discovers relay chains: observe → knows_a0 → inform → knows_a1 → ...
    """
    actions: List[str] = []
    used_action_names: set[str] = set()
    sender_ack_preds: Dict[Tuple[str, str, str], List[str]] = {}

    for kg in k_goals:
        if not isinstance(kg.inner, (Knows, Believes)):
            continue
        if isinstance(kg.inner.inner, (Knows, Believes)):
            continue

        key = (kg.agent, kg.inner.agent, _leaf_fact_hash(kg.inner.inner))
        pred_name = f"knows_{kg.agent}_{kg.fact_id}"
        sender_ack_preds.setdefault(key, []).append(pred_name)

    for fhash in sorted(leaf_facts):
        for sender in all_agents:
            for receiver in all_agents:
                if sender == receiver:
                    continue
                if (sender, receiver) not in can_comm:
                    continue

                sender_pred = f"knows_{sender}_{fhash}"
                receiver_pred = f"knows_{receiver}_{fhash}"
                ack_preds = sorted(set(sender_ack_preds.get((sender, receiver, fhash), [])))
                effect_preds = [receiver_pred, *ack_preds]

                budget = obs.message_limits.get(sender)
                if budget is not None:
                    for tok_idx in range(1, budget + 1):
                        tok_pred = f"msg_tok_{sender}_{tok_idx}"
                        effect_clause = " ".join(f"({pred})" for pred in effect_preds)
                        action = (
                            f"(:action inform_{receiver_pred}_from_{sender}_tok{tok_idx}_{fhash}\n"
                            f"  :parameters ()\n"
                            f"  :precondition (and ({sender_pred}) "
                            f"(can_communicate {sender} {receiver}) "
                            f"({tok_pred}))\n"
                            f"  :effect (and {effect_clause} (not ({tok_pred}))))"
                        )
                        actions.append(action)

                # Ensure unique action names even if receiver_pred repeats due to normalization or hashing collisions
                # by suffixing with a stable per-(fhash,sender,receiver) index.
                else:
                    effect_clause = (
                        f"({effect_preds[0]})"
                        if len(effect_preds) == 1
                        else f"(and {' '.join(f'({pred})' for pred in effect_preds)})"
                    )
                    action = (
                        f"(:action inform_{receiver_pred}_from_{sender}_{fhash}\n"
                        f"  :parameters ()\n"
                        f"  :precondition (and ({sender_pred}) "
                        f"(can_communicate {sender} {receiver}))\n"
                        f"  :effect {effect_clause})"
                    )
                    actions.append(action)

    return actions


# ---------------------------------------------------------------------------
# Nested K actions (outer-layer communication)
# ---------------------------------------------------------------------------

def _build_nested_k_actions(
    k_goals: List[KGoalNode],
    all_agents: List[str],
    can_comm: Set[Tuple[str, str]],
    obs: ObservabilityModel,
) -> List[str]:
    """Build actions for nested K goals (K(a0, K(a2, phi))).

    The outer agent can only learn nested knowledge via communication from
    the inner agent. Direct observation of the physical fact is not enough to
    establish what another agent knows.
    """
    actions: List[str] = []
    used_action_names: set[str] = set()

    for kg in k_goals:
        if not isinstance(kg.inner, (Knows, Believes)):
            continue

        outer_agent = kg.agent
        inner_agent = kg.inner.agent
        outer_pred = f"knows_{outer_agent}_{kg.fact_id}"

        # The inner K's fact_id is for the inner agent's knowledge of the leaf
        inner_fid = kg.inner_k_fact_id
        if inner_fid is None:
            continue
        inner_pred = f"knows_{inner_agent}_{inner_fid}"

        if _get_leaf_formula(kg.inner.inner) is None:
            continue

        # Communication: the inner agent tells the outer agent about the
        # inner knowledge fact.
        for sender in all_agents:
            if sender == outer_agent:
                continue
            if (sender, outer_agent) not in can_comm:
                continue

            if sender != inner_agent:
                continue

            budget = obs.message_limits.get(sender)
            if budget is not None:
                for tok_idx in range(1, budget + 1):
                    tok_pred = f"msg_tok_{sender}_{tok_idx}"
                    action = (
                        f"(:action inform_{outer_pred}_from_{sender}_tok{tok_idx}\n"
                        f"  :parameters ()\n"
                        f"  :precondition (and ({inner_pred}) "
                        f"(can_communicate {sender} {outer_agent}) "
                        f"({tok_pred}))\n"
                        f"  :effect (and ({outer_pred}) (not ({tok_pred}))))"
                    )
                    actions.append(action)
            else:
                action = (
                    f"(:action inform_{outer_pred}_from_{sender}\n"
                    f"  :parameters ()\n"
                    f"  :precondition (and ({inner_pred}) "
                    f"(can_communicate {sender} {outer_agent}))\n"
                    f"  :effect ({outer_pred}))"
                )
                actions.append(action)

    return actions


# ---------------------------------------------------------------------------
# Budget tokens
# ---------------------------------------------------------------------------

def _build_budget_tokens(obs: ObservabilityModel) -> List[str]:
    """Build init literals for message budget tokens.

    For agent with budget N: (msg_tok_agent_1_1) ... (msg_tok_agent_1_N).
    """
    tokens: List[str] = []
    for agent, limit in sorted(obs.message_limits.items()):
        if limit is None:
            continue
        for i in range(1, limit + 1):
            tokens.append(f"(msg_tok_{agent}_{i})")
    return tokens


def _build_budget_predicates(obs: ObservabilityModel) -> List[str]:
    """Build 0-ary predicate declarations for budget tokens."""
    preds: List[str] = []
    seen: Set[str] = set()
    for agent, limit in sorted(obs.message_limits.items()):
        if limit is None:
            continue
        for i in range(1, limit + 1):
            pred = f"(msg_tok_{agent}_{i})"
            if pred not in seen:
                seen.add(pred)
                preds.append(pred)
    return preds


# ---------------------------------------------------------------------------
# Goal replacement
# ---------------------------------------------------------------------------

def _replace_epistemic_in_goal(
    formula: Formula,
    k_goal_map: Dict[str, KGoalNode],
) -> Formula:
    """Replace K(a, phi) in goal with And(phi_stripped, knows_a_HASH).

    The physical part (phi stripped of K/B) must still hold at end-state,
    AND the knowledge predicate must be achieved. K() goals must reference
    stable facts that remain true throughout the episode.
    """
    if isinstance(formula, (Knows, Believes)):
        key = _k_goal_key_from_formula(formula)
        kg = k_goal_map.get(key)
        if kg:
            pred_name = f"knows_{kg.agent}_{kg.fact_id}"
            knowledge_lit = Literal(pred_name)

            # Strip K/B directly from inner formula for physical requirements.
            # For K(a0, K(a1, phi)) this produces And(phi, knows_a0_hash).
            # The inner K(a1, phi), when it appears as a separate conjunct in
            # the top-level goal, gets its own expansion.
            physical_inner = _strip_epistemic_formula(formula.inner)

            return And((physical_inner, knowledge_lit))
        # Fallback: strip epistemic
        return _strip_epistemic_formula(formula)

    if isinstance(formula, And):
        return And(tuple(
            _replace_epistemic_in_goal(op, k_goal_map)
            for op in formula.operands
        ))

    if isinstance(formula, Or):
        return Or(tuple(
            _replace_epistemic_in_goal(op, k_goal_map)
            for op in formula.operands
        ))

    if isinstance(formula, Not) and formula.operand is not None:
        return Not(operand=_replace_epistemic_in_goal(formula.operand, k_goal_map))

    return formula  # Literal unchanged


# ---------------------------------------------------------------------------
# PDDL string manipulation
# ---------------------------------------------------------------------------

def _collect_action_constants(
    extra_actions: List[str],
    problem: Problem,
) -> Dict[str, str]:
    """Extract grounded object names from action strings and map to types.

    Unified-planning's PDDLReader requires any object referenced in domain
    actions to be declared as :constants. We scan the action strings for
    known problem objects AND collect all agents (since inform actions
    reference them via can_communicate).
    """
    constants: Dict[str, str] = {}
    # Combine all action text for a single scan pass
    combined = "\n".join(extra_actions)
    for obj_name, obj_type in problem.objects.items():
        # Agents are always needed as constants: inform actions reference them
        # via can_communicate, and :init facts like agent_in_room use them.
        # Excluding any agent from :constants causes unified-planning grounding
        # failures (UNSOLVABLE_INCOMPLETELY).
        if obj_type == "agent":
            constants[obj_name] = obj_type
        # Look for the object name as a whole word in any action
        elif re.search(rf'\b{re.escape(obj_name)}\b', combined):
            constants[obj_name] = obj_type
    return constants


def _build_augmented_domain_pddl(
    base_domain: str,
    extra_preds: List[str],
    extra_actions: List[str],
    constants: Optional[Dict[str, str]] = None,
) -> str:
    """Inject extra predicates, actions, and constants into base domain PDDL."""
    result = base_domain

    # Insert :constants section after :types if needed
    if constants:
        by_type: Dict[str, List[str]] = {}
        for name, typ in constants.items():
            by_type.setdefault(typ, []).append(name)
        const_lines = []
        for typ, names in sorted(by_type.items()):
            const_lines.append(f"    {' '.join(sorted(names))} - {typ}")
        const_block = "  (:constants\n" + "\n".join(const_lines) + "\n  )\n"
        # Insert after (:types ...) line
        types_match = re.search(r'\(:types\b[^)]*\)', result)
        if types_match:
            insert_pos = types_match.end()
            result = result[:insert_pos] + "\n" + const_block + result[insert_pos:]

    # Insert extra predicates before closing ) of :predicates section
    if extra_preds:
        preds_str = "\n    ".join(extra_preds)
        pred_pattern = re.compile(r'(\(:predicates\b.*?)(  \))', re.DOTALL)
        match = pred_pattern.search(result)
        if match:
            result = (
                result[:match.end(1)]
                + "\n    " + preds_str + "\n"
                + result[match.start(2):]
            )

    # Insert extra actions before the final closing ) of the domain
    if extra_actions:
        actions_str = "\n\n".join(extra_actions)
        last_paren = result.rfind(")")
        if last_paren >= 0:
            result = result[:last_paren] + "\n" + actions_str + "\n)"

    return result


def _build_augmented_problem_pddl(
    problem: Problem,
    classical_goal: Formula,
    budget_tokens: List[str],
    constants: Optional[Dict[str, str]] = None,
) -> str:
    """Build a problem PDDL string with the classical goal and budget tokens.

    Objects already declared as :constants in the domain are excluded from
    the problem's :objects section to avoid duplicate declarations.
    """
    constant_names = set(constants or {})

    # Objects (excluding constants)
    obj_lines = []
    by_type: Dict[str, List[str]] = {}
    for name, typ in problem.objects.items():
        if name in constant_names:
            continue
        by_type.setdefault(typ, []).append(name)
    for typ, names in sorted(by_type.items()):
        obj_lines.append(f"    {' '.join(sorted(names))} - {typ}")
    objects_str = "\n".join(obj_lines)

    # Init: original + budget tokens
    init_parts = [l.to_pddl() for l in problem.init if not l.negated]
    init_parts.extend(budget_tokens)
    init_str = "\n    ".join(init_parts)

    # Goal
    goal_str = classical_goal.to_pddl()

    return (
        f"(define (problem {problem.name})\n"
        f"  (:domain {problem.domain_name})\n"
        f"  (:objects\n{objects_str}\n  )\n"
        f"  (:init\n    {init_str}\n  )\n"
        f"  (:goal {goal_str})\n"
        f")"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_epistemic_formula(formula: Formula) -> Formula:
    """Unwrap K()/B() to get the physical formula."""
    if isinstance(formula, (Knows, Believes)):
        return _strip_epistemic_formula(formula.inner)
    if isinstance(formula, And):
        return And(tuple(_strip_epistemic_formula(op) for op in formula.operands))
    if isinstance(formula, Or):
        return Or(tuple(_strip_epistemic_formula(op) for op in formula.operands))
    if isinstance(formula, Not) and formula.operand is not None:
        return Not(operand=_strip_epistemic_formula(formula.operand))
    return formula


def _k_goal_key(kg: KGoalNode) -> str:
    """Build a unique key for a K goal node."""
    return f"{kg.agent}:{kg.inner.to_pddl()}"


def _k_goal_key_from_formula(formula: Formula) -> str:
    """Build a key from a K/B formula (matches _k_goal_key)."""
    if isinstance(formula, (Knows, Believes)):
        return f"{formula.agent}:{formula.inner.to_pddl()}"
    return ""


def _k_goal_pddl(kg: KGoalNode) -> str:
    """Reconstruct the PDDL string for a K goal."""
    op = "K" if True else "B"  # Both treated same for compilation
    return f"({op} {kg.agent} {kg.inner.to_pddl()})"


def _problem_to_classical_pddl(problem: Problem) -> str:
    """Serialize a Problem to classical PDDL (no epistemic extensions)."""
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
