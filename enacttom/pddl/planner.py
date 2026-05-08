"""
Deterministic golden trajectory planner backed by strict PDDL solving.

Given a task spec and optional scene data, this module solves the authoritative
PDDL problem with Fast Downward and translates the resulting plan into a
golden trajectory. Same spec + scene always produces the same trajectory.

This module is used by both agent.py (ReAct loop) and the CLI verify/submit
commands, ensuring a single source of truth for trajectory generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from enacttom.task_gen.task_generator import normalize_mechanic_bindings

logger = logging.getLogger(__name__)


@dataclass
class InformAction:
    """A parsed inform action from an FD epistemic plan."""
    receiver: str       # agent who receives knowledge
    fact_hash: str      # 8-char hex hash of the leaf fact
    sender: str         # agent who sends the information


@dataclass(frozen=True)
class RuntimeOp:
    """One translated runtime action before parallel turn packing."""
    agent: str
    action: str
    resources: Tuple[Tuple[str, str], ...] = ()


@dataclass
class RuntimeState:
    """Minimal runtime state used to recover conservative action dependencies."""
    object_supports: Dict[str, str]
    held_by: Dict[str, str]


# Regex for FD inform action names:
#   inform_knows_{receiver}_{hash}_from_{sender}[_tokN]
_INFORM_RE = re.compile(
    r"inform_knows_(agent_\d+)_([0-9a-f]{8})_from_(agent_\d+)(?:_tok\d+)?"
)


def parse_fd_inform_actions(fd_plan: List[str]) -> List[InformAction]:
    """Parse inform actions from an FD plan, preserving plan order.

    FD plan steps are strings like:
        "inform_knows_agent_0_abc12345_from_agent_1"
        "inform_knows_agent_0_abc12345_from_agent_1_tok1"

    Returns InformActions in plan order (preserves relay chain ordering).
    """
    results: List[InformAction] = []
    for step in fd_plan:
        m = _INFORM_RE.search(step)
        if m:
            results.append(InformAction(
                receiver=m.group(1),
                fact_hash=m.group(2),
                sender=m.group(3),
            ))
    return results


_PLAN_STEP_RE = re.compile(
    r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?:\((?P<args>.*)\))?$"
)


def _scene_to_dict(scene_data: Any) -> Dict[str, Any]:
    """Normalize SceneData-like inputs into plain dicts."""
    if isinstance(scene_data, dict):
        return scene_data
    if hasattr(scene_data, "to_dict"):
        return scene_data.to_dict()
    return {}


def _parse_plan_step(step: str) -> Tuple[str, List[str]]:
    """Parse a unified-planning action string into name and ordered args."""
    stripped = (step or "").strip()
    match = _PLAN_STEP_RE.match(stripped)
    if not match:
        raise ValueError(f"Unsupported planner step format: {step}")

    name = match.group("name")
    args_str = (match.group("args") or "").strip()
    if not args_str:
        return name, []
    return name, [arg.strip() for arg in args_str.split(",") if arg.strip()]


def _collect_goal_relation_preferences(goal: Any) -> Dict[Tuple[str, str], str]:
    """Map goal object/receptacle pairs to the intended runtime place relation."""
    from enacttom.pddl.dsl import And, Believes, Knows, Literal, Not, Or

    relation_by_pair: Dict[Tuple[str, str], str] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, (Knows, Believes)):
            _walk(node.inner)
            return
        if isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)
            return
        if isinstance(node, (And, Or)):
            for operand in node.operands:
                _walk(operand)
            return
        if isinstance(node, Literal) and not node.negated and len(node.args) >= 2:
            if node.predicate == "is_inside":
                relation_by_pair[(node.args[0], node.args[1])] = "within"
            elif node.predicate == "is_on_top":
                relation_by_pair.setdefault((node.args[0], node.args[1]), "on")

    _walk(goal)
    return relation_by_pair


def _build_epistemic_message_maps(goal: Any, observability: Any) -> Tuple[Dict[str, Any], Dict[str, tuple]]:
    """Build fact-hash maps needed to translate inform actions to messages."""
    from enacttom.pddl.dsl import Believes, Knows
    from enacttom.pddl.epistemic_compiler import (
        _collect_k_goals,
        _collect_leaf_facts,
        _get_leaf_formula,
    )

    k_goals = _collect_k_goals(goal, observability)
    leaf_facts = _collect_leaf_facts(k_goals)

    nested_k_map: Dict[str, tuple] = {}
    for kg in k_goals:
        if kg.depth < 2 or not isinstance(kg.inner, (Knows, Believes)):
            continue
        leaf = _get_leaf_formula(kg.inner.inner)
        if leaf is None:
            continue
        nested_k_map[kg.fact_id] = (kg.agent, kg.inner.agent, leaf)

    return leaf_facts, nested_k_map


def _build_communicate_action(
    inform: InformAction,
    num_agents: int,
    leaf_facts: Dict[str, Any],
    nested_k_map: Dict[str, tuple],
) -> Dict[str, Any]:
    """Convert an epistemic inform action into one golden trajectory step."""
    from enacttom.pddl.describe import _literal_to_nl, goal_to_natural_language
    from enacttom.pddl.dsl import Literal

    formula = leaf_facts.get(inform.fact_hash)
    nested = nested_k_map.get(inform.fact_hash)

    if formula is not None and isinstance(formula, Literal):
        message = _literal_to_nl(formula)
    elif formula is not None:
        message = goal_to_natural_language(formula)
    elif nested is not None:
        _outer_agent, inner_agent, leaf = nested
        if isinstance(leaf, Literal):
            inner_msg = _literal_to_nl(leaf)
        else:
            inner_msg = goal_to_natural_language(leaf)
        message = f"{inner_agent} confirmed: {inner_msg}"
    else:
        raise RuntimeError(
            f"Unknown fact_hash '{inform.fact_hash}' in inform action; "
            "cannot derive deterministic Communicate step."
        )

    return wrap_parallel_step(
        num_agents,
        inform.sender,
        f'Communicate["{message}", {inform.receiver}]',
    )


def _solve_task_for_trajectory(
    task_data: Dict[str, Any],
    scene_data: Any,
) -> Tuple[Any, Any, Any]:
    """Solve the authoritative compiled task and return (problem, obs, solver_result)."""
    from enacttom.pddl.compiler import compile_task
    from enacttom.pddl.domain import ENACTTOM_DOMAIN
    from enacttom.pddl.epistemic import ObservabilityModel
    from enacttom.pddl.fd_solver import FastDownwardSolver
    from enacttom.task_gen.task_generator import GeneratedTask

    scene_dict = _scene_to_dict(scene_data)
    task = GeneratedTask.from_dict(task_data)
    problem = compile_task(task, scene_dict)
    observability = ObservabilityModel.from_task_with_scene(task, scene_dict)

    if _has_epistemic_goal(problem.goal) and not observability.object_rooms:
        raise ValueError(
            "Epistemic trajectory derivation requires scene object-room mapping "
            "(missing observability.object_rooms)."
        )

    solver = FastDownwardSolver()
    result = solver.solve(
        ENACTTOM_DOMAIN,
        problem,
        observability,
        max_belief_depth=3,
        strict=False,
    )
    if not result.solvable:
        raise RuntimeError(
            f"Strict PDDL solve failed: {result.error or 'no plan found'}"
        )

    return problem, observability, result


def _derive_communicate_steps(
    task_data: Dict[str, Any],
    scene_data: Any,
    pddl_goal: str,
    num_agents: int,
) -> Dict[str, Any]:
    """Derive Communicate trajectory steps from the strict solver plan."""
    from enacttom.pddl.fd_solver import HAS_UP

    if not HAS_UP:
        raise RuntimeError(
            "unified-planning is required for epistemic trajectory derivation "
            "(install unified-planning and up-fast-downward)."
        )

    problem, observability, result = _solve_task_for_trajectory(task_data, scene_data)
    if not _has_epistemic_goal(problem.goal):
        return {
            "steps": [],
            "notes": ["no epistemic goals in problem; no communicate steps needed"],
            "communication_required": False,
        }

    informs = parse_fd_inform_actions(result.plan or [])
    if not informs:
        return {
            "steps": [],
            "notes": [
                "belief_depth=0 or direct observation made communication unnecessary"
            ],
            "communication_required": False,
        }

    leaf_facts, nested_k_map = _build_epistemic_message_maps(problem.goal, observability)
    steps = [
        _build_communicate_action(inform, num_agents, leaf_facts, nested_k_map)
        for inform in informs
    ]
    return {
        "steps": steps,
        "notes": [f"derived {len(steps)} Communicate step(s) from strict solver plan"],
        "communication_required": bool(steps),
    }


def _has_epistemic_goal(formula: Any) -> bool:
    """Return True when goal formula contains K()/B() operators."""
    from enacttom.pddl.dsl import And, Or, Not, Knows, Believes

    if isinstance(formula, (Knows, Believes)):
        return True
    if isinstance(formula, (And, Or)):
        return any(_has_epistemic_goal(op) for op in formula.operands)
    if isinstance(formula, Not) and formula.operand is not None:
        return _has_epistemic_goal(formula.operand)
    return False


def canonical_json(value: Any) -> str:
    """Canonical JSON serialization for stable hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_task_spec_hash(task_data: Dict[str, Any]) -> str:
    """Hash the authoritative task spec (excluding golden trajectory artifacts)."""
    spec_keys = [
        "task_id", "title", "category", "task",
        "scene_id", "episode_id", "num_agents",
        "active_mechanics", "mechanic_bindings",
        "agent_secrets", "agent_actions",
        "pddl_domain", "problem_pddl",
        "initial_states",
        "message_targets",
        "agent_spawns",
    ]
    spec_payload = {k: task_data.get(k) for k in spec_keys}
    return hashlib.sha256(canonical_json(spec_payload).encode("utf-8")).hexdigest()


def extract_room_restrictions(task_data: Dict[str, Any]) -> Dict[str, set]:
    """Build agent -> restricted rooms map from room_restriction mechanics."""
    restrictions: Dict[str, set] = {}
    for binding in normalize_mechanic_bindings(
        task_data.get("mechanic_bindings", []),
        problem_pddl=task_data.get("problem_pddl"),
    ):
        if not isinstance(binding, dict):
            continue
        if binding.get("mechanic_type") != "room_restriction":
            continue
        rooms = binding.get("restricted_rooms") or []
        agents = binding.get("for_agents") or []
        for agent_id in agents:
            if not isinstance(agent_id, str):
                continue
            restrictions.setdefault(agent_id, set()).update(
                room for room in rooms if isinstance(room, str)
            )
    return restrictions


def build_target_to_room_map(scene_data: Optional[Any]) -> Dict[str, str]:
    """Map room/furniture/object IDs to room IDs."""
    if not scene_data:
        return {}

    def _get(field: str, default):
        if hasattr(scene_data, field):
            return getattr(scene_data, field)
        if isinstance(scene_data, dict):
            return scene_data.get(field, default)
        return default

    rooms = _get("rooms", []) or []
    furniture_in_rooms = _get("furniture_in_rooms", {}) or {}
    objects_on_furniture = _get("objects_on_furniture", {}) or {}

    target_to_room: Dict[str, str] = {}
    for room in rooms:
        if isinstance(room, str):
            target_to_room[room] = room

    furniture_to_room: Dict[str, str] = {}
    if isinstance(furniture_in_rooms, dict):
        for room, furns in furniture_in_rooms.items():
            if not isinstance(room, str) or not isinstance(furns, list):
                continue
            for furn in furns:
                if isinstance(furn, str):
                    furniture_to_room[furn] = room
                    target_to_room[furn] = room

    if isinstance(objects_on_furniture, dict):
        for furn, objs in objects_on_furniture.items():
            room = furniture_to_room.get(furn)
            if not room or not isinstance(objs, list):
                continue
            for obj in objs:
                if isinstance(obj, str):
                    target_to_room[obj] = room

    return target_to_room


def _agent_sort_key(agent_id: str, agent_loads: Dict[str, int]) -> tuple:
    """Stable deterministic tie-breaker: (load, numeric agent index)."""
    try:
        idx = int(agent_id.split("_", 1)[1])
    except Exception:
        idx = 0
    return (agent_loads.get(agent_id, 0), idx)


def _resolve_target_room(
    target_id: str,
    target_to_room: Dict[str, str],
    restrictions: Dict[str, set],
) -> Optional[str]:
    """Resolve target_id to its room, checking restriction maps as fallback."""
    room = target_to_room.get(target_id)
    if room is not None:
        return room
    if restrictions:
        restricted_rooms = {
            r for rooms in restrictions.values() for r in rooms if isinstance(r, str)
        }
        if target_id in restricted_rooms:
            return target_id
    return None


def _feasible_agents(
    target_rooms: List[Optional[str]],
    num_agents: int,
    restrictions: Dict[str, set],
) -> List[str]:
    """Return agents that can reach ALL of the given rooms."""
    feasible: List[str] = []
    for i in range(max(1, num_agents)):
        agent_id = f"agent_{i}"
        agent_restricted = restrictions.get(agent_id, set())
        if all(
            room is None or room not in agent_restricted
            for room in target_rooms
        ):
            feasible.append(agent_id)
    return feasible


def pick_agent_for_target(
    target_id: str,
    num_agents: int,
    target_to_room: Dict[str, str],
    restrictions: Dict[str, set],
    agent_loads: Optional[Dict[str, int]] = None,
) -> str:
    """
    Pick a deterministic feasible agent for interacting with a target.

    When agent_loads is provided, prefers the least-loaded feasible agent to
    reduce trajectories where one agent does all work while others wait.
    """
    target_room = _resolve_target_room(target_id, target_to_room, restrictions)

    if target_room is None and restrictions:
        raise ValueError(
            f"Cannot assign agent for target '{target_id}': missing room mapping "
            "while room_restriction mechanics are active. Provide scene_data with "
            "furniture/object room mappings."
        )

    feasible = _feasible_agents([target_room], num_agents, restrictions)

    if not feasible:
        raise ValueError(
            f"No agent can reach {target_id} (room={target_room}). "
            f"All agents are restricted. Check room_restriction mechanics."
        )
    if not agent_loads:
        return feasible[0]

    feasible.sort(key=lambda a: _agent_sort_key(a, agent_loads))
    return feasible[0]


def pick_agent_for_targets(
    target_ids: List[str],
    num_agents: int,
    target_to_room: Dict[str, str],
    restrictions: Dict[str, set],
    agent_loads: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """Pick an agent that can reach ALL targets. Returns None if impossible."""
    target_rooms = [
        _resolve_target_room(tid, target_to_room, restrictions)
        for tid in target_ids
    ]
    feasible = _feasible_agents(target_rooms, num_agents, restrictions)
    if not feasible:
        return None
    if not agent_loads:
        return feasible[0]
    feasible.sort(key=lambda a: _agent_sort_key(a, agent_loads))
    return feasible[0]


def find_handoff_furniture(
    agent_a: str,
    agent_b: str,
    restrictions: Dict[str, set],
    scene_data: Optional[Any],
) -> Optional[str]:
    """Find a furniture item in a room accessible by both agents for handoff.

    Returns the first furniture item in a shared room, or None if no shared
    room exists.
    """
    if not scene_data:
        return None

    if hasattr(scene_data, "furniture_in_rooms"):
        fir = scene_data.furniture_in_rooms
    elif isinstance(scene_data, dict):
        fir = scene_data.get("furniture_in_rooms", {})
    else:
        return None

    a_restricted = restrictions.get(agent_a, set())
    b_restricted = restrictions.get(agent_b, set())

    for room in sorted(fir.keys()):
        if room in a_restricted or room in b_restricted:
            continue
        furns = fir.get(room, [])
        if furns:
            return furns[0]
    return None


def build_parallel_step(
    num_agents: int,
    actions_by_agent: Dict[str, str],
) -> Dict[str, Any]:
    """Create one parallel step, filling missing agents with Wait."""
    actions = []
    for i in range(max(1, num_agents)):
        agent_id = f"agent_{i}"
        actions.append({
            "agent": agent_id,
            "action": actions_by_agent.get(agent_id, "Wait[]"),
        })
    return {"actions": actions}


def wrap_parallel_step(num_agents: int, acting_agent: str, action: str) -> Dict[str, Any]:
    """Create one parallel step with one active agent and Wait for others."""
    return build_parallel_step(num_agents, {acting_agent: action})


def _dedupe_resources(resources: List[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    """Preserve resource order while removing duplicates."""
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for resource in resources:
        if resource in seen:
            continue
        seen.add(resource)
        deduped.append(resource)
    return tuple(deduped)


def append_runtime_ops(
    runtime_ops: List[RuntimeOp],
    acting_agent: str,
    action: str,
    resources: Optional[List[Tuple[str, str]]] = None,
    navigate_target: Optional[str] = None,
) -> None:
    """Append translated runtime ops, optionally inserting target navigation first."""
    if navigate_target:
        runtime_ops.append(RuntimeOp(agent=acting_agent, action=f"Navigate[{navigate_target}]"))
    runtime_ops.append(
        RuntimeOp(
            agent=acting_agent,
            action=action,
            resources=_dedupe_resources(resources or []),
        )
    )


def extract_runtime_state(problem: Any) -> RuntimeState:
    """Extract the minimal mutable state needed for dependency-aware translation."""
    from enacttom.pddl.dsl import Literal

    object_supports: Dict[str, str] = {}
    held_by: Dict[str, str] = {}

    for literal in problem.init:
        if not isinstance(literal, Literal) or literal.negated:
            continue
        if literal.predicate in ("is_on_top", "is_inside") and len(literal.args) == 2:
            object_supports[literal.args[0]] = literal.args[1]
            held_by.pop(literal.args[0], None)
        elif literal.predicate == "is_held_by" and len(literal.args) == 2:
            held_by[literal.args[0]] = literal.args[1]
            object_supports.pop(literal.args[0], None)

    return RuntimeState(object_supports=object_supports, held_by=held_by)


def schedule_runtime_ops(num_agents: int, runtime_ops: List[RuntimeOp]) -> List[Dict[str, Any]]:
    """Pack independent runtime ops into earliest valid parallel turns."""
    if not runtime_ops:
        return [wrap_parallel_step(num_agents, "agent_0", "Wait[]")]

    step_actions: List[Dict[str, str]] = []
    last_step_by_agent: Dict[str, int] = {}
    last_step_by_resource: Dict[Tuple[str, str], int] = {}

    for op in runtime_ops:
        target_step = last_step_by_agent.get(op.agent, -1) + 1
        for resource in op.resources:
            target_step = max(target_step, last_step_by_resource.get(resource, -1) + 1)

        while True:
            if target_step == len(step_actions):
                step_actions.append({})
            if op.agent not in step_actions[target_step]:
                break
            target_step += 1

        step_actions[target_step][op.agent] = op.action
        last_step_by_agent[op.agent] = target_step
        for resource in op.resources:
            last_step_by_resource[resource] = target_step

    return [build_parallel_step(num_agents, actions) for actions in step_actions]


def extract_plannable_literals(goal_formula: Any) -> List[Any]:
    """
    Extract a deterministic list of literals to satisfy from a PDDL goal.

    For OR branches, picks the lexicographically first branch to keep outputs stable.
    Unwraps epistemic operators and plans against their inner world-state literals.
    """
    from enacttom.pddl.dsl import Literal, And, Or, Not, Knows, Believes

    def _unwrap_epistemic(node: Any) -> Any:
        while isinstance(node, (Knows, Believes)):
            node = node.inner
        return node

    def _collect(node: Any) -> List[Literal]:
        node = _unwrap_epistemic(node)
        if isinstance(node, Literal):
            return [node]
        if isinstance(node, Not):
            inner = _unwrap_epistemic(node.operand)
            if isinstance(inner, Literal):
                return [Literal(predicate=inner.predicate, args=inner.args, negated=not inner.negated)]
            return _collect(inner)
        if isinstance(node, And):
            out: List[Literal] = []
            for op in node.operands:
                out.extend(_collect(op))
            return out
        if isinstance(node, Or):
            if not node.operands:
                return []
            chosen = sorted(node.operands, key=lambda op: op.to_pddl())[0]
            return _collect(chosen)
        return []

    literals = _collect(goal_formula)

    deduped = []
    seen = set()
    for lit in literals:
        key = (lit.predicate, lit.args, lit.negated)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(lit)
    return deduped


def apply_literal_ordering(literals: List[Any], ordering: List[Dict[str, str]]) -> List[Any]:
    """Apply topological ordering constraints when they reference extracted literals."""
    if not literals:
        return literals

    literal_strs = [lit.to_pddl() for lit in literals]
    idx_by_str = {s: i for i, s in enumerate(literal_strs)}
    n = len(literals)
    incoming = {i: set() for i in range(n)}
    outgoing = {i: set() for i in range(n)}

    for rule in ordering or []:
        if not isinstance(rule, dict):
            continue
        before = rule.get("before")
        after = rule.get("after")
        if before not in idx_by_str or after not in idx_by_str:
            continue
        bi, ai = idx_by_str[before], idx_by_str[after]
        if bi == ai or ai in outgoing[bi]:
            continue
        outgoing[bi].add(ai)
        incoming[ai].add(bi)

    ready = [i for i in range(n) if not incoming[i]]
    ready.sort()
    ordered_idx = []
    while ready:
        cur = ready.pop(0)
        ordered_idx.append(cur)
        for nxt in sorted(outgoing[cur]):
            incoming[nxt].discard(cur)
            if not incoming[nxt]:
                ready.append(nxt)
        ready.sort()

    if len(ordered_idx) != n:
        return literals
    return [literals[i] for i in ordered_idx]


def _rebalance_agent_assignments(
    runtime_ops: List[RuntimeOp],
    task_data: Dict[str, Any],
    scene_data: Optional[Any],
    num_agents: int,
) -> List[RuntimeOp]:
    """Reassign agents in runtime_ops to distribute work across all agents.

    The FD planner has no multi-agent distribution preference and typically
    assigns everything to agent_0. This post-pass uses pick_agent_for_target
    with load tracking to spread actions while respecting room restrictions.

    Navigate ops inserted by append_runtime_ops are paired with the action
    op that follows them — both get the same reassigned agent.
    """
    if num_agents < 2 or not runtime_ops:
        return runtime_ops

    restrictions = extract_room_restrictions(task_data)
    target_to_room = build_target_to_room_map(scene_data)
    agent_loads: Dict[str, int] = {f"agent_{i}": 0 for i in range(num_agents)}

    # Group ops into logical units: (optional Navigate, action).
    # append_runtime_ops always emits Navigate before the action op.
    groups: List[List[int]] = []  # each group is a list of indices
    i = 0
    while i < len(runtime_ops):
        op = runtime_ops[i]
        if op.action.startswith("Navigate[") and i + 1 < len(runtime_ops):
            next_op = runtime_ops[i + 1]
            if not next_op.action.startswith("Navigate[") and next_op.agent == op.agent:
                groups.append([i, i + 1])
                i += 2
                continue
        groups.append([i])
        i += 1

    result = list(runtime_ops)
    # Track which objects are held by which agent to preserve pick/place
    # consistency: the agent that picks an object must also place it.
    held_by: Dict[str, str] = {}

    for group in groups:
        action_idx = group[-1]
        action_op = runtime_ops[action_idx]
        action_str = action_op.action

        # Extract the primary target from the action string.
        target = None
        bracket_start = action_str.find("[")
        bracket_end = action_str.find("]")
        if bracket_start >= 0 and bracket_end > bracket_start:
            inner = action_str[bracket_start + 1:bracket_end]
            parts = [p.strip() for p in inner.split(",")]
            if parts and parts[0] != "None":
                target = parts[0]

        action_name = action_str[:bracket_start] if bracket_start >= 0 else action_str

        # If this agent is placing an object it picked up, it must be the
        # same agent that picked it — don't reassign.
        if action_name == "Place" and target and target in held_by:
            assigned = held_by.pop(target)
        elif target and target_to_room:
            try:
                assigned = pick_agent_for_target(
                    target, num_agents, target_to_room, restrictions, agent_loads,
                )
            except ValueError:
                assigned = action_op.agent
        else:
            assigned = action_op.agent

        # Track picks so the same agent does the corresponding place.
        if action_name == "Pick" and target:
            held_by[target] = assigned

        agent_loads[assigned] = agent_loads.get(assigned, 0) + 1

        for idx in group:
            old = result[idx]
            if old.agent != assigned:
                result[idx] = RuntimeOp(
                    agent=assigned,
                    action=old.action,
                    resources=old.resources,
                )

    return result


def generate_deterministic_trajectory(
    task_data: Dict[str, Any],
    scene_data: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build a deterministic golden trajectory from task spec.

    This is a solver-backed, non-LLM planner: same spec -> same trajectory.

    Args:
        task_data: Parsed task dict with problem_pddl, mechanic_bindings, etc.
        scene_data: SceneData object or dict with rooms/furniture/objects.

    Returns:
        Dict with keys: trajectory, planned_literals, ignored_literals, planner_notes.
    """
    from enacttom.pddl.dsl import Literal
    from enacttom.pddl.problem_pddl import replace_goal_in_problem_pddl
    from enacttom.pddl.runtime_projection import project_runtime_from_problem

    num_agents = int(task_data.get("num_agents", 2) or 2)
    problem_pddl = task_data.get("problem_pddl")
    if not isinstance(problem_pddl, str) or not problem_pddl.strip():
        raise ValueError(
            "Cannot generate deterministic golden trajectory: missing problem_pddl."
        )

    projection = project_runtime_from_problem(problem_pddl)
    if not projection.functional_goal_pddl:
        reasons = "; ".join(projection.invalid_reasons) or "no functional goal remains"
        raise ValueError(
            "Cannot generate deterministic golden trajectory from epistemic-only goal: "
            f"{reasons}."
        )

    runtime_task_data = dict(task_data)
    runtime_task_data["problem_pddl"] = replace_goal_in_problem_pddl(
        problem_pddl,
        projection.functional_goal_pddl,
    )

    problem, observability, solver_result = _solve_task_for_trajectory(
        runtime_task_data,
        scene_data,
    )

    relation_by_pair = _collect_goal_relation_preferences(problem.goal)
    object_types = dict(problem.objects)
    plan = solver_result.plan or []
    runtime_ops: List[RuntimeOp] = []
    runtime_state = extract_runtime_state(problem)
    ignored_epistemic_steps = 0

    def require_type(name: str, expected: str, step: str) -> None:
        actual = object_types.get(name)
        if actual != expected:
            raise RuntimeError(
                f"Planner step '{step}' uses '{name}' as {expected}, "
                f"but problem declares it as {actual or 'unknown'}."
            )

    def require_movable_object(name: str, step: str) -> None:
        actual = object_types.get(name)
        if actual != "object":
            raise RuntimeError(
                f"Planner step '{step}' tries to move '{name}', "
                f"but runtime only supports movable objects (declared type: {actual or 'unknown'})."
            )

    for step in plan:
        if _INFORM_RE.search(step):
            ignored_epistemic_steps += 1
            continue

        step_name, args = _parse_plan_step(step)
        if step_name.startswith("observe_knows_"):
            ignored_epistemic_steps += 1
            continue

        if step_name == "open" and len(args) == 3:
            require_type(args[0], "agent", step)
            require_type(args[1], "furniture", step)
            require_type(args[2], "room", step)
            append_runtime_ops(
                runtime_ops,
                args[0],
                f"Open[{args[1]}]",
                resources=[("furniture", args[1])],
                navigate_target=args[1],
            )
            continue
        if step_name == "close" and len(args) == 3:
            require_type(args[0], "agent", step)
            require_type(args[1], "furniture", step)
            require_type(args[2], "room", step)
            append_runtime_ops(
                runtime_ops,
                args[0],
                f"Close[{args[1]}]",
                resources=[("furniture", args[1])],
                navigate_target=args[1],
            )
            continue
        if step_name == "navigate" and len(args) == 2:
            require_type(args[0], "agent", step)
            require_type(args[1], "room", step)
            runtime_ops.append(RuntimeOp(agent=args[0], action=f"Navigate[{args[1]}]"))
            continue
        if step_name == "pick" and len(args) == 3:
            require_type(args[0], "agent", step)
            require_movable_object(args[1], step)
            require_type(args[2], "room", step)
            source_support = runtime_state.object_supports.get(args[1])
            resources = [("object", args[1])]
            if source_support:
                resources.append(("furniture", source_support))
            append_runtime_ops(
                runtime_ops,
                args[0],
                f"Pick[{args[1]}]",
                resources=resources,
                navigate_target=args[1],
            )
            runtime_state.object_supports.pop(args[1], None)
            runtime_state.held_by[args[1]] = args[0]
            continue
        if step_name == "place" and len(args) == 4:
            require_type(args[0], "agent", step)
            require_movable_object(args[1], step)
            require_type(args[2], "furniture", step)
            require_type(args[3], "room", step)
            relation = relation_by_pair.get((args[1], args[2]), "on")
            action = f"Place[{args[1]}, {relation}, {args[2]}, None, None]"
            append_runtime_ops(
                runtime_ops,
                args[0],
                action,
                resources=[("object", args[1]), ("furniture", args[2])],
                navigate_target=args[2],
            )
            runtime_state.held_by.pop(args[1], None)
            runtime_state.object_supports[args[1]] = args[2]
            continue
        if step_name == "wait" and len(args) == 1:
            require_type(args[0], "agent", step)
            runtime_ops.append(RuntimeOp(agent=args[0], action="Wait[]"))
            continue

        raise RuntimeError(
            f"Unsupported planner step '{step}'. Cannot translate to golden trajectory."
        )

    # Post-pass: rebalance agent assignments so work is distributed across
    # agents instead of collapsing to a single agent (the FD solver has no
    # preference for multi-agent distribution).
    runtime_ops = _rebalance_agent_assignments(
        runtime_ops, task_data, scene_data, num_agents,
    )

    trajectory = schedule_runtime_ops(num_agents, runtime_ops)

    planned_literals = sorted({
        node.to_pddl()
        for node in problem.goal.flatten()
        if isinstance(node, Literal)
    }) if problem.goal else []

    planner_notes = [
        "Deterministic golden trajectory derived from strict Fast Downward plan.",
        f"Translated {len(plan)} planner step(s) into {len(trajectory)} golden step(s).",
    ]
    if projection.epistemic_conjuncts_removed:
        planner_notes.append(
            f"Runtime golden semantics removed {projection.epistemic_conjuncts_removed} epistemic conjunct(s)."
        )
    if ignored_epistemic_steps:
        planner_notes.append(
            f"Ignored {ignored_epistemic_steps} epistemic planner step(s) while building the physical-only golden trajectory."
        )
    if solver_result.belief_depth:
        planner_notes.append(
            f"Strict solver proved belief depth {solver_result.belief_depth}."
        )

    return {
        "trajectory": trajectory,
        "planned_literals": planned_literals,
        "ignored_literals": [],
        "planner_notes": planner_notes,
        "communication_derived": False,
        "ignored_epistemic_steps": ignored_epistemic_steps,
        "functional_goal_pddl": projection.functional_goal_pddl,
    }


def regenerate_golden_trajectory(
    task_data: Dict[str, Any],
    scene_data: Optional[Any] = None,
    source: str = "unknown",
    task_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Regenerate golden trajectory deterministically and attach metadata.

    Mutates task_data in-place (sets golden_trajectory + golden_trajectory_metadata).
    Optionally persists to task_file.

    Args:
        task_data: Parsed task dict (mutated in-place).
        scene_data: SceneData object or dict.
        source: Label for what triggered regeneration (e.g. "verify", "submit").
        task_file: If provided, write updated task_data to this path.

    Returns:
        Dict with: spec_hash, trajectory_hash, num_steps, metadata.
    """
    plan_result = generate_deterministic_trajectory(task_data, scene_data)
    trajectory = plan_result["trajectory"]
    task_data["golden_trajectory"] = trajectory

    spec_hash = compute_task_spec_hash(task_data)
    trajectory_hash = hashlib.sha256(
        canonical_json(trajectory).encode("utf-8")
    ).hexdigest()

    metadata = task_data.get("golden_trajectory_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    communication_derived = plan_result.get("communication_derived", False)

    metadata.update({
        "planner": "strict_fd_translator",
        "planner_version": "v7_parallel_runtime",
        "source": source,
        "spec_hash": spec_hash,
        "trajectory_hash": trajectory_hash,
        "generated_at": datetime.now().isoformat(),
        "num_steps": len(trajectory),
        "communication_derived": communication_derived,
        "ignored_epistemic_steps": plan_result.get("ignored_epistemic_steps", 0),
        "functional_goal_pddl": plan_result.get("functional_goal_pddl"),
        "planned_literals": plan_result.get("planned_literals", []),
        "ignored_literals": plan_result.get("ignored_literals", []),
        "planner_notes": plan_result.get("planner_notes", []),
    })
    task_data["golden_trajectory_metadata"] = metadata

    if task_file:
        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)

    return {
        "spec_hash": spec_hash,
        "trajectory_hash": trajectory_hash,
        "num_steps": len(trajectory),
        "metadata": metadata,
    }
