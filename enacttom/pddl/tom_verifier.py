"""
Theory of Mind depth verifier.

Computes the minimum ToM depth required to solve a task by analyzing
the epistemic structure of the PDDL problem.

Inspired by DAEDALUS (Bolander et al., 2025) iterative deepening approach.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from enacttom.pddl.compiler import compile_task
from enacttom.pddl.domain import ENACTTOM_DOMAIN
from enacttom.pddl.epistemic import ObservabilityModel
from enacttom.pddl.fd_solver import FastDownwardSolver
from enacttom.pddl.solver import PDKBSolver, SolverResult, _max_epistemic_depth

if TYPE_CHECKING:
    from enacttom.task_gen.task_generator import GeneratedTask


def _detect_provider_for_model(model: str) -> str:
    """Detect provider for ad hoc ToM explanation calls."""
    normalized = (model or "").strip().lower()

    if normalized.startswith("gpt"):
        return "openai_chat"

    if normalized.startswith("us.anthropic.claude-"):
        return "bedrock_claude"

    if normalized.startswith("claude-"):
        return "anthropic_claude"

    if normalized in {
        "sonnet",
        "sonnet-4.5",
        "sonnet4.5",
        "haiku",
        "haiku-4.5",
        "haiku4.5",
        "opus",
        "opus-4.5",
        "opus4.5",
    }:
        if os.getenv("ANTHROPIC_API_KEY", "").strip():
            return "anthropic_claude"
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    if key.strip() == "ANTHROPIC_API_KEY" and value.strip().strip('"').strip("'"):
                        return "anthropic_claude"
            except Exception:
                pass
        return "bedrock_claude"

    return "openai_chat"


def compute_tom_depth(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
    max_depth: int = 3,
    solver_result: Optional[SolverResult] = None,
) -> int:
    """
    Compute the minimum Theory of Mind depth for a task.

    When *solver_result* is provided (e.g. from FastDownwardSolver with
    epistemic compilation), its ``belief_depth`` is used directly —
    this is the authoritative result.  Otherwise falls back to
    PDKBSolver's structural heuristic.

    ToM depth meanings:
    - 0: No belief reasoning needed (all information is shared)
    - 1: Agent must reason about what another agent knows/sees
         ("Agent B knows where the key is")
    - 2: Agent must reason about what another agent thinks a third knows
         ("Agent A thinks Agent B believes the safe is on the left")
    - 3: Third-order nesting

    Args:
        task: The generated task
        scene_data: Optional scene data for object resolution
        max_depth: Maximum depth to check (default 3)
        solver_result: Optional pre-computed SolverResult from FD

    Returns:
        Minimum ToM depth (0-3), or -1 if unsolvable at any depth
    """
    if solver_result is not None:
        if not solver_result.solvable:
            return -1
        return solver_result.belief_depth

    proof = prove_minimal_tom_level(task, scene_data=scene_data, max_depth=max_depth)
    return proof["tom_level"]


def prove_minimal_tom_level(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
    max_depth: int = 3,
    strict: bool = True,
) -> Dict[str, Any]:
    """Prove the minimum supported ToM depth by iterative depth-bounded solving."""
    problem = compile_task(task, scene_data)
    observability = ObservabilityModel.from_task_with_scene(task, scene_data)
    solver = FastDownwardSolver() if strict else PDKBSolver()
    epistemic_goal_depth = _max_epistemic_depth(problem.goal) if problem.goal else 0

    attempts = []
    trivial_goals = []
    first_success: Optional[SolverResult] = None
    minimal_level = -1

    start_level = epistemic_goal_depth if strict and epistemic_goal_depth > 0 else 0

    for level in range(start_level, max_depth + 1):
        if strict:
            result = solver.solve(
                ENACTTOM_DOMAIN,
                problem,
                observability,
                max_belief_depth=level,
                strict=True,
            )
        else:
            result = solver.solve(
                ENACTTOM_DOMAIN,
                problem,
                observability,
                max_belief_depth=level,
            )

        attempts.append(
            {
                "level": level,
                "solvable": result.solvable,
                "belief_depth": result.belief_depth,
                "error": result.error,
            }
        )
        if result.trivial_k_goals:
            trivial_goals = result.trivial_k_goals
        if result.solvable:
            minimal_level = level
            first_success = result
            break

    return {
        "tom_level": minimal_level,
        "minimal_tom_level": minimal_level,
        "epistemic_goal_depth": epistemic_goal_depth,
        "proved_unsat_below": (
            list(range(start_level, minimal_level))
            if minimal_level >= 0
            else list(range(start_level, max_depth + 1))
        ),
        "proof_attempts": attempts,
        "proof_backend": "fast_downward_strict" if strict else "pdkb_structural",
        "proof_strict": strict,
        "solver_result": first_success,
        "trivial_k_goals": trivial_goals,
    }


def explain_tom_depth(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
    solver_result: Optional[SolverResult] = None,
    proof: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Explain why a task requires a specific ToM depth.

    When *solver_result* is provided (e.g. from FastDownwardSolver with
    epistemic compilation), its ``belief_depth`` and ``trivial_k_goals``
    are used directly instead of re-solving via PDKBSolver.

    Returns:
        Dict with:
        - tom_level: int (0-3)
        - tom_reasoning: str explaining why
        - information_gaps: list of asymmetries
        - communication_required: bool
        - trivial_k_goals: list of trivially satisfied K() goals (if any)
    """
    if proof is None and solver_result is None:
        proof = prove_minimal_tom_level(task, scene_data=scene_data, strict=True)
    if proof is not None:
        depth = proof["tom_level"]
        if solver_result is None:
            solver_result = proof.get("solver_result")
    else:
        depth = compute_tom_depth(task, scene_data, solver_result=solver_result)
    observability = ObservabilityModel.from_task_with_scene(task, scene_data)

    # Analyze information gaps
    gaps = []
    for agent, rooms in observability.restricted_rooms.items():
        gaps.append(f"{agent} cannot see rooms: {sorted(rooms)}")
    for trigger, agents in observability.hidden_effects.items():
        gaps.append(f"Effect of {trigger} hidden from: {sorted(agents)}")

    # Determine if communication is required
    comm_required = bool(observability.restricted_rooms or observability.hidden_effects)

    # Get trivial K() goals from solver result or fallback
    trivial_goals = []
    if solver_result is not None:
        trivial_goals = solver_result.trivial_k_goals or []
    elif proof is not None:
        trivial_goals = proof.get("trivial_k_goals", [])

    # Build reasoning explanation
    if depth == 0:
        reasoning = (
            "The task is solvable without any epistemic reasoning or communication. "
            "A purely physical plan succeeds at depth 0."
        )
    elif depth == 1:
        reasoning = (
            f"Task includes first-order knowledge goals and the strict proof succeeds at depth 1. "
            f"Gaps: {'; '.join(gaps) if gaps else 'private information is encoded in the task'}."
        )
    elif depth == 2:
        reasoning = (
            f"Task includes second-order nested knowledge goals and the strict proof succeeds at depth 2. "
            f"Agents must model what others think about third parties' knowledge. "
            f"Gaps: {'; '.join(gaps)}."
        )
    elif depth == 3:
        reasoning = (
            f"Task includes third-order nested knowledge goals and the strict proof succeeds at depth 3. "
            f"Complex nested beliefs about others' models of others. "
            f"Gaps: {'; '.join(gaps)}."
        )
    else:
        reasoning = f"Task appears unsolvable at belief depth <= 3."

    if trivial_goals:
        reasoning += (
            f" WARNING: {len(trivial_goals)} K() goal(s) are trivially satisfied "
            f"(agent can directly observe the fact)."
        )

    return {
        "tom_level": depth,
        "minimal_tom_level": depth,
        "epistemic_goal_depth": proof["epistemic_goal_depth"] if proof is not None else depth,
        "tom_reasoning": reasoning,
        "information_gaps": gaps,
        "communication_required": comm_required,
        "trivial_k_goals": trivial_goals,
        "proof_attempts": proof["proof_attempts"] if proof is not None else [],
        "proof_backend": proof["proof_backend"] if proof is not None else "solver_result",
        "proof_strict": proof["proof_strict"] if proof is not None else True,
        "proved_unsat_below": proof["proved_unsat_below"] if proof is not None else [],
    }


def generate_tom_reasoning(
    task_data: dict,
    tom_level: int,
    information_gaps: list[str],
    model: str = "gpt-5.2",
) -> str:
    """
    Use an LLM to generate a clear explanation of why a task requires its ToM level.

    The tom_level is computed programmatically (unchanged); this function only
    explains *why* that level is needed, naming specific agents, facts, and rooms.

    Args:
        task_data: Full task JSON dict.
        tom_level: Programmatically computed ToM level (1-3).
        information_gaps: List of observability asymmetries from explain_tom_depth().
        model: LLM model name (default "gpt-5.2").

    Returns:
        LLM-generated explanation string.

    Raises:
        RuntimeError: If the LLM call fails or returns an unusable response.
    """
    # Build context for the LLM
    task_desc = task_data.get("task", "")
    problem_pddl = task_data.get("problem_pddl", "")
    agent_secrets = task_data.get("agent_secrets", {})
    mechanic_bindings = task_data.get("mechanic_bindings", [])
    message_targets = task_data.get("message_targets")
    category = task_data.get("category", "cooperative")
    num_agents = task_data.get("num_agents", 2)

    # Build mechanic summary
    mechanic_lines = []
    for b in mechanic_bindings:
        mtype = b.get("mechanic_type", "unknown")
        agents = b.get("for_agents", [])
        if mtype == "room_restriction":
            rooms = b.get("restricted_rooms", [])
            mechanic_lines.append(f"- {', '.join(agents)} cannot enter: {', '.join(rooms)}")
        elif mtype == "limited_bandwidth":
            limit = b.get("message_limit", "?")
            mechanic_lines.append(f"- {', '.join(agents)} limited to {limit} message(s)")
        elif mtype == "restricted_communication":
            mechanic_lines.append(f"- {', '.join(agents)} have restricted communication targets")
        else:
            mechanic_lines.append(f"- {mtype} applies to {', '.join(agents)}")

    # Build secrets summary
    secrets_lines = []
    for agent, secrets in agent_secrets.items():
        if isinstance(secrets, list):
            secrets_lines.append(f"- {agent}: {'; '.join(secrets)}")

    # Communication graph
    comm_graph = ""
    if message_targets:
        comm_graph = f"\nCommunication graph: {json.dumps(message_targets)}"

    prompt = f"""You are analyzing a multi-agent Theory of Mind (ToM) task. The task's ToM level has been computed as {tom_level}.

ToM levels:
- Level 1: An agent must reason about what another agent knows or doesn't know. ("I know that agent_1 can't see the kitchen, so I need to tell them what's there.")
- Level 2: An agent must reason about what another agent THINKS a third agent knows. ("Agent_0 needs agent_1 to know that agent_2 has seen the object.")
- Level 3: Third-order nested beliefs about others' models of others.

## Task
Category: {category}
Agents: {num_agents}
Description: {task_desc}

## PDDL Goals
{problem_pddl}

## Agent Secrets
{chr(10).join(secrets_lines) if secrets_lines else "None"}

## Mechanics
{chr(10).join(mechanic_lines) if mechanic_lines else "None"}{comm_graph}

## Information Gaps
{chr(10).join('- ' + g for g in information_gaps) if information_gaps else "None"}

## Instructions
Explain in 2-3 sentences WHY this task requires ToM level {tom_level}. Be specific: name the agents, the key facts they need, and the rooms involved. Explain the belief chain — who needs to know what, and why they can't get that knowledge directly.

Do NOT start with "This task requires ToM level N because..." — just explain the reasoning directly."""

    from habitat_llm.llm import instantiate_llm

    provider = _detect_provider_for_model(model)

    llm = instantiate_llm(
        provider,
        generation_params={
            "model": model,
            "temperature": 0.0,
            "max_tokens": 300,
        },
    )

    response = llm.generate(prompt).strip()
    if not response or len(response) < 20:
        raise RuntimeError(f"LLM returned unusable tom_reasoning: {response!r}")
    return response
