#!/usr/bin/env python3
"""
Category-aware Task Judge for EnactToM task validation.

Evaluates tasks using category-specific criteria:
- Cooperative: agent necessity, secrets, interdependence
- Mixed: agent necessity, secrets, subgoal tension

Priority criteria (evaluated first, fixes prioritized):
- agent_necessity: Every agent must be indispensable
- secret_relevance: Secrets must be required for task completion

Uses `gpt-5.4-mini` by default.

Usage:
    # CLI
    python -m enacttom.task_gen.judge --task <path>
    python -m enacttom.task_gen.judge --task <path> --models gpt-5.4-mini

    # Programmatic
    from enacttom.task_gen.judge import Judge
    judge = Judge(models=["gpt-5.4-mini"])
    verdict = judge.evaluate(task_data, scene_data)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .authoring_surface import (
    AUTHORING_CONSTRAINTS_NOTICE,
    format_supported_mechanics,
    get_authoring_action_descriptions,
    get_authoring_mechanics,
    get_authoring_predicates,
)

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM
    from .task_generator import GeneratedTask
    from .scene_loader import SceneData
    from .diversity import DiversityTracker


# ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _detect_provider_for_model(model: str) -> str:
    """Detect provider for judge-side ad hoc LLM calls."""
    normalized = (model or "").strip().lower()

    if normalized.startswith("gpt"):
        return "openai_chat"

    if normalized.startswith("accounts/fireworks/models/"):
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

    if normalized.startswith("kimi"):
        return "openai_chat"

    return "openai_chat"


def _resolve_client_model_name(model: str) -> str:
    """Expand lightweight aliases into provider-native model names."""
    normalized = (model or "").strip().lower()
    if normalized == "kimi-k2.5" and os.getenv("FIREWORKS_API_KEY", "").strip():
        return "accounts/fireworks/models/kimi-k2p5"
    return model


@dataclass
class CriterionScore:
    """Score for a single evaluation criterion."""
    score: float  # 0.0 to 1.0, or -1.0 for automatic fail
    reasoning: str


@dataclass
class BenchmarkRollout:
    """Data from a benchmark test run."""
    success: bool
    steps: int
    turns: int
    percent_complete: float
    action_history: List[Dict[str, Any]]
    subtask_status: Dict[str, bool]
    agent_traces: Dict[str, str]  # agent_id -> trace text
    snapshot_spec_hash: Optional[str] = None
    snapshot_task: Optional[str] = None

    @classmethod
    def from_trajectory_dir(cls, trajectory_dir: Path) -> Optional["BenchmarkRollout"]:
        """Load rollout data from trajectory directory."""
        if not trajectory_dir.exists():
            return None

        result_file = trajectory_dir / "result.txt"
        if not result_file.exists():
            return None

        # Parse result.txt
        success = False
        steps = 0
        turns = 0
        percent_complete = 0.0
        action_history = []
        subtask_status = {}

        try:
            content = result_file.read_text()
            for line in content.split("\n"):
                if line.startswith("Success:"):
                    success = "True" in line
                elif line.startswith("Steps:"):
                    steps = int(line.split(":")[1].strip())
                elif line.startswith("Turns:"):
                    turns = int(line.split(":")[1].strip())
                elif line.startswith("Percent Complete:"):
                    pct = line.split(":")[1].strip().replace("%", "")
                    percent_complete = float(pct) / 100
                elif ":" in line and line.strip().startswith("s"):
                    # Subtask status line like "s1_...: COMPLETE"
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        subtask_status[parts[0].strip()] = "COMPLETE" in parts[1]
        except Exception:
            pass

        # Load agent traces
        agent_traces = {}
        for trace_file in trajectory_dir.glob("agent_*.txt"):
            agent_id = trace_file.stem
            try:
                agent_traces[agent_id] = trace_file.read_text()
            except Exception:
                pass

        # Load snapshot metadata when present (used to detect stale rollouts).
        snapshot_spec_hash = None
        snapshot_task = None
        snapshot_file = trajectory_dir / "task_snapshot.json"
        if snapshot_file.exists():
            try:
                snapshot = json.loads(snapshot_file.read_text())
                raw_hash = snapshot.get("spec_hash")
                if isinstance(raw_hash, str) and raw_hash.strip():
                    snapshot_spec_hash = raw_hash.strip()
                raw_task = snapshot.get("task")
                if isinstance(raw_task, str) and raw_task.strip():
                    snapshot_task = raw_task.strip()
            except Exception:
                pass

        return cls(
            success=success,
            steps=steps,
            turns=turns,
            percent_complete=percent_complete,
            action_history=action_history,
            subtask_status=subtask_status,
            agent_traces=agent_traces,
            snapshot_spec_hash=snapshot_spec_hash,
            snapshot_task=snapshot_task,
        )


# =============================================================================
# Category Configuration
# =============================================================================

# Shared quality criteria (apply to ALL categories)
# Ordered by importance - agent design criteria first
SHARED_CRITERIA = [
    "agent_necessity",       # Every agent must be essential
    "secret_quality",        # Secrets are actionable, natural, and non-leaking
    "task_naturalness",      # Task description uses natural language, not object IDs
    "narrative_consistency",
    "goal_relevance",        # Renamed from subtask_relevance for PDDL
    "mechanic_utilization",
    "pddl_solvability",     # PDDL goal solvability and ToM depth
]

# Category-specific criteria
CATEGORY_CRITERIA = {
    "cooperative": SHARED_CRITERIA + ["task_interdependence"],
    "mixed": SHARED_CRITERIA + ["subgoal_tension"],
}

# Criteria descriptions for prompts
CRITERIA_DESCRIPTIONS = {
    # Core agent design criteria (most important)
    "agent_necessity": {
        "name": "Agent Necessity",
        "description": "Does every agent make a distinct, goal-relevant contribution? Judge based on whether each agent materially changes the optimal plan or reachable success path, not on a literal proof that removing one agent makes success impossible in all circumstances.",
        "rubric": """0.0: One or more agents are idle, decorative, or obviously removable
0.3: Some agents act, but their contribution is trivial or easily absorbed by others with no real plan change
0.5: All agents participate and at least one extra agent matters somewhat, but one or more roles are still weak, redundant, or mostly relay-only
0.7: Every agent makes a material, distinct contribution; removing one would significantly weaken the task or collapse an intended dependency, even if a convoluted fallback might still exist
1.0: Every agent has a clear non-substitutable role in the intended solution; removing any agent would fundamentally break a required dependency, access path, or incentive structure""",
    },
    "secret_quality": {
        "name": "Secret Quality",
        "description": "Secrets must state only positive private facts, constraints, and private objectives. They must NEVER prescribe communication strategy, encode missing knowledge as 'you do not know ...', or coach the agent with epistemic directives like 'By the end, you must be confident ...'. Exact IDs are appropriate only for agents who already know or observed that fact.",
        "rubric": """0.0: Secrets prescribe the relay chain, include ignorance lines like 'you do not know ...', or include epistemic coaching like 'By the end, you must be confident ...'
0.3: Secrets hint at coordination strategy, contain boilerplate/self-intro text, or include other non-actionable task coaching
0.5: Secrets state mostly useful facts and constraints, but still include some leakage, redundancy, or overly directive phrasing
0.7: Secrets are mostly clean and actionable, with only minor redundancy or wording issues
1.0: Secrets are minimal and precise — only positive private facts, constraints, and private objectives, with exact IDs reserved for agents who genuinely know those facts.""",
    },
    "task_naturalness": {
        "name": "Public/Secret Grounding Split",
        "description": "Does the public `task` stay high-level while `agent_secrets` provide the precise private grounding? The public task should not read like a machine spec, reveal hidden target object IDs, or expose epistemic K() goals (e.g. 'must know', 'knows that'). K() requirements belong only in `problem_pddl`. Secrets should carry exact IDs/states only for the agents who genuinely know those facts.",
        "rubric": """0.0: Public task leaks exact hidden target IDs or other machine-style targets, and secrets are still vague or generic
0.3: Either the public task is overly specific, or the secrets leak hidden object IDs to agents who are supposed to lack that information
0.5: Split is partly right, but public task still over-specifies some targets or secrets are inconsistent in how they reveal private grounding
0.7: Public task is mostly high-level and secrets are usually explicit, with only minor leakage or ambiguity
1.0: Public task is clean, high-level, and non-leaking; secrets carry the precise actionable grounding only where that private knowledge belongs""",
    },
    # Task quality criteria
    "narrative_consistency": {
        "name": "Narrative Consistency",
        "description": "Does the task description accurately describe what agents must do?",
        "rubric": """0.0: Description misleading or unrelated to actual subtasks
0.3: Major discrepancies (mentions goals not in subtasks, or misses key objectives)
0.5: Partial match (captures main idea but omits or misrepresents details)
0.7: Good match with minor omissions
1.0: Perfect - description precisely matches subtask objectives""",
    },
    "goal_relevance": {
        "name": "Goal Relevance",
        "description": "Does every shared PDDL goal conjunct materially contribute to the benchmarked task, rather than acting as filler or decoration?",
        "rubric": """0.0: Goal is mostly filler or arbitrary conjuncts with no coherent task objective
0.3: Several conjuncts are decorative, redundant, or weakly related to the main objective
0.5: Core objective is present but some conjuncts could be removed without changing the task much
0.7: Most conjuncts are essential, with at most one questionable or weakly motivated goal
1.0: Every conjunct materially contributes to the task objective; removing any would meaningfully change the task""",
    },
    "pddl_solvability": {
        "name": "Formal Goal Quality & Epistemic Coherence",
        "description": "Does the self-contained `problem_pddl` define a formally meaningful task for this benchmark under the current split semantics? Treat hard formal invalidity as a near-automatic fail. Judge the physical functional core after removing epistemic conjuncts, and separately judge whether the K() structure yields meaningful literal-ToM probes grounded in genuine information asymmetry. Strong scores require FUNCTIONAL ToM pressure: success should depend on choosing actions based on partner-specific private information, not just relaying hidden facts.",
        "rubric": """0.0: Raw problem_pddl is invalid, contradictory, or impossible; or the functional projection becomes vacuous/single-agent/trivial; or K() goals are fake/decorative
0.3: Barely benchmark-meaningful: weak functional core, shaky category logic, or K() probes are loosely attached to irrelevant facts / pure relay events
0.5: Formally valid task, but either the functional projection is weak after dropping K(), or the epistemic structure is mostly literal hidden-fact reporting rather than partner-dependent action choice
0.7: Strong functional core plus mostly meaningful K()-derived probes, with some genuine partner-modeling pressure and only minor weaknesses
1.0: Self-contained, formally coherent, and benchmark-meaningful under split semantics: the functional projection remains strong, and the task requires adapting to partner-specific knowledge, access, incentives, or communication limits rather than merely forwarding facts""",
    },
    "mechanic_utilization": {
        "name": "Mechanic Utilization & Balance",
        "description": "Are the listed mechanics genuinely used to create the intended coordination or ToM challenge? Penalize mechanics that are redundant, decorative, or disconnected from the goal structure. Empty mechanics are fine when the task is intentionally simple.",
        "rubric": """0.0: Mechanics are listed but unused, misleading, or disconnected from the actual task
0.3: Mechanics appear in the spec but most could be removed without changing the challenge
0.5: Mechanics matter somewhat, but at least one is decorative or only weakly connected to the benchmark challenge
0.7: Mechanics are well integrated and each serves a distinct purpose, with only minor redundancy
1.0: Mechanics are tightly integrated with the formal goals and task design; each one materially contributes to the benchmark challenge""",
    },
    # Cooperative-specific
    "task_interdependence": {
        "name": "Task Interdependence",
        "description": "Do agents genuinely NEED information from each other to complete physical goals? At least one physical goal must be information-dependent: an agent cannot determine WHAT to do or WHERE to act without receiving a message from another agent. If all physical goals can be completed in parallel without any communication, score 0.",
        "rubric": """0.0: All physical goals are independently solvable — agents can complete everything in parallel without communicating
0.3: Agents help but aren't required, or interdependence is just a hidden-fact relay with no impact on physical goal completion
0.5: Some interdependence but key physical steps can be done solo or without modeling which teammate is best positioned
0.7: Strong interdependence with partner-specific dependencies, though some steps are still generic handoffs
1.0: Impossible for any single agent to succeed, and rational success depends on choosing actions based on what specific teammates know, can access, or are likely to prioritize""",
    },
    # Mixed-specific
    "subgoal_tension": {
        "name": "Subgoal Tension",
        "description": "Does every agent have a personal objective in `:goal-owners`, and do they create meaningful tension that changes how teammates should coordinate with them?",
        "rubric": """0.0: No personal objectives in :goal-owners, or goals trivially satisfied
0.3: Some agents have personal objectives but they do not affect partner expectations or coordination
0.5: Personal objectives exist for most agents with minor tension, but teammates can mostly ignore them
0.7: Every agent has a personal objective; meaningful conflicts require strategic choices about who to trust, inform, or rely on
1.0: Every agent has a personal objective; real dilemmas where pursuing them risks the main goal, and success depends on modeling which teammate is likely to deviate or cooperate""",
    },
    # User requirements (added dynamically when query is provided)
    "user_requirements_alignment": {
        "name": "User Requirements Alignment",
        "description": "Does the task align with the user's specific request/query?",
        "rubric": """0.0: Task completely ignores user's request (wrong items, wrong mechanics, wrong theme)
0.3: Task vaguely relates but misses the key elements requested
0.5: Task partially addresses the request but missing important aspects
0.7: Task mostly aligns with minor omissions
1.0: Task fully incorporates what the user requested""",
    },
    # Task novelty (added dynamically when diversity tracker is provided)
    "task_novelty": {
        "name": "Task Novelty",
        "description": "Is this task structurally different from existing tasks in the dataset?",
        "rubric": """0.0: Nearly identical to an existing task (same structure, just different items/names)
0.3: Very similar to existing task(s), feels like a reskin
0.5: Shares significant structural elements with existing tasks
0.7: Mostly novel with some minor similarities to existing patterns
1.0: Completely novel structure, nothing similar exists in the dataset""",
    },
}


@dataclass
class Judgment:
    """Result of task evaluation by a single model."""

    category: str  # Task category that was evaluated
    criteria_scores: Dict[str, CriterionScore]  # Dynamic based on category
    overall_score: float
    is_valid: bool
    overall_reasoning: str
    required_fixes: List[str] = field(default_factory=list)

    # Optional rollout-based assessment
    rollout_assessment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        criteria = {}
        for name, score in self.criteria_scores.items():
            criteria[name] = {"score": score.score, "reasoning": score.reasoning}

        return {
            "category": self.category,
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "criteria": criteria,
            "overall_reasoning": self.overall_reasoning,
            "required_fixes": self.required_fixes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class CouncilVerdict:
    """Aggregated verdict from configured judge model(s)."""

    judgments: Dict[str, Judgment]  # model -> judgment
    passed: bool  # True only if ALL models pass
    overall_score: float  # Average of all model scores
    required_fixes: List[str]  # Merged from all models, deduplicated
    disagreements: List[str] = field(default_factory=list)  # Where models disagreed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "model_judgments": {
                model: j.to_dict() for model, j in self.judgments.items()
            },
            "required_fixes": self.required_fixes,
            "disagreements": self.disagreements,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Category-aware evaluation prompt template
EVALUATION_PROMPT = """You are an expert evaluator for multi-agent tasks.

## Task Category: {category}
{category_description}
{difficulty_section}
{user_requirements_section}

## System Capabilities (use these in required fixes)
### Available Actions
{available_actions}
### Available Mechanics
{available_mechanics}
### Authoring Surface
{authoring_constraints}
### Available Predicates (for success_condition)
{available_predicates}
### Scene Objects
{scene_objects}

## Checks
- `task` is GLOBAL; for mixed tasks it must not leak secret targets or agent-specific objectives
- Secrets must be actionable positive facts or constraints, not missing-knowledge reminders or epistemic coaching
- Secrets must be explicit and actionable, naming exact IDs/states only for agents who already know those facts, and not step-by-step
- Penalize exact-ID leakage: if the public `task` or a secret for an uninformed agent names the exact hidden object ID, the task is too revealing
- Penalize banned secret styles: 'you do not know ...', 'By the end, you must be confident ...', 'Epistemic probe: ...', or self-intro boilerplate
- Single-format goal source is `problem_pddl`
- Runtime semantics are split:
  - functional benchmark success uses the non-epistemic projection only
  - `K()` goals are design-time / probe-time only and become end-of-episode literal-ToM probes
- Category intent must be reflected in `problem_pddl` objective structure
- **Mechanic consistency**: Every mechanic referenced in `task` or `agent_secrets` (e.g., "the handle is reversed", "you have limited messages") MUST have a corresponding entry in `mechanic_bindings`. `message_targets` is a valid standalone way to encode communication restrictions and does not require a duplicate `restricted_communication` binding.
{formal_checks_section}

## Derived Runtime View
Use this derived runtime view when judging split-semantics quality.
{runtime_semantics_section}

## Secret Text Heuristics
Use these concrete leakage and secret-style checks as hard evidence for `secret_quality` and `task_naturalness`.
{id_leakage_section}

{compiled_formal_view_block}

## Benchmark Comparison
Use this when calibration data exists.
{benchmark_comparison_section}

## Evaluation Criteria
Score each criterion from 0.0 to 1.0.
{criteria_section}

## Task to Evaluate
```json
{task_json}
```

## Response Format
Respond with ONLY valid JSON. Keep reasoning brief (under 15 words each).
{{
{response_format}
  "overall_reasoning": "<1 sentence>",
  "required_fixes": ["<minimum concrete fix required to pass>", "..."]
}}

## Required Fix Rules
- List only the minimum concrete changes required for this task to pass.
- Do NOT include optional improvements, stretch ideas, or alternate redesigns.
- Prefer 1-3 fixes total.
- Preserve the current scene, category, and main objects/goals when possible.
- Be specific and only use available system capabilities.
- Prefer fixes that strengthen the functional projection after dropping K(), make K()-derived probes more grounded and informative without making them runtime success conditions, and increase partner-dependent action choice rather than fact relay.
- If the task already passes, return `"required_fixes": []`.
"""

# Category descriptions for the prompt
CATEGORY_PROMPT_DESCRIPTIONS = {
    "cooperative": """**COOPERATIVE** - All agents united toward shared goals
- Every agent contributes unique knowledge, skills, or access that others lack
- Information is distributed: one agent might know key locations, another may have restricted access
- Success requires piecing together distributed information through communication
- Complex tasks can have parallel workstreams that converge
- Uses `agent_secrets` to distribute knowledge and shared objective in `problem_pddl`""",
    "mixed": """**MIXED** - Cooperation with hidden personal objectives
- All agents share a main objective in `problem_pddl :goal`
- Each agent MUST have a personal objective in `:goal-owners` (supplementary, not part of :goal)
- Personal objectives create tension: they may conflict with the main goal or with other agents' objectives
- `agent_secrets` should hint at each agent's personal objective in natural language
- Public `task` must describe ONLY the shared objective; personal objectives belong in secrets""",
}


DIFFICULTY_DESCRIPTIONS = {
    "easy": """## Intended Difficulty: EASY
This task is designed for WEAKER models. Calibrate your evaluation accordingly:
- **Agent necessity**: 2-3 agents with clear, distinct roles is sufficient. Simple role division (e.g., one agent fetches, another places) counts as high agent necessity.
- **Task interdependence / goal opposition / subgoal tension**: Simple dependencies are fine. One clear handoff or information exchange between agents is enough.
- **Secret quality**: Secrets should state constraints, roles, and goals precisely. Mechanic hints (e.g., "the handle is reversed" for inverse_state) are required. But secrets must NEVER prescribe coordination strategy or leak hidden object IDs to agents who lack that information. Score LOW if secrets tell agents HOW to coordinate.
- **Mechanic utilization**: Using 0-1 mechanics is sufficient. Prefer simple, observable mechanics. Avoid stacking multiple mechanics.
- **Overall**: A well-structured simple task with clear agent roles, mechanic hints in secrets, and basic ToM should score HIGH. Do NOT penalize simplicity.""",
    "medium": """## Intended Difficulty: MEDIUM
This task targets mid-tier models. Standard evaluation applies:
- Agents should have meaningful distinct roles with some interdependence.
- Secrets should require reasoning to use effectively.
- Tasks should use mechanics appropriately (typically 2-4).
- Moderate complexity in coordination is expected.""",
    "hard": """## Intended Difficulty: HARD — Must defeat gpt-5.4-mini
This task must be difficult enough that gpt-5.4-mini CANNOT solve it. Apply strict standards:
- **Agent necessity**: Each agent MUST hold unique information. Score LOW if any agent is removable.
- **Task interdependence / goal opposition / subgoal tension**: Require information relay chains. Score LOW unless at least one goal depends on relayed (not directly observed) information.
- **Secret quality**: Secrets must state only constraints, private facts, and abstract epistemic goals. NEVER prescribe relay chains, communication strategy, or what to tell other agents. Score 0 if any secret says "tell agent_X", "forward to agent_X", includes parenthetical strategy hints, or leaks the hidden object ID to an agent who is explicitly missing that information.
- **Mechanic utilization**: limited_bandwidth MUST be present with 1 message per agent. 1-2 mechanics total is fine — complexity should come from ToM reasoning, not mechanic stacking. Score LOW if bandwidth > 1 per agent.
- **Overall**: The task should require genuine Theory of Mind reasoning. Reward tasks where agents must infer what others know. Do NOT require complex mechanics — difficulty from information asymmetry is preferred.""",
}


def _extract_typed_problem_objects(task_dict: Dict[str, Any], target_type: str) -> Set[str]:
    """Extract typed names from the `:objects` section of problem_pddl."""
    problem_pddl = str(task_dict.get("problem_pddl", "") or "")
    match = re.search(r"\(:objects(?P<body>[\s\S]*?)\)\s*\(:init", problem_pddl, re.IGNORECASE)
    if not match:
        return set()

    tokens = match.group("body").split()
    names: Set[str] = set()
    pending: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-" and index + 1 < len(tokens):
            if tokens[index + 1] == target_type:
                names.update(pending)
            pending = []
            index += 2
            continue
        pending.append(token)
        index += 1
    return names


def _analyze_id_leakage(task_dict: Dict[str, Any]) -> Dict[str, List[str]]:
    """Find exact-ID leakage and banned secret text patterns."""
    object_ids = _extract_typed_problem_objects(task_dict, "object")
    if not object_ids:
        object_ids = set()

    task_text = str(task_dict.get("task", "") or "")
    public_task_object_ids = sorted(obj_id for obj_id in object_ids if obj_id in task_text)

    ignorance_secret_ids: Set[str] = set()
    ignorance_secret_lines: List[str] = []
    epistemic_prompt_lines: List[str] = []
    boilerplate_secret_lines: List[str] = []
    for secrets in (task_dict.get("agent_secrets") or {}).values():
        if not isinstance(secrets, list):
            continue
        for secret in secrets:
            if not isinstance(secret, str):
                continue
            secret_lower = secret.lower()
            if (
                "do not know" in secret_lower
                or "don't know" in secret_lower
            ):
                ignorance_secret_lines.append(secret)
                for obj_id in object_ids:
                    if obj_id in secret:
                        ignorance_secret_ids.add(obj_id)
            if (
                "by the end, you must be confident" in secret_lower
                or "by the end, you should be confident" in secret_lower
                or secret_lower.startswith("epistemic probe:")
                or secret_lower.startswith("one success condition is:")
            ):
                epistemic_prompt_lines.append(secret)
            if re.fullmatch(
                r"\s*you are agent_\d+\. shared objective is the public task\.?\s*",
                secret_lower,
            ):
                boilerplate_secret_lines.append(secret)

    return {
        "public_task_object_ids": public_task_object_ids,
        "ignorance_secret_ids": sorted(ignorance_secret_ids),
        "ignorance_secret_lines": ignorance_secret_lines,
        "epistemic_prompt_lines": epistemic_prompt_lines,
        "boilerplate_secret_lines": boilerplate_secret_lines,
    }


def _build_id_leakage_section(task_dict: Dict[str, Any]) -> str:
    """Format exact-ID leakage heuristics for the judge prompt."""
    leakage = _analyze_id_leakage(task_dict)
    lines: List[str] = []

    public_task_ids = leakage["public_task_object_ids"]
    if public_task_ids:
        lines.append(
            "- Public task names exact object IDs: "
            + ", ".join(public_task_ids)
        )

    ignorance_secret_ids = leakage["ignorance_secret_ids"]
    if ignorance_secret_ids:
        lines.append(
            "- Ignorance secrets still reveal exact object IDs: "
            + ", ".join(ignorance_secret_ids)
        )

    ignorance_secret_lines = leakage["ignorance_secret_lines"]
    if ignorance_secret_lines:
        lines.append(
            f"- Secrets still contain ignorance text ({len(ignorance_secret_lines)} line(s))"
        )

    epistemic_prompt_lines = leakage["epistemic_prompt_lines"]
    if epistemic_prompt_lines:
        lines.append(
            f"- Secrets still contain epistemic coaching ({len(epistemic_prompt_lines)} line(s))"
        )

    boilerplate_secret_lines = leakage["boilerplate_secret_lines"]
    if boilerplate_secret_lines:
        lines.append(
            f"- Secrets still contain boilerplate/self-intro text ({len(boilerplate_secret_lines)} line(s))"
        )

    if not lines:
        return "- No obvious secret leakage or banned secret-text patterns detected by heuristic checks."
    return "\n".join(lines)


def _apply_id_leakage_penalties(
    judgment: Judgment,
    task_dict: Dict[str, Any],
    overall_threshold: float,
    min_criterion_threshold: float,
) -> Judgment:
    """Downgrade judge scores when hidden-information text leaks exact object IDs."""
    leakage = _analyze_id_leakage(task_dict)
    public_task_ids = leakage["public_task_object_ids"]
    ignorance_secret_ids = leakage["ignorance_secret_ids"]
    ignorance_secret_lines = leakage["ignorance_secret_lines"]
    epistemic_prompt_lines = leakage["epistemic_prompt_lines"]
    boilerplate_secret_lines = leakage["boilerplate_secret_lines"]
    if (
        not public_task_ids
        and not ignorance_secret_ids
        and not ignorance_secret_lines
        and not epistemic_prompt_lines
        and not boilerplate_secret_lines
    ):
        return judgment

    if public_task_ids and "task_naturalness" in judgment.criteria_scores:
        current = judgment.criteria_scores["task_naturalness"]
        judgment.criteria_scores["task_naturalness"] = CriterionScore(
            score=min(current.score, 0.3),
            reasoning=(
                f"{current.reasoning} Public task leaks exact object IDs: "
                + ", ".join(public_task_ids)
            ).strip(),
        )

    if (
        ignorance_secret_ids
        or ignorance_secret_lines
        or epistemic_prompt_lines
        or boilerplate_secret_lines
    ) and "secret_quality" in judgment.criteria_scores:
        current = judgment.criteria_scores["secret_quality"]
        reasons: List[str] = []
        if ignorance_secret_ids:
            reasons.append(
                "Ignorance secrets leak exact object IDs: "
                + ", ".join(ignorance_secret_ids)
            )
        if ignorance_secret_lines:
            reasons.append("Secrets contain 'you do not know ...' style lines.")
        if epistemic_prompt_lines:
            reasons.append("Secrets contain epistemic coaching lines.")
        if boilerplate_secret_lines:
            reasons.append("Secrets contain boilerplate self-intro text.")
        judgment.criteria_scores["secret_quality"] = CriterionScore(
            score=min(current.score, 0.3),
            reasoning=(f"{current.reasoning} " + " ".join(reasons)).strip(),
        )

    fixes: List[str] = []
    if public_task_ids or ignorance_secret_ids:
        fixes.append(
            "Remove exact hidden object IDs from the public task and from secrets for agents who do not already know that fact; keep exact IDs only in the knowing agent's secret and in problem_pddl."
        )
    if ignorance_secret_lines or epistemic_prompt_lines or boilerplate_secret_lines:
        fixes.append(
            "Delete non-knowledge secret lines such as 'you do not know ...', 'By the end, you must be confident ...', 'Epistemic probe: ...', and self-intro boilerplate."
        )
    for fix in reversed(fixes):
        if fix not in judgment.required_fixes:
            judgment.required_fixes.insert(0, fix)

    all_scores = [criterion.score for criterion in judgment.criteria_scores.values()]
    judgment.overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    judgment.is_valid = (
        judgment.overall_score >= overall_threshold
        and all(
            score.score >= min_criterion_threshold
            for score in judgment.criteria_scores.values()
        )
    )
    return judgment


def _normalize_skip_steps(skip_steps: Optional[List[str]]) -> Set[str]:
    """Normalize skip-step names for prompt and parser logic."""
    return {str(step).strip().lower() for step in (skip_steps or []) if str(step).strip()}


def _get_criteria_for_category(
    category: str,
    user_query: Optional[str] = None,
    skip_steps: Optional[List[str]] = None,
) -> List[str]:
    """Get the active criteria for a category under the current pipeline config."""
    criteria = list(CATEGORY_CRITERIA.get(category, SHARED_CRITERIA))
    if "pddl" in _normalize_skip_steps(skip_steps):
        criteria = [criterion for criterion in criteria if criterion != "pddl_solvability"]
    # Support ablation: ENACTTOM_EXCLUDE_CRITERION=X removes criterion X from the council
    exclude = os.environ.get("ENACTTOM_EXCLUDE_CRITERION", "").strip()
    if exclude:
        criteria = [c for c in criteria if c != exclude]
    if user_query:
        criteria.append("user_requirements_alignment")
    return criteria


def _build_criteria_section(
    category: str,
    user_query: Optional[str] = None,
    skip_steps: Optional[List[str]] = None,
) -> str:
    """Build the criteria section for a given category."""
    criteria = _get_criteria_for_category(category, user_query, skip_steps=skip_steps)
    lines = []
    for i, criterion in enumerate(criteria, 1):
        info = CRITERIA_DESCRIPTIONS.get(criterion, {})
        lines.append(f"### {i}. {info.get('name', criterion)} (0.0-1.0)")
        lines.append(info.get('description', ''))
        lines.append(f"- {info.get('rubric', '')}")
        lines.append("")
    return "\n".join(lines)


def _build_response_format(
    category: str,
    user_query: Optional[str] = None,
    skip_steps: Optional[List[str]] = None,
) -> str:
    """Build the JSON response format for a given category."""
    criteria = _get_criteria_for_category(category, user_query, skip_steps=skip_steps)
    lines = []
    for criterion in criteria:
        lines.append(f'  "{criterion}": {{"score": <0.0-1.0>, "reasoning": "<brief>"}},')
    return "\n".join(lines)


def _build_formal_checks_section(skip_steps: Optional[List[str]] = None) -> str:
    """Build the formal-solvability instructions for the council prompt."""
    if "pddl" in _normalize_skip_steps(skip_steps):
        return (
            "- PDDL solvability verification is disabled for this run. Do NOT score or discuss "
            "`pddl_solvability`, formal proof quality, or compiled solvability evidence.\n"
            "- You may still use `problem_pddl` to understand authored goals, but judge only task "
            "design quality, runtime functional pressure, mechanic coherence, and split-semantics "
            "meaningfulness."
        )

    return (
        "- Raw `problem_pddl` should be self-contained for scene/world facts: required symbols belong "
        "in `:objects`, relevant room grounding belongs in `:init`, while mechanic-derived init facts "
        "like room restrictions should come from `mechanic_bindings`\n"
        "- For solvability and mechanic-awareness, treat the Compiled Formal View below as authoritative. "
        "Do NOT penalize raw `problem_pddl` for omitting mechanic-derived init facts such as "
        "`is_restricted` or `can_communicate`; those are expected to be compiled from "
        "`mechanic_bindings`.\n"
        "- **K() goal backing**: Every `K()` goal in `problem_pddl` (or legacy goal field) must be "
        "backed by a mechanic that prevents the agent from directly observing the fact (e.g., "
        "`room_restriction` blocks navigation, `restricted_communication` blocks direct messaging). "
        "If the agent could just walk there and see, the K() goal is fake.\n"
        "- **Functional projection quality**: Penalize tasks whose non-epistemic projection becomes "
        "vacuous, trivial, effectively single-agent, or no longer reflects the intended coordination "
        "challenge.\n"
        "- **Probe quality**: Reward K() goals when they probe who knows functionally relevant facts "
        "under real asymmetry. Do NOT require K() to be part of runtime pass/fail.\n"
        "- **Functional ToM quality**: Reward tasks where the best action depends on a partner-specific "
        "model (who can act, who will relay, who may prioritize a private goal, who has the last "
        "message). Penalize tasks that reduce to \"agent A sees a hidden fact and tells agent B.\"\n"
        "- Distinguish **formal validity** from **design quality**:\n"
        "  - If the formal task is invalid, contradictory, or not self-contained, score `Formal Goal "
        "Quality & Epistemic Coherence` near 0.\n"
        "  - If the formal task is valid, judge whether both the functional projection and the "
        "literal-ToM probe structure are meaningful for the benchmark rather than merely technically valid."
    )


def _build_runtime_semantics_section(task_dict: Dict[str, Any]) -> str:
    """Summarize the derived functional goal and literal-ToM probes for judge prompts."""
    functional_goal = task_dict.get("functional_goal_pddl")
    probes = task_dict.get("literal_tom_probes")

    if not functional_goal or probes is None:
        try:
            from enacttom.pddl.runtime_projection import build_runtime_metadata

            derived = build_runtime_metadata(task_dict)
            functional_goal = functional_goal or derived.get("functional_goal_pddl")
            probes = probes if probes is not None else derived.get("literal_tom_probes", [])
        except Exception:
            probes = probes or []

    lines = []
    if functional_goal:
        lines.append("Functional goal projection used for runtime success:")
        lines.append(functional_goal)
    else:
        lines.append("Functional goal projection: <unavailable>")

    if probes:
        lines.append("Literal-ToM probes derived from K() goals:")
        for probe in probes[:8]:
            source = probe.get("source_pddl", "<unknown>")
            agent = probe.get("agent_id", "<unknown>")
            question = probe.get("question", "").strip().splitlines()[0] if probe.get("question") else ""
            lines.append(f"- {agent}: {source}")
            if question:
                lines.append(f"  Probe question stem: {question}")
        if len(probes) > 8:
            lines.append(f"- ... {len(probes) - 8} more probes")
    else:
        lines.append("Literal-ToM probes: none derived")

    return "\n".join(lines)


def _build_compiled_formal_view_section(
    task_dict: Dict[str, Any],
    scene_data: Optional["SceneData"],
) -> str:
    """Build the normalized formal problem view used by the verifier."""
    try:
        from enacttom.pddl.compiler import compile_task
        from enacttom.task_gen.task_generator import GeneratedTask

        task = GeneratedTask.from_dict(task_dict)
        scene_payload: Optional[Dict[str, Any]]
        if scene_data is None:
            scene_payload = None
        elif hasattr(scene_data, "to_dict"):
            scene_payload = scene_data.to_dict()  # type: ignore[assignment]
        elif isinstance(scene_data, dict):
            scene_payload = scene_data
        else:
            scene_payload = None

        compiled = compile_task(task, scene_payload)
        lines = [
            "Compiled problem used for mechanic-aware solvability checks:",
            "```lisp",
            compiled.to_pddl(),
            "```",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Compiled formal view unavailable: {exc}"


def _build_compiled_formal_view_block(
    task_dict: Dict[str, Any],
    scene_data: Optional["SceneData"],
    skip_steps: Optional[List[str]] = None,
) -> str:
    """Build the compiled-formal-view block, or omit it when PDDL judging is disabled."""
    if "pddl" in _normalize_skip_steps(skip_steps):
        return (
            "## Compiled Formal View\n"
            "PDDL verification is disabled for this run, so ignore formal solvability and compiled-plan evidence."
        )

    return (
        "## Compiled Formal View\n"
        "Use this normalized formal problem when checking mechanic-aware solvability claims.\n"
        "It reflects the authored `problem_pddl` after mechanic-derived and planner-required\n"
        "init facts are compiled in. This is the authoritative formal view for\n"
        "`pddl_solvability`; raw `problem_pddl` may omit mechanic-derived init facts.\n"
        f"{_build_compiled_formal_view_section(task_dict, scene_data)}"
    )


def _build_benchmark_comparison_section(task_dict: Dict[str, Any]) -> str:
    """Summarize the latest standard vs baseline calibration pair when present."""
    calibration = task_dict.get("calibration", [])
    if not isinstance(calibration, list):
        calibration = []
    if not calibration:
        return "Benchmark comparison: none recorded yet."

    latest_standard = None
    for entry in calibration:
        run_mode = str(entry.get("run_mode", "standard") or "standard")
        if run_mode == "standard":
            latest_standard = entry
    if latest_standard is None:
        return "Benchmark comparison: no standard calibration recorded yet."

    latest_baseline = None
    target_models = latest_standard.get("agent_models", {})
    for entry in calibration:
        run_mode = str(entry.get("run_mode", "standard") or "standard")
        if run_mode == "baseline" and entry.get("agent_models") == target_models:
            latest_baseline = entry
    if latest_baseline is None:
        return "Benchmark comparison: standard recorded, but no matching baseline calibration yet."

    def _results_summary(entry: Dict[str, Any]) -> str:
        results = entry.get("results", {})
        if "main_goal" in results:
            return (
                f"passed={results['main_goal'].get('passed', False)}, "
                f"progress={results['main_goal'].get('progress', 0.0):.0%}"
            )
        return (
            f"passed={results.get('passed', False)}, "
            f"progress={results.get('progress', 0.0):.0%}"
        )

    lines = [
        "Latest benchmark comparison:",
        f"- Standard: {_results_summary(latest_standard)}",
        f"- Baseline: {_results_summary(latest_baseline)}",
        (
            "Interpretation: stronger functional-ToM evidence comes from tasks where the "
            "baseline run succeeds and the standard run is materially weaker."
        ),
    ]
    return "\n".join(lines)


class Judge:
    """
    LLM judge for task validation.

    Evaluates tasks using category-specific criteria:
    - Cooperative: 6 criteria (5 shared + task_interdependence)
    - Mixed: 6 criteria (5 shared + subgoal_tension)

    Priority criteria (required fixes appear first):
    - agent_necessity: Every agent must be essential
    - secret_quality: Secrets must be actionable, natural, and non-leaking

    The task passes only if every configured judge model passes.
    """

    # Priority criteria - required fixes should focus here first
    PRIORITY_CRITERIA = ["agent_necessity", "secret_quality"]

    DEFAULT_MODELS = ["gpt-5.4-mini"]
    MODEL_REQUEST_TIMEOUT_S = 45
    COUNCIL_WALL_TIMEOUT_S = 180
    COUNCIL_RETRY_ATTEMPTS = 1
    COUNCIL_RETRY_TIMEOUT_S = 300  # Longer timeout on retry

    def __init__(
        self,
        models: Optional[List[str]] = None,
        overall_threshold: float = 0.65,
        min_criterion_threshold: float = 0.5,
        verbose: bool = False,
        user_query: Optional[str] = None,
        diversity_tracker: Optional["DiversityTracker"] = None,
        difficulty: Optional[str] = None,
        skip_steps: Optional[List[str]] = None,
    ):
        """
        Initialize the judge.

        Args:
            models: List of model names for judging (default: ["gpt-5.4-mini"])
            overall_threshold: Minimum overall score to pass (default 0.65)
            min_criterion_threshold: Minimum score for any criterion (default 0.5)
            verbose: Print debug information
            user_query: Optional user query that the task should align with
            diversity_tracker: Optional tracker to check task novelty against existing tasks
            difficulty: Intended difficulty level ("easy", "medium", "hard") for calibrated evaluation
            skip_steps: Optional prompt-only ablations.
        """
        self.models = models or self.DEFAULT_MODELS
        self.overall_threshold = overall_threshold
        self.min_criterion_threshold = min_criterion_threshold
        self.verbose = verbose
        self.user_query = user_query
        self.diversity_tracker = diversity_tracker
        self.difficulty = difficulty
        self.skip_steps = sorted(_normalize_skip_steps(skip_steps))

        # LLM clients (created lazily)
        self._llm_clients: Dict[str, "BaseLLM"] = {}

        # Cache grounding info
        self._available_actions: Optional[str] = None
        self._available_mechanics: Optional[str] = None
        self._authoring_constraints: Optional[str] = None
        self._available_predicates: Optional[str] = None

    def _get_llm_client(self, model: str) -> "BaseLLM":
        """Get or create LLM client for a model."""
        if model not in self._llm_clients:
            provider = _detect_provider_for_model(model)
            client_model = _resolve_client_model_name(model)
            from habitat_llm.llm import instantiate_llm
            self._llm_clients[model] = instantiate_llm(
                provider,
                generation_params={
                    "model": client_model,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                }
            )

        return self._llm_clients[model]

    def _get_grounding_info(self) -> Dict[str, str]:
        """Get cached grounding information about system capabilities."""
        if self._available_actions is None:
            try:
                self._available_actions = get_authoring_action_descriptions()
            except Exception:
                self._available_actions = "Navigate, Open, Close, Pick, Place, Communicate, Wait"

        if self._available_mechanics is None:
            try:
                self._available_mechanics = get_authoring_mechanics()
            except Exception:
                self._available_mechanics = format_supported_mechanics()

        if self._authoring_constraints is None:
            self._authoring_constraints = AUTHORING_CONSTRAINTS_NOTICE

        if self._available_predicates is None:
            try:
                self._available_predicates = get_authoring_predicates()
            except Exception:
                self._available_predicates = "is_on_top, is_inside, is_in_room, is_on_floor, is_next_to, is_open, is_closed, is_clean, is_dirty, is_filled, is_empty, is_powered_on, is_unlocked, is_locked, is_held_by, agent_in_room"

        return {
            "available_actions": self._available_actions,
            "available_mechanics": self._available_mechanics,
            "authoring_constraints": self._authoring_constraints,
            "available_predicates": self._available_predicates,
        }

    def _format_scene_objects(self, scene_data: Optional["SceneData"]) -> str:
        """Format scene objects for the prompt."""
        if scene_data is None:
            return "Scene data not available. Use object IDs from the task JSON."

        lines = []
        lines.append(f"**Rooms**: {', '.join(scene_data.rooms[:10])}")
        lines.append(f"**Furniture**: {', '.join(scene_data.furniture[:20])}")
        if len(scene_data.furniture) > 20:
            lines.append(f"  ... and {len(scene_data.furniture) - 20} more")
        lines.append(f"**Objects**: {', '.join(scene_data.objects[:20])}")
        if len(scene_data.objects) > 20:
            lines.append(f"  ... and {len(scene_data.objects) - 20} more")

        return "\n".join(lines)

    def _format_rollout(self, rollout: BenchmarkRollout) -> str:
        """Format rollout data for the prompt."""
        lines = []
        lines.append("---")
        lines.append("## BENCHMARK ROLLOUT DATA (from actual LLM agents)")
        lines.append("")
        lines.append("This task was run with LLM agents. Use this data to assess difficulty and plausibility.")
        lines.append("")
        lines.append(f"**Result**: {'SUCCESS' if rollout.success else 'FAILED'}")
        lines.append(f"**Steps**: {rollout.steps}")
        lines.append(f"**Turns**: {rollout.turns}")
        lines.append(f"**Progress**: {rollout.percent_complete:.0%}")
        lines.append("")

        if rollout.subtask_status:
            lines.append("**Subtask Completion**:")
            for subtask_id, completed in rollout.subtask_status.items():
                status = "COMPLETE" if completed else "INCOMPLETE"
                lines.append(f"  - {subtask_id}: {status}")
            lines.append("")

        # Include agent trace excerpts (truncated)
        if rollout.agent_traces:
            lines.append("**Agent Reasoning Excerpts** (truncated):")
            for agent_id, trace in rollout.agent_traces.items():
                lines.append(f"\n  [{agent_id}]:")
                # Take first 500 chars of trace
                excerpt = trace[:500] + "..." if len(trace) > 500 else trace
                for line in excerpt.split("\n")[:10]:
                    lines.append(f"    {line}")
            lines.append("")

        lines.append("**Consider in evaluation**:")
        lines.append("- If agents failed, is the task too hard or poorly designed?")
        lines.append("- If agents succeeded easily, is the task too trivial?")
        lines.append("- Did agents exhibit ToM reasoning in their traces?")
        lines.append("---")

        return "\n".join(lines)

    def evaluate(
        self,
        task: "GeneratedTask | Dict[str, Any]",
        scene_data: Optional["SceneData"] = None,
        trajectory_dir: Optional[Path] = None,
    ) -> CouncilVerdict:
        """
        Evaluate a task using the configured judge model(s).

        Args:
            task: GeneratedTask object or task dictionary
            scene_data: Optional scene data for grounded required fixes
            trajectory_dir: Optional path to benchmark rollout data

        Returns:
            CouncilVerdict with aggregated results from all models
        """
        # Convert to dict if needed
        if hasattr(task, "to_dict"):
            task_dict = task.to_dict()
        else:
            task_dict = task

        # Load rollout data if available
        rollout = None
        if trajectory_dir:
            rollout = BenchmarkRollout.from_trajectory_dir(Path(trajectory_dir))
            if rollout:
                current_spec_hash = None
                try:
                    from enacttom.pddl.planner import compute_task_spec_hash

                    current_spec_hash = compute_task_spec_hash(task_dict)
                except Exception:
                    current_spec_hash = None

                # Only trust rollout if snapshot metadata exists and matches current spec.
                if not rollout.snapshot_spec_hash:
                    if self.verbose:
                        print("[Judge] Ignoring rollout: missing task_snapshot.json/spec_hash metadata")
                    rollout = None
                elif current_spec_hash and rollout.snapshot_spec_hash != current_spec_hash:
                    if self.verbose:
                        print(
                            "[Judge] Ignoring stale rollout: snapshot spec hash does not match current task"
                        )
                    rollout = None
                elif self.verbose:
                    print(f"[Judge] Loaded rollout: success={rollout.success}, {rollout.steps} steps")

        # Evaluate with all models in parallel.
        from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

        if self.verbose:
            print(f"[Judge] Evaluating with {len(self.models)} models in parallel: {', '.join(self.models)}")

        judgments: Dict[str, Judgment] = {}
        models_to_evaluate = list(self.models)

        for attempt in range(1 + self.COUNCIL_RETRY_ATTEMPTS):
            timeout = self.COUNCIL_WALL_TIMEOUT_S if attempt == 0 else self.COUNCIL_RETRY_TIMEOUT_S
            executor = ThreadPoolExecutor(max_workers=len(models_to_evaluate))
            future_to_model = {
                executor.submit(self._evaluate_single, task_dict, model, scene_data, rollout): model
                for model in models_to_evaluate
            }
            try:
                done, not_done = wait(
                    set(future_to_model.keys()),
                    timeout=timeout,
                    return_when=ALL_COMPLETED,
                )
                for future in done:
                    model = future_to_model[future]
                    try:
                        judgments[model] = future.result()
                        if self.verbose:
                            print(f"[Judge] {model} completed")
                    except Exception as e:
                        print(f"[Judge] {model} failed: {e}")
                        judgments[model] = self._failed_judgment(
                            f"[{model}] Error: {e}",
                            category=task_dict.get("category", "cooperative"),
                        )

                timed_out_models = []
                for future in not_done:
                    model = future_to_model[future]
                    reason = (
                        f"[{model}] Timed out after {timeout}s "
                        "waiting for judge model response"
                    )
                    print(f"[Judge] {reason}")
                    judgments[model] = self._failed_judgment(
                        reason,
                        category=task_dict.get("category", "cooperative"),
                    )
                    timed_out_models.append(model)
                    future.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            # Retry timed-out models (infrastructure failures only)
            retry_models = [
                m for m in timed_out_models
                if m in judgments and self._is_infra_failure(judgments[m])
            ]
            if not retry_models or attempt >= self.COUNCIL_RETRY_ATTEMPTS:
                break
            print(f"[Judge] Retrying {len(retry_models)} timed-out model(s): {', '.join(retry_models)}")
            models_to_evaluate = retry_models

        # Check novelty if diversity tracker is available
        if self.diversity_tracker:
            novelty_result = self.diversity_tracker.check_novelty(task_dict)
            novelty_score = CriterionScore(
                score=novelty_result["score"],
                reasoning=novelty_result["reason"],
            )
            if self.verbose:
                print(f"[Judge] Novelty check: {novelty_result['score']:.2f} - {novelty_result['reason']}")

            # Inject novelty score into each judgment
            for model, judgment in judgments.items():
                judgment.criteria_scores["task_novelty"] = novelty_score
                # Add a required fix if novelty is low.
                if novelty_result["score"] < self.min_criterion_threshold:
                    similar_str = ", ".join(novelty_result.get("similar_to", [])[:3])
                    suggestion = f"[Task Novelty] Task is too similar to existing patterns"
                    if similar_str:
                        suggestion += f" (similar to: {similar_str})"
                    suggestion += ". Try a different win condition, mechanics, or dependency structure."
                    if suggestion not in judgment.required_fixes:
                        judgment.required_fixes.insert(0, suggestion)
                # Recalculate overall score to include novelty
                all_scores = [c.score for c in judgment.criteria_scores.values()]
                judgment.overall_score = sum(all_scores) / len(all_scores)
                # Recalculate validity — novelty is a soft criterion (affects
                # the average but cannot single-handedly veto a task)
                hard_scores = [
                    c.score for k, c in judgment.criteria_scores.items()
                    if k != "task_novelty"
                ]
                judgment.is_valid = (
                    judgment.overall_score >= self.overall_threshold
                    and all(s >= self.min_criterion_threshold for s in hard_scores)
                )

        # Aggregate results
        return self._aggregate(judgments)

    def _evaluate_single(
        self,
        task_dict: Dict[str, Any],
        model: str,
        scene_data: Optional["SceneData"] = None,
        rollout: Optional[BenchmarkRollout] = None,
    ) -> Judgment:
        """Evaluate task with a single model."""
        llm = self._get_llm_client(model)

        # Get task category (default to cooperative for backwards compatibility)
        category = task_dict.get("category", "cooperative")
        if category not in CATEGORY_CRITERIA:
            category = "cooperative"

        # Build user requirements section if query was provided
        user_requirements_section = ""
        if self.user_query:
            user_requirements_section = f"""
## User Requirements

The user specifically requested:
> {self.user_query}

**IMPORTANT**: The task MUST align with this request. Evaluate whether the task incorporates the requested mechanics, themes, and scene constraints.
"""

        # Build difficulty section
        difficulty_section = ""
        if self.difficulty and self.difficulty in DIFFICULTY_DESCRIPTIONS:
            difficulty_section = DIFFICULTY_DESCRIPTIONS[self.difficulty]

        # Build category-aware prompt
        grounding = self._get_grounding_info()
        scene_objects = self._format_scene_objects(scene_data)
        task_json = json.dumps(task_dict, indent=2)

        prompt = EVALUATION_PROMPT.format(
            category=category.upper(),
            category_description=CATEGORY_PROMPT_DESCRIPTIONS.get(category, ""),
            difficulty_section=difficulty_section,
            user_requirements_section=user_requirements_section,
            formal_checks_section=_build_formal_checks_section(self.skip_steps),
            runtime_semantics_section=_build_runtime_semantics_section(task_dict),
            id_leakage_section=_build_id_leakage_section(task_dict),
            compiled_formal_view_block=_build_compiled_formal_view_block(
                task_dict,
                scene_data,
                skip_steps=self.skip_steps,
            ),
            benchmark_comparison_section=_build_benchmark_comparison_section(task_dict),
            criteria_section=_build_criteria_section(
                category,
                self.user_query,
                skip_steps=self.skip_steps,
            ),
            response_format=_build_response_format(
                category,
                self.user_query,
                skip_steps=self.skip_steps,
            ),
            task_json=task_json,
            available_actions=grounding["available_actions"],
            available_mechanics=grounding["available_mechanics"],
            authoring_constraints=grounding["authoring_constraints"],
            available_predicates=grounding["available_predicates"],
            scene_objects=scene_objects,
        )

        # Add rollout data if available
        if rollout:
            rollout_section = self._format_rollout(rollout)
            prompt += f"\n\n{rollout_section}"

        if self.verbose:
            print(f"[Judge/{model}] Evaluating {category} task ({len(prompt)} chars)")

        # Get response with retries for transient provider/network failures.
        # Use aggressive backoff with jitter to handle rate limits (429)
        # under high parallelism (e.g. 128 concurrent bulk gen processes).
        response = None
        max_retries = 8
        for attempt in range(1, max_retries + 1):
            try:
                response = llm.generate(
                    prompt,
                    request_timeout=self.MODEL_REQUEST_TIMEOUT_S,
                )
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                # Longer backoff for rate limits (429), shorter for other errors
                exc_str = str(exc)
                if "429" in exc_str or "too_many_requests" in exc_str or "overloaded" in exc_str:
                    base_backoff = min(120, 15 * (2 ** (attempt - 1)))
                else:
                    base_backoff = min(30, 2 ** attempt)
                # Add jitter to avoid thundering herd
                import random
                backoff_s = base_backoff * (0.5 + random.random())
                if self.verbose:
                    print(
                        f"[Judge/{model}] LLM call failed (attempt {attempt}/{max_retries}), "
                        f"retrying in {backoff_s:.0f}s"
                    )
                time.sleep(backoff_s)

        if self.verbose:
            print(f"[Judge/{model}] Received response ({len(response or '')} chars)")

        # Parse response (pass user_query so it knows which criteria to expect)
        judgment = self._parse_response(
            response or "",
            model,
            category,
            self.user_query,
            skip_steps=self.skip_steps,
        )
        return _apply_id_leakage_penalties(
            judgment,
            task_dict,
            self.overall_threshold,
            self.min_criterion_threshold,
        )

    def _parse_response(
        self,
        response: str,
        model: str,
        category: str = "cooperative",
        user_query: Optional[str] = None,
        skip_steps: Optional[List[str]] = None,
    ) -> Judgment:
        """Parse LLM response into Judgment."""
        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return self._failed_judgment(f"[{model}] Failed to parse response", category)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return self._failed_judgment(f"[{model}] JSON parse error: {e}", category)

        # Get criteria for this category (includes user_requirements_alignment if query provided)
        criteria_names = _get_criteria_for_category(category, user_query, skip_steps=skip_steps)

        criteria_scores = {}
        scores = []

        for name in criteria_names:
            if name in data and isinstance(data[name], dict):
                score = float(data[name].get("score", 0.0))
                reasoning = data[name].get("reasoning", "No reasoning provided")
                criteria_scores[name] = CriterionScore(score=score, reasoning=reasoning)
                scores.append(score)
            else:
                criteria_scores[name] = CriterionScore(score=0.0, reasoning="Missing from response")
                scores.append(0.0)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0.0

        is_valid = (
            overall_score >= self.overall_threshold
            and all(s >= self.min_criterion_threshold for s in scores)
        )

        # Extract minimum required fixes.
        raw_required_fixes = data.get("required_fixes", [])
        if not isinstance(raw_required_fixes, list):
            raw_required_fixes = [str(raw_required_fixes)]
        required_fixes: List[str] = []
        for fix in raw_required_fixes:
            fix_text = str(fix).strip()
            if fix_text and fix_text not in required_fixes:
                required_fixes.append(fix_text)

        # Fall back to criterion-scoped fixes only when the model omitted
        # concrete required_fixes entirely.
        if not required_fixes:
            for name, criterion in criteria_scores.items():
                if criterion.score < self.min_criterion_threshold:
                    criterion_fix = f"Fix {name} ({criterion.score:.2f}): {criterion.reasoning}"
                    if criterion_fix not in required_fixes:
                        required_fixes.append(criterion_fix)

        if not required_fixes and not is_valid and overall_score < self.overall_threshold:
            required_fixes.append(
                f"Raise overall task quality above {self.overall_threshold:.2f} without weakening current strengths."
            )

        return Judgment(
            category=category,
            criteria_scores=criteria_scores,
            overall_score=overall_score,
            is_valid=is_valid,
            overall_reasoning=data.get("overall_reasoning", "No reasoning provided"),
            required_fixes=required_fixes[:3],
        )

    @staticmethod
    def _is_infra_failure(judgment: "Judgment") -> bool:
        """Check if a judgment failed due to infrastructure (not task quality)."""
        reason = (judgment.overall_reasoning or "").lower()
        markers = (
            "429", "too_many_requests", "overloaded", "rate limit",
            "connection error", "could not connect", "timed out",
            "endpoint url", "name resolution", "error:",
        )
        return judgment.overall_score == 0.0 and any(m in reason for m in markers)

    def _failed_judgment(self, reason: str, category: str = "cooperative") -> Judgment:
        """Create a failed judgment for parse errors."""
        criteria_names = _get_criteria_for_category(
            category,
            self.user_query,
            skip_steps=self.skip_steps,
        )
        failed_scores = {
            name: CriterionScore(score=0.0, reasoning=reason)
            for name in criteria_names
        }
        return Judgment(
            category=category,
            criteria_scores=failed_scores,
            overall_score=0.0,
            is_valid=False,
            overall_reasoning=reason,
            required_fixes=["Re-run evaluation"],
        )

    def _aggregate(self, judgments: Dict[str, Judgment]) -> CouncilVerdict:
        """Aggregate judgments from multiple models."""
        infra_failures = {
            m: j for m, j in judgments.items()
            if self._is_infra_failure(j)
        }
        if infra_failures:
            failed_models = ", ".join(infra_failures.keys())
            raise RuntimeError(
                "Judge failed due to infrastructure errors: "
                f"{failed_models}. Check API keys, billing, and network connectivity."
            )

        if not judgments:
            raise RuntimeError("Judge failed: no model judgments were produced.")


        # Check if all models pass
        all_pass = all(j.is_valid for j in judgments.values())

        # Average overall scores
        avg_score = sum(j.overall_score for j in judgments.values()) / len(judgments)

        # Merge required fixes (deduplicated) across models.
        required_fixes = []
        seen = set()
        for j in judgments.values():
            for fix in j.required_fixes:
                if fix not in seen:
                    seen.add(fix)
                    required_fixes.append(fix)

        # Find disagreements
        disagreements = []
        if len(judgments) > 1:
            models = list(judgments.keys())
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    if judgments[m1].is_valid != judgments[m2].is_valid:
                        disagreements.append(
                            f"{m1} ({'PASS' if judgments[m1].is_valid else 'FAIL'}) vs "
                            f"{m2} ({'PASS' if judgments[m2].is_valid else 'FAIL'})"
                        )

        return CouncilVerdict(
            judgments=judgments,
            passed=all_pass,
            overall_score=avg_score,
            required_fixes=required_fixes[:3],
            disagreements=disagreements,
        )

    def format_result(self, verdict: CouncilVerdict) -> str:
        """Format verdict as human-readable string."""
        lines = []
        lines.append("=" * 60)
        lines.append("TASK EVALUATION (Council)")
        lines.append("=" * 60)

        status = "PASS" if verdict.passed else "FAIL"
        lines.append(f"\nStatus: {status}")
        lines.append(f"Overall Score: {verdict.overall_score:.2f} (threshold: {self.overall_threshold})")
        lines.append(f"Models: {', '.join(self.models)}")

        if verdict.disagreements:
            lines.append(f"\nDisagreements:")
            for d in verdict.disagreements:
                lines.append(f"  - {d}")

        # Show per-model breakdown
        for model, judgment in verdict.judgments.items():
            lines.append(f"\n--- {model} ({'PASS' if judgment.is_valid else 'FAIL'}) ---")
            lines.append(f"Category: {judgment.category}")
            lines.append(f"Score: {judgment.overall_score:.2f}")

            lines.append("\nCriteria:")
            for name, criterion in judgment.criteria_scores.items():
                icon = "+" if criterion.score >= self.min_criterion_threshold else "!"
                lines.append(f"  [{icon}] {name}: {criterion.score:.2f}")

        if verdict.required_fixes:
            lines.append("\nRequired Fixes:")
            for i, s in enumerate(verdict.required_fixes[:10], 1):
                lines.append(f"  {i}. {s}")

        lines.append("=" * 60)
        return "\n".join(lines)


# CLI functionality
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate task quality with category-specific criteria"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Path to task JSON file"
    )
    parser.add_argument(
        "--models", type=str, default="gpt-5.4-mini",
        help="Comma-separated list of judge models (default: gpt-5.4-mini)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Overall score threshold (default: 0.65)"
    )
    parser.add_argument(
        "--min-criterion", type=float, default=0.5,
        help="Minimum criterion score (default: 0.5)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for JSON results (default: auto-generated)"
    )
    parser.add_argument(
        "--trajectory-dir", type=str, default=None,
        help="Path to benchmark rollout data (agent traces, result.txt)"
    )

    args = parser.parse_args()

    # Load task
    task_path = Path(args.task)
    if not task_path.exists():
        print(f"{Colors.RED}Error: Task file not found: {task_path}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Error: Invalid JSON: {e}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    # Parse models
    models = [m.strip() for m in args.models.split(",")]

    print(f"{Colors.CYAN}Evaluating task with judge model(s): {models}{Colors.RESET}", file=sys.stderr)

    # Load trajectory dir if specified
    trajectory_dir = Path(args.trajectory_dir) if args.trajectory_dir else None
    if trajectory_dir:
        print(f"{Colors.CYAN}Including rollout data from: {trajectory_dir}{Colors.RESET}", file=sys.stderr)

    # Create judge and evaluate
    judge = Judge(
        models=models,
        overall_threshold=args.threshold,
        min_criterion_threshold=args.min_criterion,
        verbose=args.verbose,
    )

    verdict = judge.evaluate(task_data, trajectory_dir=trajectory_dir)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/enacttom") / f"{timestamp}-judge"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"verdict_{task_path.stem}.json"

    with open(output_path, "w") as f:
        f.write(verdict.to_json())

    # Print results
    print(verdict.to_json())

    # Print summary
    if verdict.passed:
        print(f"\n{Colors.BOLD}{Colors.GREEN}PASSED{Colors.RESET}", file=sys.stderr)
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}FAILED{Colors.RESET}", file=sys.stderr)
        if verdict.disagreements:
            print(f"{Colors.YELLOW}Models disagreed - defaulting to FAIL{Colors.RESET}", file=sys.stderr)

    print(f"\nSaved to: {Colors.CYAN}{output_path}{Colors.RESET}", file=sys.stderr)

    sys.exit(0 if verdict.passed else 1)


if __name__ == "__main__":
    main()
