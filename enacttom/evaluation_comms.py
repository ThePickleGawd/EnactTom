"""
Communication metrics for EnactToM benchmark evaluation.

Measures how efficiently and strategically agents communicated during a
benchmark run. Detects secret leakage (agents dumping their private info
into group chat) and scores overall communication discipline.

All metrics are post-hoc (no constraints on agents during the run).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from enacttom.task_gen.task_generator import GeneratedTask


@dataclass
class AgentCommMetrics:
    """Communication metrics for a single agent."""
    message_count: int
    avg_message_length: float  # average word count per message
    secret_leakage_score: float  # 0.0 = leaked everything, 1.0 = no leakage
    leaked_secrets: List[str]  # which secrets were leaked
    messages: List[str]  # actual messages sent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_count": self.message_count,
            "avg_message_length": round(self.avg_message_length, 1),
            "secret_leakage_score": round(self.secret_leakage_score, 3),
            "leaked_secrets": self.leaked_secrets,
            "messages": self.messages,
        }


@dataclass
class CommunicationMetrics:
    """Aggregated communication metrics for a benchmark run."""
    per_agent: Dict[str, AgentCommMetrics]
    overall_leakage_score: float  # 0-1, higher = less leakage
    overall_efficiency_score: float  # 0-1, higher = more strategic communication
    overall_score: float  # weighted combination
    efficiency_reasoning: str = ""  # LLM's reasoning for efficiency score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_agent": {k: v.to_dict() for k, v in self.per_agent.items()},
            "overall_leakage_score": round(self.overall_leakage_score, 3),
            "overall_efficiency_score": round(self.overall_efficiency_score, 3),
            "overall_score": round(self.overall_score, 3),
            "efficiency_reasoning": self.efficiency_reasoning,
        }


def _detect_provider_for_model(model: str) -> str:
    """Detect provider for communication-efficiency LLM scoring."""
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
        env_path = Path(__file__).resolve().parents[1] / ".env"
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


def _extract_messages(action_history: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Extract Communicate messages from action history, grouped by agent."""
    agent_messages: Dict[str, List[str]] = {}

    for entry in action_history:
        if entry.get("type") == "subtask_completion":
            continue

        action = entry.get("action", "")
        agent_id = entry.get("agent", "")

        # Match Communicate["message", recipients] or Communicate[message] pattern
        match = re.match(r'Communicate\["(.+?)"(?:,\s*[^]]*)?\]$', action, re.DOTALL)
        if not match:
            # Fallback for unquoted legacy format
            match = re.match(r"Communicate\[(.+)\]$", action, re.DOTALL)
        if match:
            message = match.group(1)
            if agent_id not in agent_messages:
                agent_messages[agent_id] = []
            agent_messages[agent_id].append(message)

    return agent_messages


def _detect_secret_leakage(
    messages: List[str],
    secrets: List[str],
    threshold: float = 0.6,
) -> tuple[float, List[str]]:
    """
    Detect if an agent leaked their secrets in messages.

    Uses fuzzy substring matching. If >threshold of a secret appears
    in any message, it's flagged as leaked.

    Args:
        messages: Messages sent by the agent
        secrets: The agent's private secrets
        threshold: Similarity ratio to count as leaked (default 0.6)

    Returns:
        (leakage_score, leaked_secrets_list)
        leakage_score: 1.0 = no leakage, 0.0 = all secrets leaked
    """
    if not secrets:
        return 1.0, []

    # Filter out placeholder secrets once upfront
    real_secrets = [s for s in secrets if "REPLACE" not in s]
    if not real_secrets:
        return 1.0, []

    leaked = []
    all_messages_lower = " ".join(messages).lower()

    for secret in real_secrets:
        secret_lower = secret.lower()

        # Check direct substring containment first (fast path)
        if secret_lower in all_messages_lower:
            leaked.append(secret)
            continue

        # Fuzzy matching: check each message against the secret
        for msg in messages:
            ratio = SequenceMatcher(None, secret_lower, msg.lower()).ratio()
            if ratio >= threshold:
                leaked.append(secret)
                break

            # Also check if most words from the secret appear in the message
            secret_words = set(secret_lower.split())
            msg_words = set(msg.lower().split())
            if len(secret_words) > 3:
                overlap = len(secret_words & msg_words) / len(secret_words)
                if overlap >= threshold:
                    leaked.append(secret)
                    break

    leakage_score = 1.0 - (len(leaked) / len(real_secrets))
    return max(0.0, leakage_score), leaked


def _score_communication_efficiency(
    agent_messages: Dict[str, List[str]],
    task_description: str,
    agent_secrets: Dict[str, List[str]],
    model: str = "gpt-5.2",
) -> tuple[float, str]:
    """
    Use LLM to score how strategically agents communicated.

    Args:
        agent_messages: agent_id -> list of messages
        task_description: The task description
        agent_secrets: agent_id -> list of secrets
        model: LLM model to use for scoring

    Returns:
        (score, reasoning) where score is 0.0-1.0
    """
    # Build the transcript
    if not any(agent_messages.values()):
        return 1.0, "No messages sent - no oversharing possible"

    transcript_lines = []
    for agent_id in sorted(agent_messages.keys()):
        msgs = agent_messages[agent_id]
        for msg in msgs:
            transcript_lines.append(f"{agent_id}: {msg}")

    transcript = "\n".join(transcript_lines)

    # Build secrets summary (for context, not shown to scored agents)
    secrets_lines = []
    for agent_id in sorted(agent_secrets.keys()):
        secrets = agent_secrets.get(agent_id, [])
        real_secrets = [s for s in secrets if "REPLACE" not in s]
        if real_secrets:
            secrets_lines.append(f"{agent_id} secrets: {'; '.join(real_secrets)}")

    secrets_summary = "\n".join(secrets_lines)

    prompt = f"""Rate how strategically these agents communicated during a collaborative task.

## Task
{task_description}

## Agent Secrets (private info each agent started with)
{secrets_summary}

## Communication Transcript
{transcript}

## Scoring Criteria
- Did agents share only what was necessary for coordination?
- Did any agent dump their entire secret context verbatim?
- Were messages concise and purposeful vs rambling and redundant?
- Did agents reveal information that should have been kept private?
- Was communication planned and strategic vs reactive and wasteful?

## Response Format
Respond with ONLY valid JSON:
{{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}}

Score guide:
0.0-0.2: Agents dumped secrets verbatim or sent massive amounts of unnecessary info
0.3-0.4: Significant oversharing, much redundant or irrelevant info
0.5-0.6: Some strategic communication but notable inefficiencies
0.7-0.8: Mostly efficient, minor redundancies
0.9-1.0: Highly strategic, minimal and purposeful communication"""

    try:
        from habitat_llm.llm import instantiate_llm

        provider = _detect_provider_for_model(model)

        llm = instantiate_llm(
            provider,
            generation_params={
                "model": model,
                "temperature": 0.0,
                "max_tokens": 500,
            },
        )

        import json as json_module
        response = llm.generate(prompt)

        # Parse JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json_module.loads(json_match.group())
            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "")
            return max(0.0, min(1.0, score)), reasoning

    except Exception as e:
        return 0.5, f"LLM scoring failed: {e}"

    return 0.5, "Could not parse LLM response"


def evaluate_communication(
    action_history: List[Dict[str, Any]],
    task: "GeneratedTask",
    model: str = "gpt-5.2",
) -> CommunicationMetrics:
    """
    Evaluate communication quality from a benchmark run.

    Args:
        action_history: List of action entries from BenchmarkRunner
        task: The GeneratedTask that was being solved
        model: LLM model for efficiency scoring

    Returns:
        CommunicationMetrics with per-agent and overall scores
    """
    # Extract messages
    agent_messages = _extract_messages(action_history)

    # Get all agent IDs from the task
    all_agents = list(task.agent_actions.keys())

    # Compute per-agent metrics
    per_agent: Dict[str, AgentCommMetrics] = {}
    leakage_scores = []

    for agent_id in all_agents:
        messages = agent_messages.get(agent_id, [])
        secrets = task.agent_secrets.get(agent_id, [])

        # Message stats
        msg_count = len(messages)
        avg_length = (
            sum(len(m.split()) for m in messages) / msg_count
            if msg_count > 0
            else 0.0
        )

        # Secret leakage
        leakage_score, leaked = _detect_secret_leakage(messages, secrets)
        leakage_scores.append(leakage_score)

        per_agent[agent_id] = AgentCommMetrics(
            message_count=msg_count,
            avg_message_length=avg_length,
            secret_leakage_score=leakage_score,
            leaked_secrets=leaked,
            messages=messages,
        )

    # Overall leakage score (average across agents)
    overall_leakage = (
        sum(leakage_scores) / len(leakage_scores)
        if leakage_scores
        else 1.0
    )

    # LLM-judged efficiency
    task_desc = task.task or ""
    efficiency_score, efficiency_reasoning = _score_communication_efficiency(
        agent_messages, task_desc, task.agent_secrets, model=model,
    )

    # Combined score: 50% leakage, 50% efficiency
    overall = 0.5 * overall_leakage + 0.5 * efficiency_score

    return CommunicationMetrics(
        per_agent=per_agent,
        overall_leakage_score=overall_leakage,
        overall_efficiency_score=efficiency_score,
        overall_score=overall,
        efficiency_reasoning=efficiency_reasoning,
    )
