"""Diversity tracking for task generation.

Maintains a persistent log of structural patterns from generated tasks
to guide the LLM toward creating structurally diverse tasks.
"""

import json
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM


SUMMARIZE_PROMPT = """Summarize this task's STRUCTURAL PATTERN in one short phrase (5-15 words).

Focus on the MECHANICS, STRUCTURE, and WIN CONDITION - not the narrative:
- What is the win condition? (search, arrange objects, coordinate placement, etc.)
- What creates the core dependency? (information asymmetry, room restrictions, communication limits, etc.)
- What forces agent collaboration? (distributed knowledge, sequential handoffs, parallel workstreams, etc.)
- What mechanics are central? (remote triggers, state mirroring, communication restrictions, etc.)

Examples of good structural summaries:
- "Find scene object with distributed location knowledge"
- "Arrange objects with asymmetric action access"
- "Remote trigger changes shared-object state"
- "Room restriction forces sequential handoff through reachable areas"
- "Parallel collection with shared final placement objective"
- "Information asymmetry: one knows what, other knows where"

BAD summaries (too generic):
- "Find object" (missing: what makes it unique?)
- "Agents cooperate" (missing: how they cooperate?)

Task JSON:
{task_json}

Respond with ONLY the structural pattern phrase, nothing else."""


class DiversityTracker:
    """Tracks structural patterns of generated tasks for diversity."""

    # Persistent location for diversity log (shared across all runs)
    DEFAULT_LOG_DIR = Path("data/enacttom/meta")

    def __init__(self, llm: Optional["BaseLLM"] = None, log_dir: Optional[Path] = None):
        """
        Initialize diversity tracker.

        Args:
            llm: LLM for summarizing tasks into patterns (optional, for read-only mode)
            log_dir: Override directory for diversity_log.json (defaults to data/enacttom/tasks)
        """
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.log_file = self.log_dir / "diversity_log.json"
        self.llm = llm
        self.patterns: List[dict] = []
        self._load()

    def _load(self) -> None:
        """Load existing patterns from disk."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    self.patterns = data.get("patterns", [])
            except (json.JSONDecodeError, IOError):
                self.patterns = []

    def _save(self) -> None:
        """Save patterns to disk."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump({"patterns": self.patterns}, f, indent=2)

    def summarize_task(self, task_data: dict) -> str:
        """
        Use LLM to summarize a task into a structural pattern.

        Args:
            task_data: The task JSON data

        Returns:
            Short structural pattern description
        """
        if not self.llm:
            # Fallback: extract basic pattern from task structure
            return self._extract_basic_pattern(task_data)

        prompt = SUMMARIZE_PROMPT.format(task_json=json.dumps(task_data, indent=2))

        try:
            response = self.llm.generate(prompt)
            # Clean up response - take first line, strip whitespace
            pattern = response.strip().split("\n")[0].strip()
            # Remove quotes if present
            pattern = pattern.strip('"\'')
            return pattern[:100]  # Cap length
        except Exception:
            return self._extract_basic_pattern(task_data)

    def _extract_basic_pattern(self, task_data: dict) -> str:
        """Fallback pattern extraction without LLM."""
        parts = []

        # Try to infer win condition from task description and subtasks
        task_desc = task_data.get("task", "").lower()
        category = task_data.get("category", "")

        # Detect common win condition patterns
        if "race" in task_desc or "first" in task_desc or "before" in task_desc:
            parts.append("race")
        elif "collect" in task_desc or "gather" in task_desc:
            parts.append("collection")
        elif "arrange" in task_desc or "place" in task_desc or "organize" in task_desc:
            parts.append("arrangement")
        elif "find" in task_desc or "locate" in task_desc:
            parts.append("search")
        elif "unlock" in task_desc or "open" in task_desc:
            parts.append("unlocking")

        # Check mechanics
        mechanics = task_data.get("active_mechanics", [])
        if not mechanics:
            bindings = task_data.get("mechanic_bindings", []) or task_data.get("mechanics", [])
            extracted = []
            for b in bindings:
                if isinstance(b, dict):
                    mech_type = b.get("mechanic_type") or b.get("type")
                else:
                    mech_type = b
                if mech_type:
                    extracted.append(mech_type)
            # Preserve order, drop duplicates
            seen = set()
            mechanics = []
            for m in extracted:
                if m not in seen:
                    mechanics.append(m)
                    seen.add(m)
        if mechanics:
            parts.append(f"mechanics: {', '.join(mechanics[:2])}")

        # Check agent secrets
        secrets = task_data.get("agent_secrets", {})
        if any(secrets.values()):
            parts.append("information asymmetry")

        # Add category context
        if category:
            parts.insert(0, f"[{category}]")

        return " + ".join(parts) if parts else "basic task structure"

    def add_pattern(self, task_id: str, task_data: dict) -> str:
        """
        Summarize a task and add its pattern to the log.

        Args:
            task_id: Unique identifier for the task
            task_data: The task JSON data

        Returns:
            The generated pattern summary
        """
        pattern = self.summarize_task(task_data)

        self.patterns.append({
            "task_id": task_id,
            "pattern": pattern,
            "category": task_data.get("category", "unknown"),
            "num_agents": task_data.get("num_agents", 0),
            "num_subtasks": len(task_data.get("subtasks", [])),
        })

        self._save()
        return pattern

    def get_patterns_for_prompt(self, limit: int = 20) -> str:
        """
        Format recent patterns for injection into prompts.

        Args:
            limit: Maximum number of patterns to include

        Returns:
            Formatted string for prompt injection
        """
        if not self.patterns:
            return "No previous tasks yet. Be creative with your first task structure!"

        recent = self.patterns[-limit:]
        lines = []
        for i, p in enumerate(recent, 1):
            lines.append(f"{i}. [{p['category']}] {p['pattern']}")

        return "\n".join(lines)

    def get_pattern_count(self) -> int:
        """Return total number of tracked patterns."""
        return len(self.patterns)

    def check_novelty(self, task_data: dict) -> dict:
        """
        Check if a task is sufficiently novel compared to existing patterns.

        Args:
            task_data: The task JSON data to evaluate

        Returns:
            dict with:
                - score: 0.0-1.0 novelty score
                - reason: Explanation of the score
                - similar_to: List of similar existing patterns (if any)
        """
        if not self.patterns:
            return {
                "score": 1.0,
                "reason": "First task - automatically novel",
                "similar_to": []
            }

        # Get structural pattern for this task
        new_pattern = self.summarize_task(task_data)

        # If no LLM, do basic comparison
        if not self.llm:
            return self._basic_novelty_check(new_pattern, task_data)

        # Use LLM to compare against existing patterns
        existing_patterns = self.get_patterns_for_prompt(limit=30)

        prompt = f"""Compare this new task's structure to existing tasks and rate its NOVELTY.

NEW TASK PATTERN: {new_pattern}

NEW TASK JSON:
{json.dumps(task_data, indent=2)[:3000]}

EXISTING TASK PATTERNS:
{existing_patterns}

Evaluate:
1. Is the WIN CONDITION structurally different from existing tasks?
2. Are the MECHANICS used in a novel way?
3. Is the DEPENDENCY STRUCTURE (what forces collaboration) different?
4. Does it feel like a fresh design or a reskin of an existing pattern?

Respond in this exact format:
SCORE: <0.0-1.0>
SIMILAR_TO: <comma-separated list of pattern numbers that are similar, or "none">
REASON: <one sentence explanation>

Scoring guide:
- 1.0: Completely novel structure, nothing similar exists
- 0.7-0.9: Mostly novel with some minor similarities
- 0.4-0.6: Shares significant structural elements with existing tasks
- 0.1-0.3: Very similar to existing task(s), feels like a reskin
- 0.0: Nearly identical to an existing task"""

        try:
            response = self.llm.generate(prompt)
            return self._parse_novelty_response(response)
        except Exception as e:
            return {
                "score": 0.5,
                "reason": f"Novelty check failed: {e}",
                "similar_to": []
            }

    def _basic_novelty_check(self, new_pattern: str, task_data: dict) -> dict:
        """Fallback novelty check without LLM - uses simple string matching."""
        new_pattern_lower = new_pattern.lower()
        similar = []

        for i, p in enumerate(self.patterns):
            existing = p["pattern"].lower()
            # Simple word overlap check
            new_words = set(new_pattern_lower.split())
            existing_words = set(existing.split())
            overlap = len(new_words & existing_words) / max(len(new_words | existing_words), 1)
            if overlap > 0.5:
                similar.append(f"{i+1}. {p['pattern']}")

        if not similar:
            return {"score": 0.8, "reason": "No obvious pattern matches", "similar_to": []}
        elif len(similar) <= 2:
            return {"score": 0.5, "reason": f"Some similarity to existing patterns", "similar_to": similar}
        else:
            return {"score": 0.2, "reason": "High similarity to multiple existing patterns", "similar_to": similar}

    def _parse_novelty_response(self, response: str) -> dict:
        """Parse LLM novelty check response."""
        result = {"score": 0.5, "reason": "Could not parse response", "similar_to": []}

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    result["score"] = float(score_str)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("SIMILAR_TO:"):
                try:
                    similar_str = line.split(":", 1)[1].strip().lower()
                    if similar_str != "none" and similar_str:
                        result["similar_to"] = [s.strip() for s in similar_str.split(",")]
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASON:"):
                try:
                    result["reason"] = line.split(":", 1)[1].strip()
                except (ValueError, IndexError):
                    pass

        return result
