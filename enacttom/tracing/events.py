"""
Causal Event Log for EnactToM.

Provides structured event logging with causal links (caused_by) forming
a DAG. This enables debugging "why did this happen?" and exact replay.

Borrowed from ARE's event system pattern.

Usage:
    log = EventLog()
    e1 = log.log_action(step=0, agent_id="agent_0", action="Open", target="chest_1", result="success")
    e2 = log.log_mechanic(step=0, mechanic="inverse_state", trigger="chest_1", effect="closed instead", caused_by=e1.event_id)

    # Debug: what caused this state?
    chain = log.get_causal_chain(e2.event_id)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """Types of events in the trace."""
    ACTION = "action"           # Agent performed an action
    MECHANIC = "mechanic"       # Mechanic transform was applied
    STATE_CHANGE = "state_change"  # Environment state changed
    OBSERVATION = "observation"  # Agent received observation
    COMMUNICATION = "communication"  # Agent sent message
    MILESTONE = "milestone"     # Milestone was reached
    MINEFIELD = "minefield"     # Minefield was triggered
    SYSTEM = "system"           # System event (tick, reset, etc.)


@dataclass
class Event:
    """
    A single event in the trace log.

    Events form a DAG via the caused_by field, enabling causal analysis.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: EventType = EventType.ACTION
    step: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Action-specific fields
    agent_id: Optional[str] = None
    action: Optional[str] = None
    target: Optional[str] = None
    result: Optional[str] = None
    success: bool = True

    # Causal link - which event caused this one
    caused_by: Optional[str] = None

    # Additional details (mechanic name, effect description, etc.)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dict."""
        data = data.copy()
        if "event_type" in data:
            data["event_type"] = EventType(data["event_type"])
        return cls(**data)

    def __repr__(self) -> str:
        parts = [f"Event({self.event_id}"]
        parts.append(f"type={self.event_type.value}")
        parts.append(f"step={self.step}")
        if self.agent_id:
            parts.append(f"agent={self.agent_id}")
        if self.action:
            parts.append(f"action={self.action}")
        if self.target:
            parts.append(f"target={self.target}")
        if self.caused_by:
            parts.append(f"caused_by={self.caused_by}")
        return ", ".join(parts) + ")"


class EventLog:
    """
    Structured event log with causal links.

    Provides:
    - Logging of actions, mechanics, state changes
    - Causal chain analysis
    - JSON export/import
    - Filtering and querying
    """

    def __init__(self):
        self.events: List[Event] = []
        self._last_event_id: Optional[str] = None
        self._step_events: Dict[int, List[str]] = {}  # step -> event_ids for fast lookup

    def log_action(
        self,
        step: int,
        agent_id: str,
        action: str,
        target: Optional[str],
        result: str,
        success: bool = True,
        caused_by: Optional[str] = None,
    ) -> Event:
        """
        Log an agent action.

        Args:
            step: Simulation step
            agent_id: Agent that performed the action
            action: Action name (e.g., "Open", "Navigate")
            target: Target of the action
            result: Result/observation text
            success: Whether the action succeeded
            caused_by: Event ID that caused this (defaults to last event)

        Returns:
            The created Event
        """
        e = Event(
            event_type=EventType.ACTION,
            step=step,
            agent_id=agent_id,
            action=action,
            target=target,
            result=result,
            success=success,
            caused_by=caused_by or self._last_event_id,
        )
        self._add_event(e)
        return e

    def log_mechanic(
        self,
        step: int,
        mechanic: str,
        trigger: str,
        effect: str,
        caused_by: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Log a mechanic transform.

        Args:
            step: Simulation step
            mechanic: Mechanic type (e.g., "inverse_state")
            trigger: Object that triggered the mechanic
            effect: Description of the effect
            caused_by: Event ID of the action that triggered this
            details: Additional details

        Returns:
            The created Event
        """
        e = Event(
            event_type=EventType.MECHANIC,
            step=step,
            caused_by=caused_by,
            details={
                "mechanic": mechanic,
                "trigger": trigger,
                "effect": effect,
                **(details or {}),
            },
        )
        self._add_event(e)
        return e

    def log_state_change(
        self,
        step: int,
        entity: str,
        property_name: str,
        old_value: Any,
        new_value: Any,
        caused_by: Optional[str] = None,
    ) -> Event:
        """
        Log a state change.

        Args:
            step: Simulation step
            entity: Entity that changed
            property_name: Property that changed
            old_value: Previous value
            new_value: New value
            caused_by: Event ID that caused this change

        Returns:
            The created Event
        """
        e = Event(
            event_type=EventType.STATE_CHANGE,
            step=step,
            target=entity,
            caused_by=caused_by or self._last_event_id,
            details={
                "property": property_name,
                "old_value": old_value,
                "new_value": new_value,
            },
        )
        self._add_event(e)
        return e

    def log_communication(
        self,
        step: int,
        from_agent: str,
        to_agent: str,
        message: str,
        caused_by: Optional[str] = None,
    ) -> Event:
        """
        Log inter-agent communication.

        Args:
            step: Simulation step
            from_agent: Sender agent ID
            to_agent: Receiver agent ID
            message: Message content
            caused_by: Event ID that triggered this

        Returns:
            The created Event
        """
        e = Event(
            event_type=EventType.COMMUNICATION,
            step=step,
            agent_id=from_agent,
            result=message,
            caused_by=caused_by or self._last_event_id,
            details={
                "from": from_agent,
                "to": to_agent,
                "message": message,
            },
        )
        self._add_event(e)
        return e

    def log_milestone(
        self,
        step: int,
        milestone_id: str,
        description: str,
        caused_by: Optional[str] = None,
    ) -> Event:
        """
        Log a milestone being reached.

        Args:
            step: Simulation step
            milestone_id: ID of the milestone
            description: Description of the milestone
            caused_by: Event ID that caused this

        Returns:
            The created Event
        """
        e = Event(
            event_type=EventType.MILESTONE,
            step=step,
            caused_by=caused_by or self._last_event_id,
            details={
                "milestone_id": milestone_id,
                "description": description,
            },
        )
        self._add_event(e)
        return e

    def log_minefield(
        self,
        step: int,
        minefield_id: str,
        description: str,
        caused_by: Optional[str] = None,
    ) -> Event:
        """
        Log a minefield being triggered.

        Args:
            step: Simulation step
            minefield_id: ID of the minefield
            description: Description of what went wrong
            caused_by: Event ID that caused this

        Returns:
            The created Event
        """
        e = Event(
            event_type=EventType.MINEFIELD,
            step=step,
            caused_by=caused_by or self._last_event_id,
            details={
                "minefield_id": minefield_id,
                "description": description,
            },
        )
        self._add_event(e)
        return e

    def log_system(
        self,
        step: int,
        event_name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Log a system event (tick, reset, etc.).

        Args:
            step: Simulation step
            event_name: Name of the system event
            details: Additional details

        Returns:
            The created Event
        """
        e = Event(
            event_type=EventType.SYSTEM,
            step=step,
            action=event_name,
            details=details or {},
        )
        self._add_event(e)
        return e

    def _add_event(self, event: Event) -> None:
        """Add event to the log."""
        self.events.append(event)
        self._last_event_id = event.event_id
        self._step_events.setdefault(event.step, []).append(event.event_id)

    def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        for e in self.events:
            if e.event_id == event_id:
                return e
        return None

    def get_causal_chain(self, event_id: str) -> List[Event]:
        """
        Walk back the causal chain from an event.

        Returns events in chronological order (oldest first).

        Args:
            event_id: Event to trace back from

        Returns:
            List of events in the causal chain
        """
        chain = []
        current = self.get_event(event_id)
        while current:
            chain.append(current)
            current = self.get_event(current.caused_by) if current.caused_by else None
        return list(reversed(chain))

    def get_effects(self, event_id: str) -> List[Event]:
        """
        Get all events caused by a given event.

        Args:
            event_id: Event to find effects of

        Returns:
            List of events caused by this one
        """
        return [e for e in self.events if e.caused_by == event_id]

    def get_events_at_step(self, step: int) -> List[Event]:
        """Get all events at a given step."""
        event_ids = self._step_events.get(step, [])
        return [e for e in self.events if e.event_id in event_ids]

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a given type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_agent_events(self, agent_id: str) -> List[Event]:
        """Get all events for a given agent."""
        return [e for e in self.events if e.agent_id == agent_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire log to dict."""
        return {
            "events": [e.to_dict() for e in self.events],
            "total_events": len(self.events),
            "steps_covered": len(self._step_events),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> "EventLog":
        """Load from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        log = cls()
        for event_data in data.get("events", []):
            event = Event.from_dict(event_data)
            log._add_event(event)
        return log

    def to_narrative(self) -> str:
        """Generate human-readable narrative of the trace."""
        lines = []
        current_step = -1

        for event in self.events:
            if event.step != current_step:
                current_step = event.step
                lines.append(f"\n--- Step {current_step} ---")

            if event.event_type == EventType.ACTION:
                status = "✓" if event.success else "✗"
                target_str = f"[{event.target}]" if event.target else ""
                lines.append(f"  {status} {event.agent_id}: {event.action}{target_str}")
                if event.result:
                    lines.append(f"      → {event.result[:80]}...")

            elif event.event_type == EventType.MECHANIC:
                mechanic = event.details.get("mechanic", "unknown")
                effect = event.details.get("effect", "")
                lines.append(f"  ⚙ Mechanic [{mechanic}]: {effect}")

            elif event.event_type == EventType.STATE_CHANGE:
                prop = event.details.get("property", "?")
                old = event.details.get("old_value", "?")
                new = event.details.get("new_value", "?")
                lines.append(f"  Δ {event.target}.{prop}: {old} → {new}")

            elif event.event_type == EventType.COMMUNICATION:
                msg = event.details.get("message", "")
                to = event.details.get("to", "?")
                lines.append(f"  💬 {event.agent_id} → {to}: {msg[:60]}...")

            elif event.event_type == EventType.MILESTONE:
                desc = event.details.get("description", "")
                lines.append(f"  🎯 Milestone: {desc}")

            elif event.event_type == EventType.MINEFIELD:
                desc = event.details.get("description", "")
                lines.append(f"  💥 Minefield: {desc}")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the trace."""
        by_type = {}
        for e in self.events:
            by_type[e.event_type.value] = by_type.get(e.event_type.value, 0) + 1

        by_agent = {}
        for e in self.events:
            if e.agent_id:
                by_agent[e.agent_id] = by_agent.get(e.agent_id, 0) + 1

        success_count = sum(1 for e in self.events if e.event_type == EventType.ACTION and e.success)
        fail_count = sum(1 for e in self.events if e.event_type == EventType.ACTION and not e.success)

        return {
            "total_events": len(self.events),
            "steps_covered": len(self._step_events),
            "events_by_type": by_type,
            "events_by_agent": by_agent,
            "actions_succeeded": success_count,
            "actions_failed": fail_count,
        }
