"""
DAG utilities for subtask dependency management.

Tasks are represented as DAGs where:
- Nodes are subtasks with individual success_conditions
- Edges are depends_on relationships
- Progress is measured by completed nodes
- Success is achieved when all terminal nodes are completed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .task_generator import Subtask


def get_subtask_id(subtask: "Subtask") -> str:
    """Get the ID of a subtask, supporting both 'id' and 'subtask_id' fields."""
    if hasattr(subtask, "id"):
        return subtask.id
    return subtask.subtask_id


def find_terminal_nodes(subtasks: List["Subtask"]) -> List["Subtask"]:
    """
    Find terminal nodes (nodes with no dependents).

    Terminal nodes represent the final outcomes of the task.
    Task success = all terminal nodes have been completed.

    Args:
        subtasks: List of Subtask objects

    Returns:
        List of subtasks that are not depended on by any other subtask
    """
    all_ids = {get_subtask_id(s) for s in subtasks}
    depended_on: Set[str] = set()

    for s in subtasks:
        depended_on.update(s.depends_on)

    terminal_ids = all_ids - depended_on
    return [s for s in subtasks if get_subtask_id(s) in terminal_ids]


def find_root_nodes(subtasks: List["Subtask"]) -> List["Subtask"]:
    """
    Find root nodes (nodes with no dependencies).

    Root nodes can be started immediately without waiting for other subtasks.

    Args:
        subtasks: List of Subtask objects

    Returns:
        List of subtasks that have no dependencies
    """
    return [s for s in subtasks if not s.depends_on]


def validate_dag(subtasks: List["Subtask"]) -> Tuple[bool, List[str]]:
    """
    Validate the subtask DAG structure.

    Checks:
    1. All depends_on references point to existing subtask IDs
    2. No cycles exist (valid topological ordering possible)
    3. At least one terminal node exists
    4. All nodes are reachable from root nodes
    5. Each subtask has a non-empty success_condition

    Args:
        subtasks: List of Subtask objects

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if not subtasks:
        errors.append("No subtasks defined")
        return False, errors

    ids = {get_subtask_id(s) for s in subtasks}

    # Check for duplicate IDs
    if len(ids) != len(subtasks):
        errors.append("Duplicate subtask IDs found")

    # Check all depends_on references exist
    for s in subtasks:
        sid = get_subtask_id(s)
        for dep in s.depends_on:
            if dep not in ids:
                errors.append(f"Subtask '{sid}' depends on unknown subtask '{dep}'")

        # Check for self-dependency
        if sid in s.depends_on:
            errors.append(f"Subtask '{sid}' depends on itself")

    # Check for cycles using DFS
    if _has_cycle(subtasks):
        errors.append("DAG contains a cycle")

    # Check at least one terminal node exists
    terminal = find_terminal_nodes(subtasks)
    if not terminal:
        errors.append("No terminal nodes found (every subtask is depended on)")

    # Check all nodes reachable from roots
    roots = find_root_nodes(subtasks)
    if not roots:
        errors.append("No root nodes found (every subtask has dependencies)")
    else:
        reachable = _find_reachable(subtasks, roots)
        unreachable = ids - reachable
        if unreachable:
            errors.append(f"Unreachable subtasks: {unreachable}")

    # Check each subtask has a success_condition
    for s in subtasks:
        sid = get_subtask_id(s)
        if not s.success_condition:
            errors.append(f"Subtask '{sid}' has no success_condition")
        elif not isinstance(s.success_condition, dict):
            errors.append(f"Subtask '{sid}' has invalid success_condition type")
        elif "entity" not in s.success_condition or "property" not in s.success_condition:
            errors.append(f"Subtask '{sid}' success_condition missing 'entity' or 'property'")

    # Check that no two sequential nodes have the same success condition
    # Sequential = one directly depends on the other
    subtask_map = {get_subtask_id(s): s for s in subtasks}
    for s in subtasks:
        sid = get_subtask_id(s)
        if not s.success_condition:
            continue
        s_cond = _normalize_condition(s.success_condition)
        for dep_id in s.depends_on:
            dep = subtask_map.get(dep_id)
            if dep and dep.success_condition:
                dep_cond = _normalize_condition(dep.success_condition)
                if s_cond == dep_cond:
                    errors.append(
                        f"Sequential subtasks '{dep_id}' -> '{sid}' have identical "
                        f"success_condition ({s_cond}). Each step should represent distinct progress."
                    )

    # Check for duplicate success conditions across ALL subtasks
    seen_conditions: Dict[str, str] = {}  # normalized_condition -> subtask_id
    for s in subtasks:
        sid = get_subtask_id(s)
        if not s.success_condition:
            continue
        s_cond = _normalize_condition(s.success_condition)
        if s_cond in seen_conditions:
            errors.append(
                f"Subtasks '{seen_conditions[s_cond]}' and '{sid}' have identical "
                f"success_condition. Each subtask must have a unique condition."
            )
        else:
            seen_conditions[s_cond] = sid

    # Check for predicates that cause "free progress" (true at start)
    # unless properly gated by a preceding complementary predicate
    for s in subtasks:
        sid = get_subtask_id(s)
        if not s.success_condition:
            continue
        prop = s.success_condition.get("property", "")
        entity = s.success_condition.get("entity", "")

        # is_closed is only valid if preceded by is_open on the same entity
        if prop == "is_closed":
            has_open_predecessor = _has_predecessor_with_property(
                subtasks, subtask_map, s, entity, "is_open"
            )
            if not has_open_predecessor:
                errors.append(
                    f"Subtask '{sid}' uses 'is_closed' on {entity} without a preceding "
                    f"'is_open' subtask. Containers start closed, so this completes instantly."
                )

        # is_locked should never be used - use is_unlocked instead
        if prop == "is_locked":
            errors.append(
                f"Subtask '{sid}' uses 'is_locked'. Use 'is_unlocked' instead to "
                f"track unlocking progress."
            )

    return len(errors) == 0, errors


def _normalize_condition(condition: Dict[str, Any]) -> str:
    """
    Normalize a success_condition to a comparable string.

    Handles both 'value' and 'target' fields, sorts keys for consistency.
    """
    if not condition:
        return ""
    # Extract key fields in a consistent order
    entity = condition.get("entity", "")
    prop = condition.get("property", "")
    # Use 'target' if present, otherwise 'value'
    target_or_value = condition.get("target", condition.get("value", ""))
    return f"{entity}:{prop}:{target_or_value}"


def _has_predecessor_with_property(
    subtasks: List["Subtask"],
    subtask_map: Dict[str, "Subtask"],
    current: "Subtask",
    entity: str,
    required_prop: str,
) -> bool:
    """
    Check if there's a predecessor subtask with the given property on the same entity.

    Uses BFS to traverse dependency chain backwards.
    """
    visited: Set[str] = set()
    queue = list(current.depends_on)

    while queue:
        dep_id = queue.pop(0)
        if dep_id in visited:
            continue
        visited.add(dep_id)

        dep = subtask_map.get(dep_id)
        if not dep or not dep.success_condition:
            continue

        # Check if this predecessor has the required property on the same entity
        dep_entity = dep.success_condition.get("entity", "")
        dep_prop = dep.success_condition.get("property", "")
        if dep_entity == entity and dep_prop == required_prop:
            return True

        # Add this predecessor's dependencies to the queue
        queue.extend(dep.depends_on)

    return False


def _has_cycle(subtasks: List["Subtask"]) -> bool:
    """Check if the DAG contains a cycle using DFS."""
    id_to_subtask = {get_subtask_id(s): s for s in subtasks}
    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(sid: str) -> bool:
        visited.add(sid)
        rec_stack.add(sid)

        subtask = id_to_subtask.get(sid)
        if subtask:
            for dep in subtask.depends_on:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True

        rec_stack.remove(sid)
        return False

    for s in subtasks:
        sid = get_subtask_id(s)
        if sid not in visited:
            if dfs(sid):
                return True

    return False


def _find_reachable(subtasks: List["Subtask"], roots: List["Subtask"]) -> Set[str]:
    """Find all nodes reachable from the given root nodes."""
    # Build reverse adjacency (who depends on whom -> who is depended on by whom)
    id_to_subtask = {get_subtask_id(s): s for s in subtasks}
    dependents: Dict[str, List[str]] = {get_subtask_id(s): [] for s in subtasks}

    for s in subtasks:
        sid = get_subtask_id(s)
        for dep in s.depends_on:
            if dep in dependents:
                dependents[dep].append(sid)

    # BFS from roots
    reachable: Set[str] = set()
    queue = [get_subtask_id(r) for r in roots]

    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        queue.extend(dependents.get(current, []))

    return reachable


def topological_sort(subtasks: List["Subtask"]) -> List["Subtask"]:
    """
    Return subtasks in topological order (dependencies before dependents).

    Args:
        subtasks: List of Subtask objects

    Returns:
        List of subtasks in dependency order

    Raises:
        ValueError: If the graph contains a cycle
    """
    id_to_subtask = {get_subtask_id(s): s for s in subtasks}

    # Kahn's algorithm
    in_degree = {get_subtask_id(s): len(s.depends_on) for s in subtasks}
    queue = [sid for sid, degree in in_degree.items() if degree == 0]
    result: List["Subtask"] = []

    while queue:
        sid = queue.pop(0)
        result.append(id_to_subtask[sid])

        # Reduce in-degree for dependents
        for s in subtasks:
            if sid in s.depends_on:
                other_id = get_subtask_id(s)
                in_degree[other_id] -= 1
                if in_degree[other_id] == 0:
                    queue.append(other_id)

    if len(result) != len(subtasks):
        raise ValueError("Cannot topologically sort: graph contains a cycle")

    return result


@dataclass
class DAGProgress:
    """
    Track progress through a subtask DAG with latching behavior.

    NOTE: This class tracks PROGRESS, not task success. Task success is
    determined by required subtasks in benchmark.py, which may use different
    logic (e.g., checking required conditions directly without DAG gating).

    Subtasks "latch" when their condition is satisfied - once completed,
    they stay completed even if the state changes later. This is because
    intermediate states are transient (e.g., "holding kettle" -> "placed kettle").

    Progress is monotonic: it can only increase, never decrease.
    """

    subtasks: Dict[str, "Subtask"] = field(default_factory=dict)
    completed: Set[str] = field(default_factory=set)
    terminal_ids: Set[str] = field(default_factory=set)  # Legacy, kept for compatibility

    @classmethod
    def from_subtasks(cls, subtasks: List["Subtask"]) -> "DAGProgress":
        """Create a DAGProgress tracker from a list of subtasks."""
        subtask_dict = {get_subtask_id(s): s for s in subtasks}
        terminal_ids = {get_subtask_id(s) for s in find_terminal_nodes(subtasks)}
        return cls(subtasks=subtask_dict, terminal_ids=terminal_ids)

    def update(
        self,
        check_condition: Callable[["Subtask"], bool],
    ) -> Dict[str, Any]:
        """
        Check conditions and latch any newly completed subtasks.

        Args:
            check_condition: Function that takes a Subtask and returns True
                           if its success_condition is currently satisfied

        Returns:
            Dict with:
            - completed: List of all completed subtask IDs
            - newly_completed: List of subtask IDs completed in this update
            - percent_complete: Float 0.0-1.0
            - success: True if all terminal nodes are completed
        """
        newly_completed = []

        for sid, subtask in self.subtasks.items():
            if sid in self.completed:
                continue  # Already latched

            # Check if all dependencies are satisfied
            deps_met = all(dep in self.completed for dep in subtask.depends_on)
            if not deps_met:
                continue

            # Check if this subtask's condition is currently satisfied
            if check_condition(subtask):
                self.completed.add(sid)
                newly_completed.append(sid)

        total = len(self.subtasks) if self.subtasks else 1
        percent = len(self.completed) / total

        return {
            "completed": list(self.completed),
            "newly_completed": newly_completed,
            "percent_complete": percent,
            "all_complete": self.all_subtasks_complete(),
        }

    def all_subtasks_complete(self) -> bool:
        """
        Check if all subtasks in the DAG have been completed.

        NOTE: This is for progress tracking, not task success. Task success
        is determined by required subtasks evaluated in benchmark.py.
        """
        return len(self.completed) == len(self.subtasks)

    def get_status(self) -> Dict[str, Any]:
        """Get current progress status without updating."""
        total = len(self.subtasks) if self.subtasks else 1
        return {
            "completed": list(self.completed),
            "percent_complete": len(self.completed) / total,
            "all_complete": self.all_subtasks_complete(),
            "remaining": [
                sid for sid in self.subtasks.keys() if sid not in self.completed
            ],
        }

    def reset(self) -> None:
        """Reset progress (clear all completed subtasks)."""
        self.completed.clear()
