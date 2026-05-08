"""Structured generation logging utilities.

Task-generation workspaces under ``tmp/task_gen`` keep only transient working
state. Visualizer-facing logs live under ``outputs/generations/<run_id>`` and
are written via the helpers in this module.

Logging is best-effort: failures are swallowed so monitoring can never break
task generation itself.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

EVENTS_FILENAME = "events.jsonl"
WORKER_FILENAME = "worker.json"
MANIFEST_FILENAME = "manifest.json"


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_event(log_dir: str, event_type: str, **payload: Any) -> None:
    """Append one normalized event to ``events.jsonl`` inside ``log_dir``."""

    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, EVENTS_FILENAME)
        event: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **payload,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        return


def write_worker_snapshot(worker_dir: str, **snapshot: Any) -> None:
    """Create or update ``worker.json`` for a generation worker."""

    try:
        os.makedirs(worker_dir, exist_ok=True)
        path = os.path.join(worker_dir, WORKER_FILENAME)
        current = _read_json(path)
        current.update(snapshot)
        _write_json(path, current)
    except Exception:
        return


def write_run_manifest(run_dir: str, **manifest_fields: Any) -> None:
    """Create or update ``manifest.json`` for a generation run."""

    try:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, MANIFEST_FILENAME)
        current = _read_json(path)
        current.update(manifest_fields)
        _write_json(path, current)
    except Exception:
        return


def load_worker_snapshot(worker_dir: str) -> Dict[str, Any]:
    return _read_json(os.path.join(worker_dir, WORKER_FILENAME))


def load_run_manifest(run_dir: str) -> Dict[str, Any]:
    return _read_json(os.path.join(run_dir, MANIFEST_FILENAME))


def maybe_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
