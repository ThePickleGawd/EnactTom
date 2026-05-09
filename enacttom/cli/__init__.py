"""
Shared CLI infrastructure for EnactToM tools.

Every CLI module's run() returns a CLIResult dict.
Agent.py imports and calls run() directly (pure-function tools)
or spawns ``python -m enacttom.cli.<module>`` (GL-context tools).
External callers (Claude Code, run.sh) read JSON from stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional, TypedDict


class CLIResult(TypedDict):
    success: bool
    data: Dict[str, Any]
    error: Optional[str]


def success(data: Dict[str, Any]) -> CLIResult:
    """Build a successful CLIResult."""
    return {"success": True, "data": data, "error": None}


def failure(error: str, data: Optional[Dict[str, Any]] = None) -> CLIResult:
    """Build a failed CLIResult."""
    return {"success": False, "data": data or {}, "error": error}


def print_result(result: CLIResult) -> None:
    """Print CLIResult as JSON to stdout (the contract for external callers)."""
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()
