"""Helpers for strict task metadata checks shared across CLI surfaces."""

from __future__ import annotations

from typing import Any, Dict, Optional


def compute_strict_tom_metadata(
    task_data: Dict[str, Any],
    scene_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute authoritative ToM metadata from canonical problem_pddl."""
    from enacttom.pddl.tom_verifier import (
        explain_tom_depth,
        prove_minimal_tom_level,
    )
    from enacttom.task_gen.task_generator import GeneratedTask

    generated = GeneratedTask.from_dict(task_data)
    proof = prove_minimal_tom_level(generated, scene_data=scene_data, strict=True)
    backend = proof.get("proof_backend")
    if backend != "fast_downward_strict":
        raise ValueError(
            "Strict ToM verification requires the Fast Downward strict backend; "
            f"got {backend!r}."
        )
    solver_result = proof.get("solver_result")
    if solver_result is None:
        last_error = (
            proof["proof_attempts"][-1]["error"]
            if proof.get("proof_attempts")
            else "unknown reason"
        )
        raise ValueError(
            "Strict Fast Downward ToM proof failed. "
            f"{last_error}"
        )

    tom_info = explain_tom_depth(
        generated,
        scene_data,
        solver_result=solver_result,
        proof=proof,
    )
    tom_info["epistemic_goal_depth"] = proof["epistemic_goal_depth"]
    tom_info["proved_unsat_below"] = proof["proved_unsat_below"]
    tom_info["proof_backend"] = proof["proof_backend"]
    tom_info["proof_strict"] = proof["proof_strict"]
    tom_level = tom_info.get("tom_level")
    if not isinstance(tom_level, int):
        raise ValueError(f"Invalid computed tom_level: {tom_level!r}")

    result: Dict[str, Any] = {
        "tom_level": tom_level,
        "epistemic_goal_depth": tom_info["epistemic_goal_depth"],
        "proved_unsat_below": tom_info["proved_unsat_below"],
        "proof_backend": tom_info["proof_backend"],
        "proof_strict": tom_info["proof_strict"],
    }
    tom_reasoning = tom_info.get("tom_reasoning")
    if isinstance(tom_reasoning, str) and tom_reasoning.strip():
        result["tom_reasoning"] = tom_reasoning
    return result
