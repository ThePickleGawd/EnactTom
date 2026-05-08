"""
Runtime task semantics derived from canonical problem_pddl.

Runtime success uses only functional (non-epistemic) goals. Epistemic goals are
retained as separate end-of-episode literal Theory-of-Mind probes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from enacttom.pddl.describe import goal_to_natural_language
from enacttom.pddl.dsl import (
    And,
    Believes,
    EpistemicFormula,
    Formula,
    Knows,
    Literal,
    Not,
    Or,
    parse_goal_string,
)
from enacttom.pddl.problem_pddl import ParsedProblemPDDL, parse_problem_pddl


@dataclass(frozen=True)
class LiteralToMProbe:
    """Deterministic end-of-episode probe derived from a K() goal."""

    probe_id: str
    agent_id: str
    subject_agents: Tuple[str, ...]
    fact_pddl: str
    fact_natural_language: str
    source_pddl: str
    question: str
    expected_response: Dict[str, Any]
    depth: int
    owner: Optional[str] = None
    supported: bool = True
    unsupported_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeProjection:
    """Projected runtime semantics for benchmark execution."""

    functional_goal: Optional[Formula]
    functional_goal_pddl: Optional[str]
    functional_owners: Dict[str, str]
    probes: Tuple[LiteralToMProbe, ...]
    epistemic_conjuncts_removed: int
    invalid_reasons: Tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.invalid_reasons and self.functional_goal is not None


def project_runtime_from_problem(problem_pddl: str) -> RuntimeProjection:
    """Project physical runtime semantics and literal-ToM probes from problem_pddl."""
    parsed = parse_problem_pddl(problem_pddl)
    return project_runtime_from_parsed_problem(parsed)


def project_runtime_from_parsed_problem(parsed: ParsedProblemPDDL) -> RuntimeProjection:
    """Project physical runtime semantics and literal-ToM probes from a parsed problem."""
    invalid_reasons: List[str] = []
    removed_count = 0

    def _project_formula(node: Formula, *, count_removed: bool = True) -> Optional[Formula]:
        nonlocal removed_count

        if isinstance(node, EpistemicFormula):
            if count_removed:
                removed_count += 1
            return None

        if isinstance(node, And):
            projected_ops = [
                proj
                for op in node.operands
                if (proj := _project_formula(op, count_removed=count_removed)) is not None
            ]
            if not projected_ops:
                return None
            if len(projected_ops) == 1:
                return projected_ops[0]
            return And(operands=tuple(projected_ops))

        if isinstance(node, Or):
            projected_ops: List[Formula] = []
            for idx, op in enumerate(node.operands):
                projected = _project_formula(op, count_removed=count_removed)
                if projected is None:
                    invalid_reasons.append(
                        f"Runtime functional projection removed all goals from OR branch {idx}."
                    )
                    continue
                projected_ops.append(projected)
            if not projected_ops:
                return None
            if len(projected_ops) == 1:
                return projected_ops[0]
            return Or(operands=tuple(projected_ops))

        if isinstance(node, Not):
            projected = _project_formula(node.operand, count_removed=count_removed)
            if projected is None:
                invalid_reasons.append("Runtime functional projection cannot preserve epistemic negation.")
                return None
            return Not(operand=projected)

        return node

    functional_goal = _project_formula(parsed.goal_formula)
    functional_goal_pddl = functional_goal.to_pddl() if functional_goal is not None else None

    functional_owners: Dict[str, str] = {}
    for original_pddl, owner in (parsed.owners or {}).items():
        try:
            original_formula = parse_problem_pddl(
                f"(define (problem owner_projection) (:domain {parsed.domain_name}) (:init) (:goal {original_pddl}))"
            ).goal_formula
        except ValueError:
            continue
        projected_formula = _project_formula(original_formula, count_removed=False)
        if projected_formula is None:
            continue
        projected_pddl = projected_formula.to_pddl()
        existing_owner = functional_owners.get(projected_pddl)
        if existing_owner and existing_owner != owner:
            invalid_reasons.append(
                f"Projected runtime owner conflict for {projected_pddl}: {existing_owner} vs {owner}."
            )
            continue
        functional_owners[projected_pddl] = owner

    probes = tuple(derive_literal_tom_probes(parsed.goal_formula, owners=parsed.owners or {}))

    if functional_goal is None:
        invalid_reasons.append("No non-epistemic runtime goal remains after removing epistemic goals.")

    return RuntimeProjection(
        functional_goal=functional_goal,
        functional_goal_pddl=functional_goal_pddl,
        functional_owners=functional_owners,
        probes=probes,
        epistemic_conjuncts_removed=removed_count,
        invalid_reasons=tuple(invalid_reasons),
    )


def derive_literal_tom_probes(
    formula: Formula,
    *,
    owners: Optional[Dict[str, str]] = None,
) -> List[LiteralToMProbe]:
    """Build deterministic literal-ToM probes from K() formulas."""
    probes: List[LiteralToMProbe] = []
    owners = owners or {}

    def _walk(node: Formula) -> None:
        if isinstance(node, Knows):
            probes.append(_build_probe(node, probe_idx=len(probes), owner=owners.get(node.to_pddl())))
            _walk(node.inner)
            return
        if isinstance(node, Believes):
            probes.append(
                _build_unsupported_probe(
                    node,
                    probe_idx=len(probes),
                    owner=owners.get(node.to_pddl()),
                    reason="B() runtime probes are not implemented yet.",
                )
            )
            _walk(node.inner)
            return
        if isinstance(node, Not):
            _walk(node.operand)
            return
        if isinstance(node, (And, Or)):
            for op in node.operands:
                _walk(op)

    _walk(formula)
    return probes


def evaluate_literal_tom_probe(
    probe: LiteralToMProbe,
    response_data: Dict[str, Any],
    check_fn,
) -> Tuple[bool, Dict[str, Any]]:
    """Score a single literal-ToM probe deterministically."""
    if not probe.supported:
        return False, {
            "status": "unsupported",
            "reason": probe.unsupported_reason or "Unsupported probe.",
        }

    predicate = str(response_data.get("predicate", "")).strip()
    holds_raw = response_data.get("holds")
    raw_args = response_data.get("args", []) or []
    args = tuple(str(x).strip() for x in raw_args)
    expected_predicate, expected_args, expected_negated = _probe_expected_literal(probe)
    expected_holds = not expected_negated

    if isinstance(holds_raw, bool):
        holds = holds_raw
    elif isinstance(holds_raw, str):
        lowered = holds_raw.strip().lower()
        if lowered == "true":
            holds = True
        elif lowered == "false":
            holds = False
        else:
            holds = None
    else:
        holds = None

    fact_true = _evaluate_fact_pddl(probe.fact_pddl, check_fn)
    passed = (
        predicate == expected_predicate
        and args == expected_args
        and holds == expected_holds
        and fact_true
    )

    details = {
        "expected_response": dict(probe.expected_response),
        "parsed_response": {
            "predicate": predicate,
            "holds": holds,
            "args": list(args),
        },
        "fact_true": fact_true,
        "expected_negated": expected_negated,
    }
    return passed, details


def build_runtime_metadata(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build persisted runtime metadata for submit/migration flows."""
    projection = project_runtime_from_problem(task_data["problem_pddl"])
    return {
        "runtime_semantics_version": "literal_tom_probe_v1",
        "functional_goal_pddl": projection.functional_goal_pddl,
        "literal_tom_probes": [probe.to_dict() for probe in projection.probes],
        "epistemic_conjuncts_removed": projection.epistemic_conjuncts_removed,
        "runtime_projection_valid": projection.is_valid,
        "runtime_projection_errors": list(projection.invalid_reasons),
    }


def load_literal_tom_probes(raw_probes: Sequence[Dict[str, Any]]) -> List[LiteralToMProbe]:
    """Load persisted literal-ToM probes from task JSON."""
    probes: List[LiteralToMProbe] = []
    for raw in raw_probes or []:
        if not isinstance(raw, dict):
            continue
        probes.append(
            LiteralToMProbe(
                probe_id=str(raw.get("probe_id", "")),
                agent_id=str(raw.get("agent_id", "")),
                subject_agents=tuple(raw.get("subject_agents", []) or []),
                fact_pddl=str(raw.get("fact_pddl", "")),
                fact_natural_language=str(raw.get("fact_natural_language", "")),
                source_pddl=str(raw.get("source_pddl", "")),
                question=str(raw.get("question", "")),
                expected_response=dict(raw.get("expected_response", {})),
                depth=int(raw.get("depth", 0) or 0),
                owner=raw.get("owner"),
                supported=bool(raw.get("supported", True)),
                unsupported_reason=raw.get("unsupported_reason"),
            )
        )
    return probes


def _build_probe(formula: Knows, *, probe_idx: int, owner: Optional[str]) -> LiteralToMProbe:
    agents, fact_formula = _unwrap_nested_knowledge(formula)
    outer_agent = agents[0]
    subject_agents = tuple(agents[1:])
    fact_pddl = fact_formula.to_pddl()
    fact_nl = goal_to_natural_language(fact_formula)
    predicate, args, negated = _fact_formula_to_probe_fields(fact_formula)
    target_entities = ", ".join(args) if args else "(none)"
    if subject_agents:
        next_agent = subject_agents[0]
        question = (
            f"Probe {probe_idx}: predict what {next_agent} would report about "
            f'"{fact_nl}" for ordered entities [{target_entities}]. '
            "The expected argument order is fixed as listed. "
            'If unknown, return predicate "unknown" and holds null. '
            'JSON shape: {"probe_id":"k_probe_X","predicate":"<predicate_name>|unknown","holds":true|false|null,"args":["entity_or_target", ...]}'
        )
    else:
        question = (
            f'Probe {probe_idx}: report whether "{fact_nl}" holds for ordered entities '
            f"[{target_entities}]. "
            "The expected argument order is fixed as listed. "
            'If unknown, return predicate "unknown" and holds null. '
            'JSON shape: {"probe_id":"k_probe_X","predicate":"<predicate_name>|unknown","holds":true|false|null,"args":["entity_or_target", ...]}'
        )

    return LiteralToMProbe(
        probe_id=f"k_probe_{probe_idx}",
        agent_id=outer_agent,
        subject_agents=subject_agents,
        fact_pddl=fact_pddl,
        fact_natural_language=fact_nl,
        source_pddl=formula.to_pddl(),
        question=question,
        expected_response={
            "predicate": predicate,
            "holds": not negated,
            "args": list(args),
        },
        depth=len(agents),
        owner=owner,
    )


def _build_unsupported_probe(
    formula: Formula,
    *,
    probe_idx: int,
    owner: Optional[str],
    reason: str,
) -> LiteralToMProbe:
    return LiteralToMProbe(
        probe_id=f"k_probe_{probe_idx}",
        agent_id="",
        subject_agents=(),
        fact_pddl="",
        fact_natural_language="",
        source_pddl=formula.to_pddl(),
        question="",
        expected_response={},
        depth=0,
        owner=owner,
        supported=False,
        unsupported_reason=reason,
    )


def _fact_formula_to_probe_fields(formula: Formula) -> Tuple[str, Tuple[str, ...], bool]:
    if isinstance(formula, Literal):
        return formula.predicate, formula.args, formula.negated
    if isinstance(formula, Not) and isinstance(formula.operand, Literal):
        lit = formula.operand
        return lit.predicate, lit.args, not lit.negated
    parsed = parse_goal_string(formula.to_pddl())
    if isinstance(parsed, Literal):
        return parsed.predicate, parsed.args, parsed.negated
    if isinstance(parsed, Not) and isinstance(parsed.operand, Literal):
        lit = parsed.operand
        return lit.predicate, lit.args, not lit.negated
    raise ValueError(f"Unsupported probe fact formula: {formula.to_pddl()}")


def _probe_expected_literal(probe: LiteralToMProbe) -> Tuple[str, Tuple[str, ...], bool]:
    return _fact_formula_to_probe_fields(parse_goal_string(probe.fact_pddl))


def _unwrap_nested_knowledge(formula: Knows) -> Tuple[List[str], Formula]:
    agents = [formula.agent]
    inner: Formula = formula.inner
    while isinstance(inner, Knows):
        agents.append(inner.agent)
        inner = inner.inner
    return agents, inner


def _evaluate_fact_pddl(fact_pddl: str, check_fn) -> bool:
    parsed = parse_problem_pddl(
        f"(define (problem probe_eval) (:domain enacttom) (:init) (:goal {fact_pddl}))"
    ).goal_formula
    return parsed.evaluate(check_fn)
