"""
Python DSL for PDDL + Epistemic extensions (E-PDDL).

Provides dataclass-based representations of PDDL constructs that can be
serialized to PDDL text or stored as strings in task JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Type:
    """PDDL type in a type hierarchy."""
    name: str
    parent: Optional[str] = None

    def to_pddl(self) -> str:
        if self.parent:
            return f"{self.name} - {self.parent}"
        return self.name


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    """A typed parameter for predicates/actions."""
    name: str
    type: str

    def to_pddl(self) -> str:
        return f"?{self.name} - {self.type}"


@dataclass(frozen=True)
class Predicate:
    """PDDL predicate schema."""
    name: str
    params: Tuple[Param, ...] = ()

    def to_pddl(self) -> str:
        if not self.params:
            return f"({self.name})"
        params_str = " ".join(p.to_pddl() for p in self.params)
        return f"({self.name} {params_str})"


# ---------------------------------------------------------------------------
# Formulas (logical expressions)
# ---------------------------------------------------------------------------

class Formula:
    """Base class for all PDDL formulas."""

    def to_pddl(self) -> str:
        raise NotImplementedError

    def flatten(self) -> List["Formula"]:
        """Flatten to list of conjuncts (Literals or epistemic wrappers)."""
        raise NotImplementedError

    def evaluate(self, check_fn) -> bool:
        """Evaluate formula against a predicate checker function."""
        raise NotImplementedError


@dataclass(frozen=True)
class Literal(Formula):
    """A grounded predicate instance: (predicate_name arg1 arg2 ...)."""
    predicate: str
    args: Tuple[str, ...] = ()
    negated: bool = False

    def to_pddl(self) -> str:
        args_str = " ".join(self.args)
        inner = f"({self.predicate} {args_str})" if args_str else f"({self.predicate})"
        if self.negated:
            return f"(not {inner})"
        return inner

    def flatten(self) -> List["Formula"]:
        return [self]

    def evaluate(self, check_fn) -> bool:
        result = check_fn(self.predicate, self.args)
        return (not result) if self.negated else result

    def to_proposition(self) -> Dict[str, Any]:
        """Convert to evaluation.py proposition format."""
        prop: Dict[str, Any] = {
            "entity": self.args[0] if self.args else "",
            "property": self.predicate,
        }
        if len(self.args) > 1:
            prop["target"] = self.args[1]
        if len(self.args) > 2:
            prop["value"] = self.args[2]
        if self.negated:
            prop["value"] = False
        return prop

    @classmethod
    def from_proposition(cls, prop: Dict[str, Any]) -> "Literal":
        """Create from evaluation.py proposition format."""
        predicate = prop["property"]
        args = [prop["entity"]]
        target = prop.get("target")
        if target is not None:
            args.append(str(target))
        value = prop.get("value")
        negated = value is False
        if isinstance(value, (int, float)) and value is not False:
            args.append(str(value))
        return cls(predicate=predicate, args=tuple(args), negated=negated)


@dataclass(frozen=True)
class And(Formula):
    """Conjunction of formulas."""
    operands: Tuple[Formula, ...] = ()

    def to_pddl(self) -> str:
        if len(self.operands) == 1:
            return self.operands[0].to_pddl()
        inner = " ".join(f.to_pddl() for f in self.operands)
        return f"(and {inner})"

    def flatten(self) -> List["Formula"]:
        result = []
        for op in self.operands:
            result.extend(op.flatten())
        return result

    def evaluate(self, check_fn) -> bool:
        return all(op.evaluate(check_fn) for op in self.operands)


@dataclass(frozen=True)
class Or(Formula):
    """Disjunction of formulas."""
    operands: Tuple[Formula, ...] = ()

    def to_pddl(self) -> str:
        if len(self.operands) == 1:
            return self.operands[0].to_pddl()
        inner = " ".join(f.to_pddl() for f in self.operands)
        return f"(or {inner})"

    def flatten(self) -> List["Formula"]:
        # Or is a disjunction — cannot be decomposed into conjuncts.
        return [self]

    def evaluate(self, check_fn) -> bool:
        return any(op.evaluate(check_fn) for op in self.operands)


@dataclass(frozen=True)
class Not(Formula):
    """Negation of a formula."""
    operand: Formula = None  # type: ignore

    def to_pddl(self) -> str:
        return f"(not {self.operand.to_pddl()})"

    def flatten(self) -> List["Formula"]:
        if isinstance(self.operand, Literal):
            lit = self.operand
            return [Literal(predicate=lit.predicate, args=lit.args, negated=not lit.negated)]
        return [self]

    def evaluate(self, check_fn) -> bool:
        return not self.operand.evaluate(check_fn)


# ---------------------------------------------------------------------------
# Epistemic formulas
# ---------------------------------------------------------------------------

class EpistemicFormula(Formula):
    """Base class for epistemic (belief/knowledge) formulas."""
    pass


@dataclass(frozen=True)
class Knows(EpistemicFormula):
    """K(agent, formula) — agent knows formula is true."""
    agent: str = ""
    inner: Formula = None  # type: ignore

    def to_pddl(self) -> str:
        return f"(K {self.agent} {self.inner.to_pddl()})"

    def flatten(self) -> List["Formula"]:
        # Preserve the epistemic wrapper as a single conjunct
        return [self]

    def evaluate(self, check_fn) -> bool:
        # Epistemic evaluation requires a belief model, not just predicate checks.
        # For runtime goal checking, we treat K(a, phi) as phi being true
        # (conservative: if it's true in the world, the agent can know it).
        return self.inner.evaluate(check_fn)

    def get_inner_literals(self) -> List["Literal"]:
        """Extract leaf Literal nodes from inside the epistemic wrapper."""
        return self.inner.flatten()


@dataclass(frozen=True)
class Believes(EpistemicFormula):
    """B(agent, formula) — agent believes formula."""
    agent: str = ""
    inner: Formula = None  # type: ignore

    def to_pddl(self) -> str:
        return f"(B {self.agent} {self.inner.to_pddl()})"

    def flatten(self) -> List["Formula"]:
        # Preserve the epistemic wrapper as a single conjunct
        return [self]

    def get_inner_literals(self) -> List["Literal"]:
        """Extract leaf Literal nodes from inside the epistemic wrapper."""
        return self.inner.flatten()

    def evaluate(self, check_fn) -> bool:
        return self.inner.evaluate(check_fn)


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

@dataclass
class Effect:
    """An action effect, optionally conditional."""
    literal: Literal
    condition: Optional[Formula] = None  # conditional effect: (when condition literal)

    def to_pddl(self) -> str:
        if self.condition:
            return f"(when {self.condition.to_pddl()} {self.literal.to_pddl()})"
        return self.literal.to_pddl()


@dataclass
class ForallEffect:
    """(forall (?var - type) (when condition effect))"""
    variable: Param
    condition: Formula
    effect: Literal
    negative_effect: Optional[Literal] = None  # for (and pos (not neg)) patterns

    def to_pddl(self) -> str:
        if self.negative_effect:
            body = f"(and {self.effect.to_pddl()} (not {self.negative_effect.to_pddl()}))"
        else:
            body = self.effect.to_pddl()
        return (f"(forall (?{self.variable.name} - {self.variable.type}) "
                f"(when {self.condition.to_pddl()} {body}))")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """PDDL action with optional epistemic observability."""
    name: str
    params: List[Param] = field(default_factory=list)
    preconditions: Optional[Formula] = None
    effects: List[Union[Effect, "ForallEffect"]] = field(default_factory=list)
    # Observability: which agents can observe this action's effects
    # "full" = acting agent, "partial" = same-room agents, "none" = other agents
    observability: str = "full"

    def to_pddl(self) -> str:
        params_str = " ".join(p.to_pddl() for p in self.params)
        pre_str = self.preconditions.to_pddl() if self.preconditions else "()"
        lines = [
            f"(:action {self.name}",
            f"  :parameters ({params_str})",
            f"  :precondition {pre_str}",
        ]
        if self.effects:
            if len(self.effects) == 1:
                eff_str = self.effects[0].to_pddl()
            else:
                effs = " ".join(e.to_pddl() for e in self.effects)
                eff_str = f"(and {effs})"
            lines.append(f"  :effect {eff_str}")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Problem & Domain
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """A PDDL problem instance (per-task)."""
    name: str
    domain_name: str
    objects: Dict[str, str] = field(default_factory=dict)  # name -> type
    init: List[Literal] = field(default_factory=list)
    goal: Optional[Formula] = None
    epistemic_init: List[Union[Knows, Believes]] = field(default_factory=list)

    def to_pddl(self) -> str:
        obj_lines = []
        by_type: Dict[str, List[str]] = {}
        for name, typ in self.objects.items():
            by_type.setdefault(typ, []).append(name)
        for typ, names in by_type.items():
            obj_lines.append(f"    {' '.join(names)} - {typ}")
        objects_str = "\n".join(obj_lines)

        init_str = "\n    ".join(l.to_pddl() for l in self.init)
        if self.epistemic_init:
            init_str += "\n    " + "\n    ".join(e.to_pddl() for e in self.epistemic_init)

        goal_str = self.goal.to_pddl() if self.goal else "()"

        return (
            f"(define (problem {self.name})\n"
            f"  (:domain {self.domain_name})\n"
            f"  (:objects\n{objects_str}\n  )\n"
            f"  (:init\n    {init_str}\n  )\n"
            f"  (:goal {goal_str})\n"
            f")"
        )


@dataclass
class Domain:
    """A PDDL domain (shared across tasks)."""
    name: str
    types: List[Type] = field(default_factory=list)
    predicates: List[Predicate] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)

    def to_pddl(self) -> str:
        types_str = " ".join(t.to_pddl() for t in self.types)
        preds_str = "\n    ".join(p.to_pddl() for p in self.predicates)
        actions_str = "\n\n".join(a.to_pddl() for a in self.actions)
        return (
            f"(define (domain {self.name})\n"
            f"  (:requirements :strips :typing :epistemic)\n"
            f"  (:types {types_str})\n"
            f"  (:predicates\n    {preds_str}\n  )\n\n"
            f"{actions_str}\n"
            f")"
        )

    def to_planning_pddl(self) -> str:
        """Output PDDL with :strips :typing :conditional-effects (no :epistemic).

        Fast Downward and other classical planners don't understand epistemic
        extensions. This method produces standard PDDL suitable for them.

        Types are grouped by parent to avoid ambiguous PDDL (e.g. the raw
        sequence ``agent object furniture - object`` would make ``object``
        a child of itself).
        """
        # Group types by parent for unambiguous PDDL.
        # Skip user type "object" — it conflicts with PDDL's built-in root
        # type of the same name and confuses unified-planning's type checker.
        by_parent: Dict[Optional[str], List[str]] = {}
        for t in self.types:
            if t.name == "object" and t.parent is None:
                continue  # implicit root type
            by_parent.setdefault(t.parent, []).append(t.name)
        types_parts: List[str] = []
        if None in by_parent:
            types_parts.extend(by_parent[None])
        for parent, names in by_parent.items():
            if parent is not None:
                types_parts.append(f"{' '.join(names)} - {parent}")
        types_str = " ".join(types_parts)

        preds_str = "\n    ".join(p.to_pddl() for p in self.predicates)
        # Skip actions with no physical effects (e.g. communicate, wait)
        # — classical planners can't use them and FD rejects empty :effect
        planning_actions = [a for a in self.actions if a.effects]
        actions_str = "\n\n".join(a.to_pddl() for a in planning_actions)
        return (
            f"; Generated from domain.py -- do not edit manually\n"
            f"(define (domain {self.name})\n"
            f"  (:requirements :strips :typing :conditional-effects)\n"
            f"  (:types {types_str})\n"
            f"  (:predicates\n    {preds_str}\n  )\n\n"
            f"{actions_str}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_goal_string(goal_str: str) -> Formula:
    """
    Parse a PDDL goal string into a Formula tree.

    Supports: (and ...), (or ...), (not ...), (K agent ...), (B agent ...),
              (predicate arg1 arg2 ...)

    Examples:
        "(is_open cabinet_27)" -> Literal("is_open", ("cabinet_27",))
        "(and (is_open cabinet_27) (is_on_top bottle_4 table_13))"
            -> And(Literal(...), Literal(...))
        "(K agent_0 (is_open cabinet_27))" -> Knows("agent_0", Literal(...))
        "(K agent_0 (K agent_1 (is_open cabinet_27)))" -> nested Knows (depth 2)
    """
    goal_str = goal_str.strip()
    if not goal_str:
        raise ValueError("Empty goal string")

    # Check balanced parentheses
    depth = 0
    for c in goal_str:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth < 0:
            raise ValueError("Unbalanced parentheses: extra ')'")
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses: {depth} unclosed '('")

    tokens = _tokenize(goal_str)
    result, pos = _parse_tokens(tokens, 0)
    if pos < len(tokens):
        raise ValueError(f"Unexpected tokens after position {pos}: {tokens[pos:]}")
    return result


def _tokenize(s: str) -> List[str]:
    """Tokenize a PDDL s-expression into a list of tokens."""
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in ('(', ')'):
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in ('(', ')') and not s[j].isspace():
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _parse_tokens(tokens: List[str], pos: int) -> Tuple[Formula, int]:
    """Parse tokens starting at pos, return (Formula, next_pos)."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of tokens")

    if tokens[pos] == '(':
        pos += 1  # skip '('
        if pos >= len(tokens):
            raise ValueError("Unexpected end after '('")

        head = tokens[pos].lower()
        pos += 1

        if head == 'and':
            operands = []
            while pos < len(tokens) and tokens[pos] != ')':
                child, pos = _parse_tokens(tokens, pos)
                operands.append(child)
            pos += 1  # skip ')'
            return And(operands=tuple(operands)), pos

        elif head == 'or':
            operands = []
            while pos < len(tokens) and tokens[pos] != ')':
                child, pos = _parse_tokens(tokens, pos)
                operands.append(child)
            pos += 1
            return Or(operands=tuple(operands)), pos

        elif head == 'not':
            child, pos = _parse_tokens(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return Not(operand=child), pos

        elif head == 'k':
            if pos >= len(tokens) or tokens[pos] == ')':
                raise ValueError("K() requires an agent and inner formula")
            agent_name = tokens[pos]
            pos += 1
            if pos >= len(tokens) or tokens[pos] == ')':
                raise ValueError("K() requires an inner formula after agent name")
            inner, pos = _parse_tokens(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return Knows(agent=agent_name, inner=inner), pos

        elif head == 'b':
            if pos >= len(tokens) or tokens[pos] == ')':
                raise ValueError("B() requires an agent and inner formula")
            agent_name = tokens[pos]
            pos += 1
            if pos >= len(tokens) or tokens[pos] == ')':
                raise ValueError("B() requires an inner formula after agent name")
            inner, pos = _parse_tokens(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return Believes(agent=agent_name, inner=inner), pos

        else:
            # It's a predicate literal: (pred_name arg1 arg2 ...)
            args = []
            while pos < len(tokens) and tokens[pos] != ')':
                args.append(tokens[pos])
                pos += 1
            pos += 1  # skip ')'
            return Literal(predicate=head, args=tuple(args)), pos

    else:
        # Bare atom (shouldn't happen in well-formed PDDL but handle gracefully)
        return Literal(predicate=tokens[pos], args=()), pos + 1


def validate_goal_predicates(goal: Formula, domain: Domain) -> List[str]:
    """
    Validate that all predicates in a goal formula exist in the domain
    and have correct arity.

    Args:
        goal: Parsed Formula tree
        domain: Domain with predicate definitions

    Returns:
        List of error strings (empty = valid)
    """
    pred_arities = {p.name: len(p.params) for p in domain.predicates}
    errors: List[str] = []

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            if node.predicate not in pred_arities:
                errors.append(
                    f"Unknown predicate '{node.predicate}'. "
                    f"Available: {sorted(pred_arities.keys())}"
                )
            else:
                expected = pred_arities[node.predicate]
                actual = len(node.args)
                if actual != expected:
                    errors.append(
                        f"Predicate '{node.predicate}' expects {expected} "
                        f"argument(s) but got {actual}: {node.to_pddl()}"
                    )
        elif isinstance(node, EpistemicFormula):
            _walk(node.inner)
        elif isinstance(node, (And, Or)):
            for op in node.operands:
                _walk(op)
        elif isinstance(node, Not):
            if node.operand is not None:
                _walk(node.operand)

    _walk(goal)
    return errors


def collect_leaf_literals(formula: Formula) -> List[Literal]:
    """Recursively collect all Literal nodes from a formula tree.

    Unlike ``flatten()``, this traverses through Or, Not, And, and epistemic
    wrappers to return every referenced Literal regardless of logical structure.
    Useful for callers that need to enumerate referenced predicates/objects
    without caring about the evaluation semantics.
    """
    results: List[Literal] = []

    def _walk(node: Formula) -> None:
        if isinstance(node, Literal):
            results.append(node)
        elif isinstance(node, EpistemicFormula):
            _walk(node.inner)
        elif isinstance(node, (And, Or)):
            for op in node.operands:
                _walk(op)
        elif isinstance(node, Not) and node.operand is not None:
            _walk(node.operand)

    _walk(formula)
    return results


def goal_to_string(goal: Formula) -> str:
    """Serialize a Formula to its PDDL string representation."""
    return goal.to_pddl()
