"""
PDDL + epistemic extensions for EnactToM.

Replaces the hand-crafted subtask DAG with formal PDDL goal specifications.
ToM level and human-readable descriptions are derived from the PDDL, not stored redundantly.
"""

from enacttom.pddl.dsl import (
    Type,
    Predicate,
    Param,
    Formula,
    Literal,
    And,
    Or,
    Not,
    EpistemicFormula,
    Knows,
    Believes,
    Effect,
    Action,
    Problem,
    Domain,
)
from enacttom.pddl.epistemic import ObservabilityModel
from enacttom.pddl.compiler import compile_task
from enacttom.pddl.goal_checker import PDDLGoalChecker
from enacttom.pddl.describe import describe_task
from enacttom.pddl.goal_spec import GoalEntry, GoalSpec
from enacttom.pddl.problem_pddl import (
    ParsedProblemPDDL,
    parse_problem_pddl,
    extract_goal_from_problem_pddl,
    replace_goal_in_problem_pddl,
)
from enacttom.pddl.runtime_projection import LiteralToMProbe, RuntimeProjection

__all__ = [
    "Type",
    "Predicate",
    "Param",
    "Formula",
    "Literal",
    "And",
    "Or",
    "Not",
    "EpistemicFormula",
    "Knows",
    "Believes",
    "Effect",
    "Action",
    "Problem",
    "Domain",
    "ObservabilityModel",
    "compile_task",
    "PDDLGoalChecker",
    "describe_task",
    "GoalEntry",
    "GoalSpec",
    "ParsedProblemPDDL",
    "parse_problem_pddl",
    "extract_goal_from_problem_pddl",
    "replace_goal_in_problem_pddl",
    "LiteralToMProbe",
    "RuntimeProjection",
]
