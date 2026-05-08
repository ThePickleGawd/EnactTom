"""Tests for enacttom.pddl.goal_spec — GoalEntry and GoalSpec."""

import pytest

from enacttom.pddl.goal_spec import GoalEntry, GoalSpec
from enacttom.pddl.domain import ENACTTOM_DOMAIN


# -----------------------------------------------------------------------
# Construction helpers
# -----------------------------------------------------------------------

def _simple_goals_array():
    """Two physical goals with ordering."""
    return [
        {"id": 0, "pddl": "(is_open cabinet_27)", "after": []},
        {"id": 1, "pddl": "(is_on_top bottle_4 table_13)", "after": [0]},
    ]


def _epistemic_goals_array():
    """Goals with K() and ownership."""
    return [
        {"id": 0, "pddl": "(K agent_0 (is_on_top laptop_0 table_29))", "after": []},
        {"id": 1, "pddl": "(is_on_top spoon_2 couch_15)", "after": [0]},
        {"id": 2, "pddl": "(is_open chest_of_drawers_40)", "after": [0], "owner": "agent_0"},
    ]


def _legacy_gold_task():
    """Legacy 3-field format from gold_k1_cross_room_report."""
    pddl_goal = "(and (K agent_0 (is_on_top laptop_0 table_29)) (K agent_1 (is_on_top cup_3 toilet_33)) (is_on_top spoon_2 couch_15) (is_open chest_of_drawers_40))"
    ordering = [
        {"before": "(K agent_0 (is_on_top laptop_0 table_29))", "after": "(is_on_top spoon_2 couch_15)"},
        {"before": "(K agent_1 (is_on_top cup_3 toilet_33))", "after": "(is_open chest_of_drawers_40)"},
    ]
    owners = {}
    return pddl_goal, ordering, owners


# -----------------------------------------------------------------------
# from_goals_array
# -----------------------------------------------------------------------

class TestFromGoalsArray:
    def test_simple(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        assert len(spec) == 2
        assert spec.entries[0].id == 0
        assert spec.entries[0].pddl == "(is_open cabinet_27)"
        assert spec.entries[0].after == ()
        assert spec.entries[1].after == (0,)

    def test_epistemic_with_owner(self):
        spec = GoalSpec.from_goals_array(_epistemic_goals_array())
        assert len(spec) == 3
        assert spec.entries[2].owner == "agent_0"
        assert spec.entries[2].after == (0,)

    def test_duplicate_ids_raise(self):
        goals = [
            {"id": 0, "pddl": "(is_open cabinet_27)", "after": []},
            {"id": 0, "pddl": "(is_open cabinet_28)", "after": []},
        ]
        with pytest.raises(ValueError, match="Duplicate goal ID"):
            GoalSpec.from_goals_array(goals)

    def test_invalid_after_reference_raises(self):
        goals = [
            {"id": 0, "pddl": "(is_open cabinet_27)", "after": [99]},
        ]
        with pytest.raises(ValueError, match="non-existent prerequisite"):
            GoalSpec.from_goals_array(goals)

    def test_cycle_raises(self):
        goals = [
            {"id": 0, "pddl": "(is_open cabinet_27)", "after": [1]},
            {"id": 1, "pddl": "(is_open cabinet_28)", "after": [0]},
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            GoalSpec.from_goals_array(goals)

    def test_self_cycle_raises(self):
        goals = [
            {"id": 0, "pddl": "(is_open cabinet_27)", "after": [0]},
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            GoalSpec.from_goals_array(goals)

    def test_empty_goals(self):
        spec = GoalSpec.from_goals_array([])
        assert len(spec) == 0


# -----------------------------------------------------------------------
# from_legacy
# -----------------------------------------------------------------------

class TestFromLegacy:
    def test_gold_task_roundtrip(self):
        pddl_goal, ordering, owners = _legacy_gold_task()
        spec = GoalSpec.from_legacy(pddl_goal, ordering, owners)
        assert len(spec) == 4
        # Should have K() goals
        assert spec.has_epistemic_goals

    def test_ordering_preserved(self):
        pddl_goal, ordering, owners = _legacy_gold_task()
        spec = GoalSpec.from_legacy(pddl_goal, ordering, owners)
        # Entry for "(is_on_top spoon_2 couch_15)" should depend on
        # entry for "(K agent_0 (is_on_top laptop_0 table_29))"
        spoon_entry = None
        k_agent0_entry = None
        for e in spec.entries:
            if "spoon_2" in e.pddl:
                spoon_entry = e
            if "K agent_0" in e.pddl:
                k_agent0_entry = e
        assert spoon_entry is not None
        assert k_agent0_entry is not None
        assert k_agent0_entry.id in spoon_entry.after

    def test_simple_no_ordering(self):
        spec = GoalSpec.from_legacy("(is_open cabinet_27)", [], {})
        assert len(spec) == 1
        assert spec.entries[0].after == ()

    def test_owners_mapped(self):
        spec = GoalSpec.from_legacy(
            "(and (is_open cabinet_27) (is_on_top bottle_4 table_13))",
            [],
            {"(is_open cabinet_27)": "agent_0"},
        )
        assert spec.entries[0].owner == "agent_0"
        assert spec.entries[1].owner is None


# -----------------------------------------------------------------------
# Serialization round-trip
# -----------------------------------------------------------------------

class TestRoundTrip:
    def test_goals_array_roundtrip(self):
        original = _epistemic_goals_array()
        spec = GoalSpec.from_goals_array(original)
        serialized = spec.to_goals_array()
        spec2 = GoalSpec.from_goals_array(serialized)
        assert spec == spec2

    def test_legacy_to_goals_to_spec(self):
        pddl_goal, ordering, owners = _legacy_gold_task()
        spec = GoalSpec.from_legacy(pddl_goal, ordering, owners)
        array = spec.to_goals_array()
        spec2 = GoalSpec.from_goals_array(array)
        assert spec == spec2


# -----------------------------------------------------------------------
# to_formula / to_pddl_string
# -----------------------------------------------------------------------

class TestFormula:
    def test_single_goal_formula(self):
        spec = GoalSpec.from_goals_array([
            {"id": 0, "pddl": "(is_open cabinet_27)", "after": []},
        ])
        pddl = spec.to_pddl_string()
        assert "is_open" in pddl
        assert "cabinet_27" in pddl

    def test_multi_goal_and_formula(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        pddl = spec.to_pddl_string()
        assert pddl.startswith("(and ")
        assert "is_open" in pddl
        assert "is_on_top" in pddl

    def test_epistemic_formula_preserves_k(self):
        spec = GoalSpec.from_goals_array([
            {"id": 0, "pddl": "(K agent_0 (is_on_top laptop_0 table_29))", "after": []},
        ])
        pddl = spec.to_pddl_string()
        assert "(K agent_0" in pddl


# -----------------------------------------------------------------------
# validate
# -----------------------------------------------------------------------

class TestValidate:
    def test_valid_spec(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        errors = spec.validate(ENACTTOM_DOMAIN, {"agent_0", "agent_1"})
        assert errors == []

    def test_invalid_predicate(self):
        spec = GoalSpec.from_goals_array([
            {"id": 0, "pddl": "(is_flying spaceship_1)", "after": []},
        ])
        errors = spec.validate(ENACTTOM_DOMAIN, {"agent_0"})
        assert len(errors) > 0
        assert any("is_flying" in e for e in errors)

    def test_wrong_arity(self):
        spec = GoalSpec.from_goals_array([
            {"id": 0, "pddl": "(is_open cabinet_27 extra_arg)", "after": []},
        ])
        errors = spec.validate(ENACTTOM_DOMAIN, {"agent_0"})
        assert len(errors) > 0

    def test_unknown_epistemic_agent(self):
        spec = GoalSpec.from_goals_array([
            {"id": 0, "pddl": "(K agent_99 (is_open cabinet_27))", "after": []},
        ])
        errors = spec.validate(ENACTTOM_DOMAIN, {"agent_0", "agent_1"})
        assert any("agent_99" in e for e in errors)

    def test_valid_epistemic_spec(self):
        spec = GoalSpec.from_goals_array(_epistemic_goals_array())
        errors = spec.validate(ENACTTOM_DOMAIN, {"agent_0", "agent_1"})
        assert errors == []

    def test_init_only_predicate_rejected_in_goal(self):
        spec = GoalSpec.from_goals_array([
            {"id": 0, "pddl": "(K agent_0 (is_inverse cabinet_27))", "after": []},
        ])
        errors = spec.validate(ENACTTOM_DOMAIN, {"agent_0"})
        assert any("init-only" in e for e in errors)


# -----------------------------------------------------------------------
# Queries
# -----------------------------------------------------------------------

class TestQueries:
    def test_get_prerequisites(self):
        spec = GoalSpec.from_goals_array(_epistemic_goals_array())
        assert spec.get_prerequisites(0) == set()
        assert spec.get_prerequisites(1) == {0}
        assert spec.get_prerequisites(2) == {0}

    def test_get_prerequisites_missing_raises(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        with pytest.raises(KeyError):
            spec.get_prerequisites(99)

    def test_get_entries_by_owner(self):
        spec = GoalSpec.from_goals_array(_epistemic_goals_array())
        agent0 = spec.get_entries_by_owner("agent_0")
        assert len(agent0) == 1
        assert agent0[0].id == 2

    def test_get_required_entries(self):
        spec = GoalSpec.from_goals_array(_epistemic_goals_array())
        required = spec.get_required_entries()
        # Goals 0 and 1 have no owner
        assert len(required) == 2

    def test_has_epistemic_goals_true(self):
        spec = GoalSpec.from_goals_array(_epistemic_goals_array())
        assert spec.has_epistemic_goals is True

    def test_has_epistemic_goals_false(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        assert spec.has_epistemic_goals is False


# -----------------------------------------------------------------------
# Cycle detection
# -----------------------------------------------------------------------

class TestCycleDetection:
    def test_no_cycle(self):
        entries = (
            GoalEntry(id=0, pddl="x", after=()),
            GoalEntry(id=1, pddl="y", after=(0,)),
            GoalEntry(id=2, pddl="z", after=(0, 1)),
        )
        assert GoalSpec._detect_cycle(entries) is False

    def test_direct_cycle(self):
        entries = (
            GoalEntry(id=0, pddl="x", after=(1,)),
            GoalEntry(id=1, pddl="y", after=(0,)),
        )
        assert GoalSpec._detect_cycle(entries) is True

    def test_self_cycle(self):
        entries = (
            GoalEntry(id=0, pddl="x", after=(0,)),
        )
        assert GoalSpec._detect_cycle(entries) is True

    def test_three_node_cycle(self):
        entries = (
            GoalEntry(id=0, pddl="x", after=(2,)),
            GoalEntry(id=1, pddl="y", after=(0,)),
            GoalEntry(id=2, pddl="z", after=(1,)),
        )
        assert GoalSpec._detect_cycle(entries) is True

    def test_diamond_no_cycle(self):
        entries = (
            GoalEntry(id=0, pddl="x", after=()),
            GoalEntry(id=1, pddl="y", after=(0,)),
            GoalEntry(id=2, pddl="z", after=(0,)),
            GoalEntry(id=3, pddl="w", after=(1, 2)),
        )
        assert GoalSpec._detect_cycle(entries) is False


# -----------------------------------------------------------------------
# Dunder methods
# -----------------------------------------------------------------------

class TestDunder:
    def test_len(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        assert len(spec) == 2

    def test_iter(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        entries = list(spec)
        assert len(entries) == 2
        assert entries[0].id == 0

    def test_eq(self):
        spec1 = GoalSpec.from_goals_array(_simple_goals_array())
        spec2 = GoalSpec.from_goals_array(_simple_goals_array())
        assert spec1 == spec2

    def test_not_eq_different_goals(self):
        spec1 = GoalSpec.from_goals_array(_simple_goals_array())
        spec2 = GoalSpec.from_goals_array(_epistemic_goals_array())
        assert spec1 != spec2

    def test_repr(self):
        spec = GoalSpec.from_goals_array(_simple_goals_array())
        r = repr(spec)
        assert "GoalSpec" in r
