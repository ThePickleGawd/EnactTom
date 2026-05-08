"""Tests for epistemic compilation into classical PDDL."""

import pytest

from enacttom.pddl.dsl import (
    And,
    Believes,
    Knows,
    Literal,
    Not,
    Or,
    Problem,
    parse_goal_string,
)
from enacttom.pddl.domain import ENACTTOM_DOMAIN
from enacttom.pddl.epistemic import ObservabilityModel
from enacttom.pddl.epistemic_compiler import (
    EpistemicCompilation,
    KGoalNode,
    compile_epistemic,
    _collect_k_goals,
    _collect_leaf_facts,
    _fact_id,
    _leaf_fact_hash,
    _is_formula_observable_by,
    _replace_epistemic_in_goal,
    _build_observe_actions_network,
    _build_inform_actions_network,
    _build_budget_tokens,
    _build_knowledge_predicates_network,
    _build_nested_k_actions,
)
from enacttom.pddl.fd_solver import FastDownwardSolver, HAS_UP, _has_epistemic_goals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem(goal, objects=None, init=None):
    return Problem(
        name="test",
        domain_name="enacttom",
        objects=objects or {
            "agent_0": "agent",
            "agent_1": "agent",
            "cabinet_27": "furniture",
            "kitchen_1": "room",
            "bedroom_1": "room",
        },
        init=init or [
            Literal("is_closed", ("cabinet_27",)),
            Literal("can_communicate", ("agent_0", "agent_1")),
            Literal("can_communicate", ("agent_1", "agent_0")),
            Literal("agent_in_room", ("agent_0", "kitchen_1")),
            Literal("agent_in_room", ("agent_1", "bedroom_1")),
            Literal("is_in_room", ("cabinet_27", "kitchen_1")),
        ],
        goal=goal,
    )


def _make_obs(restricted=None, object_rooms=None, message_limits=None,
              message_targets=None):
    return ObservabilityModel(
        restricted_rooms=restricted or {},
        object_rooms=object_rooms or {},
        message_limits=message_limits or {},
        message_targets=message_targets or {},
    )


# ---------------------------------------------------------------------------
# _has_epistemic_goals
# ---------------------------------------------------------------------------

class TestHasEpistemicGoals:
    def test_literal_no_epistemic(self):
        from enacttom.pddl.fd_solver import _has_epistemic_goals
        assert not _has_epistemic_goals(Literal("is_open", ("cabinet_27",)))

    def test_knows_has_epistemic(self):
        from enacttom.pddl.fd_solver import _has_epistemic_goals
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        assert _has_epistemic_goals(goal)

    def test_and_with_knows(self):
        from enacttom.pddl.fd_solver import _has_epistemic_goals
        goal = And((
            Literal("is_open", ("cabinet_27",)),
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
        ))
        assert _has_epistemic_goals(goal)

    def test_or_with_knows(self):
        from enacttom.pddl.fd_solver import _has_epistemic_goals
        goal = Or((
            Literal("is_open", ("cabinet_27",)),
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
        ))
        assert _has_epistemic_goals(goal)

    def test_not_with_knows(self):
        from enacttom.pddl.fd_solver import _has_epistemic_goals
        goal = Not(operand=Knows("agent_0", Literal("is_open", ("cabinet_27",))))
        assert _has_epistemic_goals(goal)


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------

class TestFactId:
    def test_deterministic(self):
        """Same input → same fact_id."""
        lit = Literal("is_open", ("cabinet_27",))
        id1 = _leaf_fact_hash(lit)
        id2 = _leaf_fact_hash(lit)
        assert id1 == id2

    def test_different_formulas_different_ids(self):
        lit_a = Literal("is_open", ("cabinet_27",))
        lit_b = Literal("is_open", ("drawer_5",))
        id_a = _leaf_fact_hash(lit_a)
        id_b = _leaf_fact_hash(lit_b)
        assert id_a != id_b

    def test_same_fact_same_hash_across_agents(self):
        """Leaf hash is agent-independent."""
        lit = Literal("is_open", ("cabinet_27",))
        h = _leaf_fact_hash(lit)
        # K(agent_0, lit) and K(agent_1, lit) should share the same leaf hash
        assert len(h) == 8

    def test_is_8_hex_chars(self):
        lit = Literal("is_open", ("cabinet_27",))
        fid = _leaf_fact_hash(lit)
        assert len(fid) == 8
        assert all(c in "0123456789abcdef" for c in fid)

    def test_nested_k_uses_agent_specific_id(self):
        """Nested K goals use agent-specific fact_id."""
        inner = Literal("is_open", ("cabinet_27",))
        k_inner = Knows("agent_1", inner)
        id_a0 = _fact_id("agent_0", k_inner)
        id_a2 = _fact_id("agent_2", k_inner)
        assert id_a0 != id_a2


# ---------------------------------------------------------------------------
# K goal collection
# ---------------------------------------------------------------------------

class TestCollectKGoals:
    def test_simple_k(self):
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        obs = _make_obs()
        nodes = _collect_k_goals(goal, obs)
        assert len(nodes) == 1
        assert nodes[0].agent == "agent_0"
        assert nodes[0].depth == 1

    def test_nested_k(self):
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        obs = _make_obs()
        nodes = _collect_k_goals(outer, obs)
        assert len(nodes) == 2
        # Outer K(a0, K(a1, phi)) has nesting depth=2, inner K(a1, phi) has depth=1
        depths = {n.depth for n in nodes}
        assert depths == {1, 2}

    def test_and_with_multiple_k(self):
        goal = And((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Knows("agent_1", Literal("is_open", ("cabinet_27",))),
        ))
        obs = _make_obs()
        nodes = _collect_k_goals(goal, obs)
        assert len(nodes) == 2

    def test_no_k_returns_empty(self):
        goal = Literal("is_open", ("cabinet_27",))
        obs = _make_obs()
        nodes = _collect_k_goals(goal, obs)
        assert len(nodes) == 0

    def test_trivial_k_detected(self):
        """Agent with no restrictions → trivial."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        obs = _make_obs(
            restricted={"agent_1": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        nodes = _collect_k_goals(goal, obs)
        assert len(nodes) == 1
        assert nodes[0].trivial is True

    def test_non_trivial_k_detected(self):
        """Agent restricted from the room → non-trivial."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        nodes = _collect_k_goals(goal, obs)
        assert len(nodes) == 1
        assert nodes[0].trivial is False

    def test_believes_collected(self):
        goal = Believes("agent_0", Literal("is_open", ("cabinet_27",)))
        obs = _make_obs()
        nodes = _collect_k_goals(goal, obs)
        assert len(nodes) == 1


# ---------------------------------------------------------------------------
# Formula observability
# ---------------------------------------------------------------------------

class TestFormulaObservable:
    def test_unrestricted_agent(self):
        lit = Literal("is_open", ("cabinet_27",))
        obs = _make_obs(object_rooms={"cabinet_27": "kitchen_1"})
        assert _is_formula_observable_by("agent_0", lit, obs)

    def test_restricted_agent_cannot_observe(self):
        lit = Literal("is_open", ("cabinet_27",))
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        assert not _is_formula_observable_by("agent_0", lit, obs)

    def test_nested_k_not_directly_observable(self):
        """Nested K is never directly observable."""
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        obs = _make_obs()
        assert not _is_formula_observable_by("agent_0", inner, obs)

    def test_conjunction_all_observable(self):
        goal = And((
            Literal("is_open", ("cabinet_27",)),
            Literal("is_closed", ("cabinet_27",)),
        ))
        obs = _make_obs(object_rooms={"cabinet_27": "kitchen_1"})
        assert _is_formula_observable_by("agent_0", goal, obs)

    def test_conjunction_partially_unobservable(self):
        goal = And((
            Literal("is_open", ("cabinet_27",)),
            Literal("is_open", ("drawer_5",)),
        ))
        obs = _make_obs(
            restricted={"agent_0": {"bedroom_1"}},
            object_rooms={"cabinet_27": "kitchen_1", "drawer_5": "bedroom_1"},
        )
        assert not _is_formula_observable_by("agent_0", goal, obs)


# ---------------------------------------------------------------------------
# Knowledge predicates
# ---------------------------------------------------------------------------

class TestKnowledgePredicates:
    def test_builds_predicates_for_all_agents(self):
        """Network: one leaf fact → predicates for all agents."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        obs = _make_obs()
        nodes = _collect_k_goals(goal, obs)
        leaf_facts = _collect_leaf_facts(nodes)
        all_agents = ["agent_0", "agent_1"]
        preds = _build_knowledge_predicates_network(nodes, leaf_facts, all_agents)
        # 1 leaf fact × 2 agents = 2 predicates
        assert len(preds) >= 2
        assert any("knows_agent_0_" in p for p in preds)
        assert any("knows_agent_1_" in p for p in preds)

    def test_nested_k_has_outer_predicate(self):
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        obs = _make_obs()
        nodes = _collect_k_goals(outer, obs)
        leaf_facts = _collect_leaf_facts(nodes)
        all_agents = ["agent_0", "agent_1"]
        preds = _build_knowledge_predicates_network(nodes, leaf_facts, all_agents)
        # Should have leaf-fact predicates for both agents + outer K predicate
        assert len(preds) >= 3


# ---------------------------------------------------------------------------
# Observe actions
# ---------------------------------------------------------------------------

class TestObserveActions:
    def test_observe_for_agents_who_can_see(self):
        """Network: observe actions for all agents who can see the fact."""
        obs = _make_obs(
            restricted={"agent_1": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        leaf_facts = {"abc12345": Literal("is_open", ("cabinet_27",))}
        all_agents = ["agent_0", "agent_1"]
        actions = _build_observe_actions_network(leaf_facts, all_agents, obs)
        # agent_0 can see kitchen_1, agent_1 cannot
        assert len(actions) == 1
        assert "observe_knows_agent_0" in actions[0]
        assert "(is_open cabinet_27)" in actions[0]

    def test_both_restricted_no_observe(self):
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}, "agent_1": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        leaf_facts = {"abc12345": Literal("is_open", ("cabinet_27",))}
        all_agents = ["agent_0", "agent_1"]
        actions = _build_observe_actions_network(leaf_facts, all_agents, obs)
        assert len(actions) == 0


# ---------------------------------------------------------------------------
# Inform actions
# ---------------------------------------------------------------------------

class TestInformActions:
    def test_inform_for_all_comm_pairs(self):
        """Network: inform actions for all (sender, receiver) with can_communicate."""
        obs = _make_obs()
        leaf_facts = {"abc12345": Literal("is_open", ("cabinet_27",))}
        all_agents = ["agent_0", "agent_1"]
        can_comm = {("agent_0", "agent_1"), ("agent_1", "agent_0")}
        actions = _build_inform_actions_network(leaf_facts, [], all_agents, can_comm, obs)
        # 2 directions × 1 fact = 2 inform actions
        assert len(actions) == 2
        assert any("from_agent_0" in a for a in actions)
        assert any("from_agent_1" in a for a in actions)

    def test_inform_with_budget_tokens(self):
        """Budget-limited sender → token variants."""
        obs = _make_obs(message_limits={"agent_1": 3})
        leaf_facts = {"abc12345": Literal("is_open", ("cabinet_27",))}
        all_agents = ["agent_0", "agent_1"]
        can_comm = {("agent_1", "agent_0")}
        actions = _build_inform_actions_network(leaf_facts, [], all_agents, can_comm, obs)
        assert len(actions) == 3  # 3 token variants
        assert any("tok1" in a for a in actions)
        assert any("tok2" in a for a in actions)
        assert any("tok3" in a for a in actions)

    def test_no_comm_path_no_inform(self):
        """No can_communicate → no inform actions."""
        obs = _make_obs()
        leaf_facts = {"abc12345": Literal("is_open", ("cabinet_27",))}
        all_agents = ["agent_0", "agent_1"]
        can_comm: set = set()  # empty!
        actions = _build_inform_actions_network(leaf_facts, [], all_agents, can_comm, obs)
        assert len(actions) == 0

    def test_sender_knows_receiver_knows_effect(self):
        """Direct inform should also establish sender knowledge of receiver knowledge."""
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        obs = _make_obs()
        nodes = _collect_k_goals(outer, obs)
        leaf_facts = _collect_leaf_facts(nodes)
        all_agents = ["agent_0", "agent_1"]
        can_comm = {("agent_0", "agent_1")}
        actions = _build_inform_actions_network(leaf_facts, nodes, all_agents, can_comm, obs)

        outer_node = next(n for n in nodes if n.agent == "agent_0" and n.depth == 2)
        expected_pred = f"knows_agent_0_{outer_node.fact_id}"
        assert any(expected_pred in action for action in actions)


# ---------------------------------------------------------------------------
# Budget tokens
# ---------------------------------------------------------------------------

class TestBudgetTokens:
    def test_builds_correct_count(self):
        obs = _make_obs(message_limits={"agent_0": 2, "agent_1": 3})
        tokens = _build_budget_tokens(obs)
        assert len(tokens) == 5  # 2 + 3

    def test_no_limits_empty(self):
        obs = _make_obs()
        tokens = _build_budget_tokens(obs)
        assert len(tokens) == 0

    def test_none_limit_skipped(self):
        obs = _make_obs(message_limits={"agent_0": None, "agent_1": 2})
        tokens = _build_budget_tokens(obs)
        assert len(tokens) == 2


# ---------------------------------------------------------------------------
# Goal replacement
# ---------------------------------------------------------------------------

class TestGoalReplacement:
    def test_k_replaced_with_and_physical_knowledge(self):
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        nodes = _collect_k_goals(goal, obs)
        k_map = {f"{n.agent}:{n.inner.to_pddl()}": n for n in nodes}
        result = _replace_epistemic_in_goal(goal, k_map)
        pddl = result.to_pddl()
        assert "(is_open cabinet_27)" in pddl
        assert "knows_agent_0_" in pddl

    def test_literal_unchanged(self):
        goal = Literal("is_open", ("cabinet_27",))
        result = _replace_epistemic_in_goal(goal, {})
        assert result is goal

    def test_mixed_k_and_physical(self):
        goal = And((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        obs = _make_obs()
        nodes = _collect_k_goals(goal, obs)
        k_map = {f"{n.agent}:{n.inner.to_pddl()}": n for n in nodes}
        result = _replace_epistemic_in_goal(goal, k_map)
        pddl = result.to_pddl()
        assert "(is_on_top bottle_4 table_13)" in pddl
        assert "knows_agent_0_" in pddl

    def test_nested_k_no_inner_knows_leak(self):
        """K(a0, K(a1, phi)) should NOT leak knows_a1_hash into the goal.

        The outer K expansion should produce And(phi, knows_a0_hash).
        The inner K, when it appears as a separate conjunct, gets its own
        expansion to And(phi, knows_a1_hash).
        """
        inner_lit = Literal("is_open", ("cabinet_27",))
        inner_k = Knows("agent_1", inner_lit)
        outer_k = Knows("agent_0", inner_k)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        nodes = _collect_k_goals(outer_k, obs)
        k_map = {f"{n.agent}:{n.inner.to_pddl()}": n for n in nodes}
        result = _replace_epistemic_in_goal(outer_k, k_map)
        pddl = result.to_pddl()
        # Should have the physical fact and outer agent's knowledge predicate
        assert "(is_open cabinet_27)" in pddl
        assert "knows_agent_0_" in pddl
        # Should NOT contain inner agent's leaf-fact knowledge predicate
        # (that would be a leaked inner knows_hash)
        inner_leaf_hash = _leaf_fact_hash(inner_lit)
        assert f"knows_agent_1_{inner_leaf_hash}" not in pddl


# ---------------------------------------------------------------------------
# Inference actions (nested K)
# ---------------------------------------------------------------------------

class TestNestedKActions:
    def test_both_can_observe_does_not_generate_inference(self):
        """Observing the world does not create knowledge about another agent's knowledge."""
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        obs = _make_obs(
            object_rooms={"cabinet_27": "kitchen_1"},
        )  # No restrictions → both can observe
        nodes = _collect_k_goals(outer, obs)
        all_agents = ["agent_0", "agent_1"]
        can_comm = {("agent_0", "agent_1"), ("agent_1", "agent_0")}
        actions = _build_nested_k_actions(nodes, all_agents, can_comm, obs)
        assert not any("infer_knows_agent_0" in a for a in actions)

    def test_outer_restricted_no_inference_but_has_inform(self):
        """K(a0, K(a1, phi)) where a0 restricted → no inference, but inform from a1."""
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        nodes = _collect_k_goals(outer, obs)
        all_agents = ["agent_0", "agent_1"]
        can_comm = {("agent_1", "agent_0")}
        actions = _build_nested_k_actions(nodes, all_agents, can_comm, obs)
        assert not any("infer_" in a for a in actions)
        assert any("inform_" in a and "from_agent_1" in a for a in actions)

    def test_k3_uses_nested_inner_predicate(self):
        """K3 preconditions must depend on the inner nested knowledge predicate, not the leaf fact."""
        inner = Knows("agent_2", Literal("is_open", ("cabinet_27",)))
        middle = Knows("agent_1", inner)
        outer = Knows("agent_0", middle)
        obs = _make_obs(object_rooms={"cabinet_27": "kitchen_1"})
        nodes = _collect_k_goals(outer, obs)
        all_agents = ["agent_0", "agent_1", "agent_2"]
        can_comm = {("agent_1", "agent_0"), ("agent_2", "agent_1")}
        actions = _build_nested_k_actions(nodes, all_agents, can_comm, obs)

        middle_node = next(n for n in nodes if n.agent == "agent_1" and n.depth == 2)
        leaf_hash = _leaf_fact_hash(Literal("is_open", ("cabinet_27",)))
        outer_action = next(a for a in actions if "inform_knows_agent_0" in a and "from_agent_1" in a)
        assert f"knows_agent_1_{middle_node.fact_id}" in outer_action
        assert f"knows_agent_1_{leaf_hash}" not in outer_action


# ---------------------------------------------------------------------------
# Full compilation (compile_epistemic)
# ---------------------------------------------------------------------------

class TestCompileEpistemic:
    def test_no_k_passthrough(self):
        """No epistemic goals → passthrough, no augmentation."""
        goal = Literal("is_open", ("cabinet_27",))
        problem = _make_problem(goal)
        obs = _make_obs(object_rooms={"cabinet_27": "kitchen_1"})
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        assert result.belief_depth == 0
        assert result.trivial_k_goals == []
        # Domain PDDL should be unchanged planning PDDL
        assert "knows_" not in result.domain_pddl

    def test_trivial_k_compiles_observe(self):
        """Trivial K → observe action in domain."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_1": {"kitchen_1"}},  # Only agent_1 restricted
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        assert "observe_knows_agent_0" in result.domain_pddl
        assert "knows_agent_0_" in result.domain_pddl
        assert result.belief_depth == 0  # Trivial K → depth 0
        assert len(result.trivial_k_goals) == 1

    def test_non_trivial_k_compiles_inform(self):
        """Non-trivial K → inform action in domain."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        assert "inform_knows_agent_0" in result.domain_pddl
        assert result.belief_depth == 1
        assert result.trivial_k_goals == []

    def test_mixed_k_and_physical_goal(self):
        """Both physical and K goals in compilation output."""
        goal = And((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Literal("is_on_top", ("cabinet_27", "cabinet_27")),
        ))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        assert "is_on_top" in result.problem_pddl
        assert "knows_agent_0_" in result.problem_pddl

    def test_budget_tokens_in_problem(self):
        """Budget tokens appear in problem init."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
            message_limits={"agent_1": 2},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        assert "msg_tok_agent_1_1" in result.problem_pddl
        assert "msg_tok_agent_1_2" in result.problem_pddl

    def test_or_with_k_branches(self):
        """Or with K branches both get compiled."""
        goal = Or((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Literal("is_open", ("cabinet_27",)),
        ))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        assert "knows_agent_0_" in result.problem_pddl
        assert "(or" in result.problem_pddl

    def test_not_k_knowledge_starts_false(self):
        """Not(K(a, phi)): knowledge predicate starts false (CWA) → satisfied."""
        goal = Not(operand=Knows("agent_0", Literal("is_open", ("cabinet_27",))))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        # The goal should contain (not ... knows_agent_0_...)
        assert "(not" in result.problem_pddl

    def test_nested_k2_compiles(self):
        """K(a0, K(a1, phi)) → two knowledge predicates."""
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        problem = _make_problem(outer)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(outer, ENACTTOM_DOMAIN, problem, obs)
        # Should have two knowledge predicates
        assert "knows_agent_0_" in result.domain_pddl
        assert "knows_agent_1_" in result.domain_pddl
        # K(a0, K(a1, phi)) where a0 is restricted → belief depth 2
        assert result.belief_depth == 2


# ---------------------------------------------------------------------------
# Domain PDDL augmentation
# ---------------------------------------------------------------------------

class TestDomainAugmentation:
    def test_predicates_injected(self):
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        # Knowledge predicate should be in :predicates section
        assert "(knows_agent_0_" in result.domain_pddl

    def test_actions_injected(self):
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        # Inform action should be before closing paren
        assert "(:action inform_knows_agent_0" in result.domain_pddl

    def test_original_actions_preserved(self):
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = compile_epistemic(goal, ENACTTOM_DOMAIN, problem, obs)
        # Original domain actions should still be present
        assert "(:action open" in result.domain_pddl
        assert "(:action navigate" in result.domain_pddl
        assert "(:action place" in result.domain_pddl


# ---------------------------------------------------------------------------
# FastDownwardSolver integration (requires unified-planning)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
class TestFDSolverEpistemicCompilation:
    def test_trivial_k_solvable(self):
        """K(a0, phi) where a0 can observe → solvable via observe."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_1": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert len(result.trivial_k_goals) == 1

    def test_non_trivial_k_solvable_via_inform(self):
        """K(a0, phi) where a0 restricted → solvable via inform from a1."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert result.belief_depth == 1

    def test_no_comm_path_unsolvable(self):
        """K(a0, phi), a0 restricted, no can_communicate to a0 → unsolvable."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(
            goal,
            init=[
                Literal("is_closed", ("cabinet_27",)),
                Literal("agent_in_room", ("agent_0", "bedroom_1")),
                Literal("agent_in_room", ("agent_1", "kitchen_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                # No can_communicate!
            ],
        )
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert not result.solvable

    def test_budget_exhaustion_unsolvable(self):
        """3 K-facts needed but budget is 2 → unsolvable."""
        goal = And((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Knows("agent_0", Literal("is_open", ("drawer_5",))),
            Knows("agent_0", Literal("is_open", ("shelf_8",))),
        ))
        problem = _make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "cabinet_27": "furniture",
                "drawer_5": "furniture",
                "shelf_8": "furniture",
                "kitchen_1": "room",
                "bedroom_1": "room",
            },
            init=[
                Literal("is_closed", ("cabinet_27",)),
                Literal("is_closed", ("drawer_5",)),
                Literal("is_closed", ("shelf_8",)),
                Literal("can_communicate", ("agent_1", "agent_0")),
                Literal("agent_in_room", ("agent_0", "bedroom_1")),
                Literal("agent_in_room", ("agent_1", "kitchen_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                Literal("is_in_room", ("drawer_5", "kitchen_1")),
                Literal("is_in_room", ("shelf_8", "kitchen_1")),
            ],
        )
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={
                "cabinet_27": "kitchen_1",
                "drawer_5": "kitchen_1",
                "shelf_8": "kitchen_1",
            },
            message_limits={"agent_1": 2},  # Only 2 messages, need 3
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert not result.solvable

    def test_mixed_k_and_physical_solvable(self):
        """Both K and physical goals must be satisfied."""
        goal = And((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Literal("is_open", ("cabinet_27",)),
        ))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable

    def test_no_k_fast_path(self):
        """Non-epistemic goals skip compilation entirely."""
        goal = Literal("is_open", ("cabinet_27",))
        problem = _make_problem(goal)
        obs = _make_obs(object_rooms={"cabinet_27": "kitchen_1"})
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable

    def test_nested_k2_solvable_with_relay(self):
        """K(a0, K(a1, phi)) with communication → solvable."""
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        problem = _make_problem(
            outer,
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "cabinet_27": "furniture",
                "kitchen_1": "room",
                "bedroom_1": "room",
            },
            init=[
                Literal("is_closed", ("cabinet_27",)),
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_0")),
                Literal("agent_in_room", ("agent_0", "bedroom_1")),
                Literal("agent_in_room", ("agent_1", "kitchen_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            ],
        )
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert result.belief_depth == 2

    def test_both_restricted_unsolvable(self):
        """Both agents restricted from target room → unsolvable epistemic goal."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(
            goal,
            init=[
                Literal("is_closed", ("cabinet_27",)),
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_0")),
                Literal("agent_in_room", ("agent_0", "bedroom_1")),
                Literal("agent_in_room", ("agent_1", "bedroom_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            ],
        )
        obs = _make_obs(
            restricted={
                "agent_0": {"kitchen_1"},
                "agent_1": {"kitchen_1"},
            },
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert not result.solvable

    def test_no_observability_falls_back_to_physical(self):
        """When observability has no object_rooms, fall back to physical-only."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            # No object_rooms → falls back
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert "Strict epistemic proof requires explicit observability grounding" not in (result.error or "")

    def test_no_observability_strict_fails_closed(self):
        """Strict mode must not drop epistemic goals to physical-only solving."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        obs = _make_obs(
            restricted={"agent_0": {"kitchen_1"}},
            # No object_rooms → strict proof should fail
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs, strict=True)
        assert not result.solvable
        assert "Strict epistemic proof requires explicit observability grounding" in result.error

    def test_relay_chain_solvable(self):
        """K(a2, phi): a2 restricted, a0 can see, but 0→1→2 only.

        Relay: a0 observes → tells a1 → a1 tells a2.
        """
        goal = Knows("agent_2", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "agent_2": "agent",
                "cabinet_27": "furniture",
                "kitchen_1": "room",
                "bedroom_1": "room",
            },
            init=[
                Literal("is_closed", ("cabinet_27",)),
                # Chain: 0→1, 1→2 only
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_2")),
                Literal("agent_in_room", ("agent_0", "kitchen_1")),
                Literal("agent_in_room", ("agent_1", "bedroom_1")),
                Literal("agent_in_room", ("agent_2", "bedroom_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            ],
        )
        obs = _make_obs(
            restricted={
                "agent_1": {"kitchen_1"},
                "agent_2": {"kitchen_1"},
            },
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert result.solvable
        # Plan should include relay steps
        assert any("inform" in step for step in result.plan)

    def test_relay_chain_unsolvable_broken_link(self):
        """K(a2, phi): relay needed but 0→1 only, no 1→2."""
        goal = Knows("agent_2", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "agent_2": "agent",
                "cabinet_27": "furniture",
                "kitchen_1": "room",
                "bedroom_1": "room",
            },
            init=[
                Literal("is_closed", ("cabinet_27",)),
                # Only 0→1, no path to a2
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("agent_in_room", ("agent_0", "kitchen_1")),
                Literal("agent_in_room", ("agent_1", "bedroom_1")),
                Literal("agent_in_room", ("agent_2", "bedroom_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            ],
        )
        obs = _make_obs(
            restricted={
                "agent_1": {"kitchen_1"},
                "agent_2": {"kitchen_1"},
            },
            object_rooms={"cabinet_27": "kitchen_1"},
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert not result.solvable

    def test_k3_ring_unsolvable(self):
        """One-way relay can spread the leaf fact without achieving the required nested knowledge."""
        inner = Knows("agent_2", Literal("is_open", ("cabinet_27",)))
        middle = Knows("agent_1", inner)
        outer = Knows("agent_0", middle)
        problem = _make_problem(
            outer,
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "agent_2": "agent",
                "agent_3": "agent",
                "cabinet_27": "furniture",
                "kitchen_1": "room",
                "bedroom_1": "room",
            },
            init=[
                Literal("is_closed", ("cabinet_27",)),
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_2")),
                Literal("can_communicate", ("agent_2", "agent_3")),
                Literal("can_communicate", ("agent_3", "agent_0")),
                Literal("agent_in_room", ("agent_0", "kitchen_1")),
                Literal("agent_in_room", ("agent_1", "bedroom_1")),
                Literal("agent_in_room", ("agent_2", "kitchen_1")),
                Literal("agent_in_room", ("agent_3", "bedroom_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            ],
        )
        obs = _make_obs(
            restricted={
                "agent_1": {"kitchen_1"},
                "agent_3": {"kitchen_1"},
            },
            object_rooms={"cabinet_27": "kitchen_1"},
            message_limits={
                "agent_0": 1,
                "agent_1": 1,
                "agent_2": 1,
                "agent_3": 1,
            },
        )
        result = FastDownwardSolver().solve(ENACTTOM_DOMAIN, problem, obs)
        assert not result.solvable
