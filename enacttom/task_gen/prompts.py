"""Prompt template for external minisweagent task generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


MINISWEAGENT_TASKGEN_PROMPT = """You are generating multi-agent benchmark tasks in `{working_dir}`.

Use normal shell commands for inspection and file edits.
Use the repo-owned `taskgen` commands for scene loading, judging, testing, submission, and finish/fail.
{query_block}{verification_block}{calibration_block}{k_level_block}{sampled_task_block}
Generate {num_tasks} quality benchmark tasks.

## Constraints
- Agents: {agents_min}-{agents_max}
- Goal conjuncts: {subtasks_min}-{subtasks_max}

## Required Commands
- `taskgen status`
- `taskgen new_scene N`
- `taskgen new_scene N --keep`
- `taskgen judge`
{test_command}- `taskgen verify_task`
- `taskgen submit_task`
- `taskgen finish`
- `taskgen fail "reason"` (ONLY for truly unrecoverable errors like broken environment; NEVER for judge failures)

## Workflow
{workflow_block}

## Working Files
- `{task_file}`: edit this task JSON.
- `{working_dir}/current_scene.json`: current scene after `taskgen new_scene`.
- `{working_dir}/template.json`: task structure reference.
- `{benchmark_feedback_file}`: created after failed `taskgen test_task` or `taskgen verify_task`. If it exists, read it before editing again and fix the listed issues first.
- Commands already start in `{working_dir}`. Do not prefix every command with `cd {working_dir} &&`.
{sampled_files_block}- `available_predicates.md`, `available_mechanics.md`, `available_actions.md`: inspect only when needed.

## Hard Authoring Rules
- Use exact scene IDs and only valid agent IDs returned by `taskgen new_scene`.
- Remove placeholder text.
- Every mechanic must materially affect the task.
- Do not hand-author `golden_trajectory`.
- If `message_targets` is present, it already acts as a valid communication restriction.
- Use scene objects only; do not invent runtime schemas outside the template.
- Use canonical mechanic schema only:
  `room_restriction` -> `restricted_rooms` + `for_agents`
  `limited_bandwidth` -> `message_limits`
  `restricted_communication` -> `allowed_targets`
{pddl_rules}{skip_test_rule}
## Secret Formatting Rules (judge hard-blocks on violations)
- Secrets must state ONLY positive private facts or constraints: room bans, communication limits/targets, private observations, exact IDs for facts the agent already knows, goal states, and private objectives.
- NEVER use prescriptive language: 'Tell your partner', 'Ask them', 'Leave it at', 'Coordinate with', 'You should'.
- NEVER add ignorance lines like 'You do not know where ...', 'You do not know which ...', or 'You do not know whether ...'. If a fact is unknown to the agent, omit it.
- NEVER add epistemic coaching like 'By the end, you must be confident ...' or 'Epistemic probe: ...' to `agent_secrets`.
- If a task uses `inverse_state`, `state_mirroring`, or `remote_control`, the affected agent's secret may briefly state that mechanic fact in plain language, but it must NOT tell the agent what message or plan to use.
- If an agent lacks an object's identity or location, do NOT reveal the exact runtime object ID in that agent's secret or in the public `task`. Prefer role/type language in the public task and keep exact IDs only in the secrets of agents who actually know them.
- BUG WARNING: writing 'agent_X cannot enter room_Y' in agent_Z's secrets is parsed as agent_Z's own restriction. Use 'agent_X is barred from room_Y' when describing another agent's restriction.

## Category Rules
{category_rules}

## Good ToM
- The core task should require an agent's correct action choice to depend on another agent's private knowledge, access, or observation.
- Good pattern: agent A cannot determine the right object, room, or target state until agent B observes or communicates it.
- Bad pattern: agents can finish the physical goal independently and communication only reports what already happened.
- Every essential agent should contribute distinct knowledge, access, or incentive.
- The main difference between standard and baseline is information access. For hard tasks, focus first on secrets, knowledge placement, and answerable hidden facts before adding more physical complexity.
- Use `K()` only for facts that matter for planning or coordination.
- The outermost `K()` agent should not be able to directly observe the fact with no blocker.

## K() Epistemic Goal Rules
- Every task MUST include at least one `(K agent_X (predicate args))` in problem_pddl `:goal`. K=0 is rejected.
- The K() agent must be restricted from the room where the predicate becomes true, forcing them to learn via communication.
- Example: agent_0 restricted from kitchen_1 -> add `(K agent_0 (is_open fridge_27))` where fridge_27 is in kitchen_1.
- Do NOT add a matching `agent_secrets` line for the K() goal. The epistemic requirement belongs in `problem_pddl`, while secrets should only contain private facts and constraints.
- NEVER expose K() goals in the `task` text field. The `task` must read as a natural household instruction with no mention of "must know", "knows that", belief, or epistemic requirements. K() goals live only in `problem_pddl`.

## Empirical Solvability
- Keep the physical execution short and direct. Prefer tasks that baseline can finish in roughly 6-10 turns.
- Prefer one clean asymmetry over stacked brittle mechanics. One room/access blocker plus one decisive hidden fact is better than a long chain of dependencies.
- If you want baseline to pass but standard to fail, first improve the agent secrets and information split. Do not default to piling on extra objects, rooms, or mechanics.
- For harder tasks, actively consider the full supported mechanic set. `remote_control`, `state_mirroring`, and `inverse_state` are valid choices when they create one decisive hidden semantic twist or confirmation dependency without bloating the physical core.
- Use exact scene IDs in `problem_pddl` and in the secrets of agents who already know the fact. Do NOT leak hidden target object IDs in the public `task` or in any secret for an agent who does not already know that fact.
- Avoid relying on vague aliases like 'display table' or hidden trigger objects whose exact runtime ID is hard to recover.
- NEVER design tasks requiring object handoff through a shared room, agents try Place[obj, on, room_name] and fail at runtime.
- If a task passes `judge` but fails `test_task`, simplify the physical core first before adding more ToM structure.
- If `{benchmark_feedback_file}` exists, treat it as the ground truth for why the last benchmark gate rejected the task. Fix those issues before trying new variations.

## Empirical Winning Formula (from 46 tasks that passed test_task)
This is the most common solvable pattern in the current pool, not the only acceptable stack. Use it as one reference point, then vary.

**Most common mechanics:** `room_restriction` + `limited_bandwidth` (1 msg per agent). Optionally add `restricted_communication`.
**Physical goals:** 3-5 for cooperative, 6-9 for mixed. Keep physical actions simple (Place, Open, Close).
**Agents:** 2-3 agents. 3 agents works best for mixed (one per hidden role).
**Room restrictions:** 2 restrictions (each agent barred from one room). This creates natural information gaps.
**Key insight:** The decisive physical action must depend on a fact observable ONLY from a restricted room. The agent who CAN observe it must communicate it within the 1-message limit.
**Hard-task variation:** Also consider `remote_control`, `state_mirroring`, and `inverse_state` when one of them creates a cleaner hidden dependency than another room/access bottleneck.

**Common failure modes to avoid:**
- Cooperative tasks are "too easy" 67% of the time → the information gap isn't strong enough. Use non-binary choices (which of 3+ objects/surfaces) so the standard agent can't guess.
- Tasks with 5+ room_restrictions almost always have baseline failures — too many access constraints create deadlocks.

## Task Diversity
Vary at least TWO of these dimensions each task. Check sampled tasks to avoid duplicating patterns:
- Object type: plates, cups, bottles, vases, toys, boxes, laptops, candle holders, or scene-native alternatives.
- Room pair: kitchen<->bedroom, office<->dining, garage<->living, or other scene-supported topologies.
- Agent count: 2-3 agents (not 4+ unless the scene has 4+ rooms with distinct objects).
- Mechanic stack: draw from the full supported set: `room_restriction`, `limited_bandwidth`, `restricted_communication`, `remote_control`, `state_mirroring`, `inverse_state`. Use the mechanic that creates the cleanest ToM bottleneck for the scene.
- Goal structure: placement + state change + K(); or two placements + K(). 3-5 goals for cooperative, 6-9 for mixed.
- Knowledge split: A knows object / B knows target; A knows both but cannot reach; B can reach but knows neither; both know partial info.
- Narrative framing: household chore, museum setup, safety inspection, party prep, moving day, or another concrete scene-grounded story.
{test_gate_line}
- `taskgen verify_task` is the final submission gate: it runs `gpt-5.4-mini` in standard mode. You may submit if that model fails.

## Pre-Submit Checklist
- The physical goal requires communication or partner modeling.
- All referenced agents, objects, furniture, and rooms exist in the current scene.
- Category fields are valid for the selected category.
- Mechanics and secrets agree about actual constraints, but secrets do not explain the coordination plan.
- No malformed bindings, missing required mechanic fields, or invalid message limits.
{pddl_checklist}{test_checklist}- After `taskgen test_task` passes, run `taskgen verify_task`. Submit only if `gpt-5.4-mini` fails.
## References
- `available_predicates.md`: valid predicates and goal syntax.
- `available_mechanics.md`: mechanic names and fields.
- `available_actions.md`: supported runtime actions.
- Avoid repeating the same mechanic stack or target pattern across tasks in this run.
{removed_components_block}"""


def _strip_secret_strategy_rules(prompt: str) -> str:
    lines_to_remove = {
        "- NEVER use prescriptive language: 'Tell your partner', 'Ask them', 'Leave it at', 'Coordinate with', 'You should'.",
        "- If a task uses `inverse_state`, `state_mirroring`, or `remote_control`, the affected agent's secret may briefly state that mechanic fact in plain language, but it must NOT tell the agent what message or plan to use.",
        "- Mechanics and secrets agree about actual constraints, but secrets do not explain the coordination plan.",
    }
    filtered_lines = [line for line in prompt.splitlines() if line not in lines_to_remove]
    return "\n".join(filtered_lines)


def _count_raw_sampled_tasks(sampled_dir: Path) -> int:
    return sum(
        1
        for path in sampled_dir.glob("*.json")
        if not path.stem.endswith("_fields")
    )


def build_external_taskgen_prompt(
    *,
    working_dir: str,
    task_file: str,
    category: str,
    num_tasks: int,
    agents_min: int,
    agents_max: int,
    subtasks_min: int,
    subtasks_max: int,
    query: Optional[str] = None,
    verification_feedback: Optional[Dict[str, Any]] = None,
    calibration_stats: Optional[Dict[str, Any]] = None,
    difficulty: Optional[str] = None,
    current_k_level: Optional[int] = None,
    seed_tasks_dir: Optional[str] = None,
    seed_pass_ratio: float = 0.20,
    seed_fail_ratio: float = 0.80,
    skip_steps: Optional[List[str]] = None,
) -> str:
    skip = set(skip_steps or [])
    unsupported_skips = sorted(skip - {"seed-sampling", "secret-strategy"})
    if unsupported_skips:
        raise ValueError(
            "Cannot remove required pipeline components: "
            + ", ".join(unsupported_skips)
        )
    skip_seed_sampling = "seed-sampling" in skip
    skip_secret_strategy = "secret-strategy" in skip

    query_block = ""
    if query:
        query_block = f"\n\n## User Requirements\n{query}"

    verification_block = ""
    if verification_feedback:
        fixes = verification_feedback.get("required_fixes", [])
        fix_lines = "\n".join(f"- {fix}" for fix in fixes) or "- No fixes listed."
        verification_block = (
            "\n\n## Previous ToM Verification Failed\n"
            f"{verification_feedback.get('overall_reasoning', '')}\n\n"
            "Required Fixes:\n"
            f"{fix_lines}"
        )

    calibration_block = ""
    if difficulty:
        difficulty_map = {
            "medium": (
                "## Difficulty: MEDIUM\n"
                "- Keep the physical core small: 2-4 subtasks and usually one non-trivial K() chain.\n"
                "- Put most of the challenge in the information split, not in extra physical clutter.\n"
                "- Favor secrets that cleanly encode who knows the object, target, or decisive state fact.\n"
                "- Prefer one grounded final-state fact reused by both the physical goal and the K() goal."
            ),
            "hard": (
                "## Difficulty: HARD\n"
                "- Prefer tasks that the target model fails.\n"
                "- Hard-mode acceptance requires standard benchmark progress to stay below 45%; if standard reaches 45% or more, `test_task` will reject the task.\n"
                "- Make the core difficulty a **belief-routing problem**, not a search problem.\n"
                "- Use one or two decisive hidden facts, not a pile of decorative secrets.\n"
                "- Constrained but meaningful communication — messages must carry load.\n"
                "- Asymmetric room/action access that forces genuine delegation.\n"
                "- Precise confirmation chains (K() goals) that require agents to reason about "
                "what others have observed.\n"
                "- Keep shared goals crisp enough that a sensible role decomposition is possible early.\n"
                "- Actively consider `remote_control`, `state_mirroring`, or `inverse_state` when they "
                "create a clean hidden dependency; do not default to only room/access constraints.\n\n"
                "**Avoid** making tasks hard by:\n"
                "- Piling on unrelated subtasks (goal load without epistemic depth).\n"
                "- Broad object/room search (search burden without ToM).\n"
                "- Too many overlapping private conflicts (clutter, not genuine tension).\n\n"
                "## Diversity Requirement\n"
                "Your task MUST differ from any sampled tasks in at least TWO of:\n"
                "- Difficulty mechanism (different failure mode)\n"
                "- Scenario theme (not another staging/inspection/cleanup/walkthrough task)\n"
                "- ToM structure (different K-level or nesting pattern)\n"
                "- Mechanic combination (try different mechanic stacks)"
            ),
        }
        default_difficulty_text = (
            "## Difficulty Guidance\n"
            "- Keep the physical core compact and scene-grounded.\n"
            "- Preserve a real hidden-information dependency; do not weaken secrets just to make the task easier.\n"
            "- Prefer answerable K() facts and clean information flow over extra mechanics."
        )
        calibration_block = "\n\n" + difficulty_map.get(difficulty, default_difficulty_text)
    else:
        calibration_stats = calibration_stats or {}
        model = calibration_stats.get("model", "unknown")
        target_rate = calibration_stats.get("target_rate", 0.10)
        current_rate = calibration_stats.get("rate")
        hard_cap_mode = target_rate <= 0.05 + 1e-9
        if current_rate is None:
            if hard_cap_mode:
                calibration_text = (
                    f"No calibration data yet for {model}. Hard-task generation uses a hard pass-rate cap of {target_rate:.0%}.\n"
                    "Anything below the cap is acceptable. Discard any task whose standard pass would push the calibrated pool above the cap.\n"
                    "Hard-mode tasks must also keep standard benchmark progress below 45%; tasks at 45% or higher are too easy and must be discarded.\n"
                    "The test gate still requires baseline to pass."
                )
            else:
                calibration_text = (
                    f"No calibration data yet for {model}. Generate varied tasks.\n"
                    "The test gate only requires baseline to pass. Standard mode results are tracked but do not block submission."
                )
        elif hard_cap_mode:
            calibration_text = (
                f"Current {model} pass rate is {current_rate:.1%}; the hard cap is {target_rate:.0%}.\n"
                "Anything below the cap is acceptable. Discard any task whose standard pass would leave the calibrated pool above the cap.\n"
                "Hard-mode tasks must also keep standard benchmark progress below 45%; tasks at 45% or higher are too easy and must be discarded.\n"
                "The test gate REQUIRES baseline to pass, and tasks where standard also passes may still be rejected if they break the cap.\n"
                "To keep hard tasks under the cap:\n"
                "- The decisive action must depend on a fact the standard agent cannot observe.\n"
                "- The fact should be non-binary or otherwise hard to guess from public information.\n"
                "- Keep communication load-bearing so the right fact must be routed, not trivially broadcast.\n"
                "- Keep the physical core simple enough that baseline still passes."
            )
        elif current_rate > target_rate + 0.05:
            calibration_text = (
                f"Current {model} pass rate is {current_rate:.1%}, above the {target_rate:.0%} target.\n"
                "The test gate REQUIRES standard to FAIL while baseline passes. Tasks where standard also passes will be rejected.\n"
                "To create tasks that standard fails:\n"
                "- The decisive action must depend on a fact the standard agent cannot observe (room restriction blocks it).\n"
                "- The fact must be non-binary (not just open/closed) so the agent cannot guess correctly.\n"
                "- Limit bandwidth to 1 message per agent so information must be routed efficiently.\n"
                "- The physical core should be simple (2-3 goals) so baseline easily passes."
            )
        elif current_rate < target_rate - 0.05:
            calibration_text = (
                f"Current {model} pass rate is {current_rate:.1%}, below the {target_rate:.0%} target.\n"
                "Keep the physical core compact and make the hidden information easier to recover, without removing the real ToM dependency.\n"
                "The test gate requires baseline to pass."
            )
        else:
            calibration_text = (
                f"Current {model} pass rate is {current_rate:.1%}, near the {target_rate:.0%} target. Keep varied difficulty.\n"
                "The test gate requires baseline to pass."
            )
        by_category = calibration_stats.get("by_category", {})
        cat_lines = []
        for cat_name in ("cooperative", "mixed"):
            cs = by_category.get(cat_name, {})
            if cs.get("total", 0) > 0:
                cat_lines.append(f"  {cat_name}: {cs['rate']:.0%} pass ({cs['passed']}/{cs['total']})")
        if cat_lines:
            label = f"hard cap {target_rate:.0%}" if hard_cap_mode else f"each targeting {target_rate:.0%}"
            calibration_text += f"\n\nPer-category standard pass rates ({label}):\n" + "\n".join(cat_lines)

        calibration_block = f"\n\n## Dataset Calibration\n{calibration_text}"

    k_level_block = ""
    if current_k_level is not None:
        k_level_block = (
            f"\n\n## Required K-Level: {current_k_level}\n"
            f"This task must verify at ToM level {current_k_level}.\n"
            "K=0 tasks are invalid and will be rejected.\n"
            "Submissions are rejected if the computed tom_level does not match."
        )

    benchmark_feedback_file = str(Path(working_dir) / "benchmark_retry_feedback.md")
    sampled_task_block = ""
    sampled_files_block = ""
    # Only show sampled task guidance if the sampled_tasks directory has files.
    sampled_dir = Path(working_dir) / "sampled_tasks"
    sampled_task_count = _count_raw_sampled_tasks(sampled_dir) if sampled_dir.exists() else 0
    has_sampled_files = sampled_task_count > 0
    if not skip_seed_sampling and has_sampled_files:
        target_model = (calibration_stats or {}).get("model", "unknown")
        sampled_task_block = (
            "\n\n## Sampled Task Context\n"
            f"Target model: {target_model}. Sampled-task mix: fail {seed_fail_ratio:.0%}, pass {seed_pass_ratio:.0%}.\n"
            "Inspect sampled tasks before authoring. Read all sampled tasks in `sampled_tasks/`.\n"
            "Files named `failed_*` have a `_benchmark_result` with the agent's trajectory.\n"
            "Files named `passed_*` show tasks the target model could solve.\n"
            "Start with any `*_fields.json` compact views when present, then open the matching raw task JSON only when you need full benchmark-behavior detail.\n"
            "Study `task`, `active_mechanics`, `mechanic_bindings`, `agent_secrets`, `agent_actions`, `problem_pddl`, and `num_agents`.\n"
            "Borrow only structural patterns that look empirically solvable under test_task, especially short physical cores, strong secrets, and clean mechanic usage.\n"
            "Do not infer that rare mechanics are forbidden just because the sampled pool underuses them. The supported authoring set is `room_restriction`, `limited_bandwidth`, `restricted_communication`, `remote_control`, `state_mirroring`, and `inverse_state`.\n"
            "Start each task from the scene-grounded template in working_task.json. Do not copy a seed task directly."
        )
        sampled_files_block = (
            f"- `{working_dir}/sampled_tasks/*_fields.json`: compact sampled-task views when present. Start here.\n"
            f"- `{working_dir}/sampled_tasks/*.json`: raw sampled tasks with benchmark results.\n"
            "- Study `task`, `active_mechanics`, `mechanic_bindings`, `agent_secrets`, `agent_actions`, `problem_pddl`, and `num_agents` first.\n"
        )

    test_command = "- `taskgen test_task`\n"
    skip_test_rule = ""
    test_gate_line = "- `taskgen test_task` is the real execution gate: `judge` is not enough if baseline still cannot complete the task.\n\n"
    test_checklist = "- After `taskgen judge` passes, run `taskgen test_task` before submitting.\n\n"

    pddl_rules = (
        "- Treat `problem_pddl` as machine-owned except for `:goal` and optional `:goal-owners`.\n"
        "- Do not hand-edit `:objects` or `:init`.\n"
        "- Use only predicates from `available_predicates.md`.\n\n"
    )
    pddl_checklist = (
        "- `problem_pddl` has a valid `:goal` and, when needed, valid `:goal-owners`.\n"
        "- `:objects` and `:init` were not hand-edited.\n"
    )

    workflow_lines = [
        "1. Run `taskgen status`.",
        f"2. Run `taskgen new_scene N` with `N` between {agents_min} and {agents_max}. Never use `1`. Use only the returned `valid_agent_ids` in mechanic bindings, secrets, and message targets.",
    ]
    if not skip_seed_sampling and has_sampled_files:
        workflow_lines.append(
            f"3. Read all {sampled_task_count} sampled tasks in `{working_dir}/sampled_tasks/` before authoring. Start with any `*_fields.json` compact views when present. For each task, inspect `task`, `active_mechanics`, `mechanic_bindings`, `agent_secrets`, `agent_actions`, `problem_pddl`, and `num_agents`. Open the matching raw task JSON only if you need `calibration` or extra benchmark-behavior detail. Look for good practices in secrets, information splits, and mechanic usage. Reuse only structural patterns that look empirically solvable. Do not copy IDs directly."
        )
    edit_step_number = len(workflow_lines) + 1
    edit_text = "Author `problem_pddl :goal` first, then make the natural-language fields and mechanics match it."
    workflow_lines.append(f"{edit_step_number}. Edit `{task_file}`. {edit_text}")
    workflow_lines.append(f"{edit_step_number + 1}. Run `taskgen judge`, fix issues, and repeat until it passes.")
    final_step_number = edit_step_number + 2
    workflow_lines.append(f"{final_step_number}. Run `taskgen test_task`.")
    final_step_number += 1
    workflow_lines.append(f"{final_step_number}. Run `taskgen verify_task`.")
    workflow_lines.append(
        f"{final_step_number + 1}. If `taskgen test_task` or `taskgen verify_task` rejects the task, read `{benchmark_feedback_file}` and fix those exact issues before editing further."
    )
    workflow_lines.append(
        f"{final_step_number + 2}. Submit only if verification passes, meaning `gpt-5.4-mini` fails in standard mode."
    )
    workflow_lines.append(f"{final_step_number + 3}. Run `taskgen submit_task`.")
    workflow_lines.append(f"{final_step_number + 4}. When you have submitted {num_tasks} tasks, run `taskgen finish`.")
    workflow_block = "\n".join(workflow_lines)

    if category == "cooperative":
        category_rules = (
            "- Requested category: `COOPERATIVE`.\n"
            "- All goals are shared.\n"
            + "- Do not include `:goal-owners`.\n"
        )
    elif category == "mixed":
        category_rules = (
            "- Requested category: `MIXED`.\n"
            "- Public `task` covers only the shared objective.\n"
            "- Each relevant agent must have a hidden personal objective.\n"
            + (
                "- Put personal objectives in `:goal-owners` using entries like `(agent_0 (is_open cabinet_10))`, not `(personal agent_0 ...)`.\n"
            )
        )
    else:
        category_rules = (
            "- Requested category: random over `cooperative` and `mixed`.\n"
            "- Pick the category that best fits the scene and obey its invariants.\n"
        )

    removed_components_block = ""
    if skip:
        removed_lines = [
            "",
            "## Omitted Optional Context",
            f"The following optional prompt context is omitted for this run: **{', '.join(sorted(skip))}**.",
            "Do not attempt to run or rely on these components.",
        ]
        if skip_seed_sampling:
            removed_lines.append("- `seed-sampling`: no seed tasks or calibration data are available.")
        if skip_secret_strategy:
            removed_lines.append("- `secret-strategy`: the prompt omits the rule that forbids strategy instructions in `agent_secrets`.")
        removed_components_block = "\n" + "\n".join(removed_lines)

    prompt = MINISWEAGENT_TASKGEN_PROMPT.format(
        working_dir=working_dir,
        task_file=task_file,
        benchmark_feedback_file=benchmark_feedback_file,
        query_block=query_block,
        verification_block=verification_block,
        calibration_block=calibration_block,
        k_level_block=k_level_block,
        sampled_task_block=sampled_task_block,
        num_tasks=num_tasks,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        test_command=test_command,
        workflow_block=workflow_block,
        sampled_files_block=sampled_files_block,
        pddl_rules=pddl_rules,
        skip_test_rule=skip_test_rule,
        category_rules=category_rules.rstrip(),
        test_gate_line=test_gate_line,
        pddl_checklist=pddl_checklist,
        test_checklist=test_checklist,
        removed_components_block=removed_components_block,
    )
    if skip_secret_strategy:
        prompt = _strip_secret_strategy_rules(prompt)
    return prompt
