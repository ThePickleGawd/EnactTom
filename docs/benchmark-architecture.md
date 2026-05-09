# Benchmark Architecture

This file is the source of truth for the EnactToM benchmark architecture.

## Goal

EnactToM measures embodied Theory of Mind. A benchmark task is valid when physical success requires agents to act under asymmetric information, communicate, and reason about what other agents know.

Scoring keeps two signals separate:

- `functional_success`: whether the agents physically complete the task under the benchmark information setting
- `literal_tom_probe`: whether agents can answer end-of-episode belief probes derived from authored `K()` goals

Task generation optimizes for functional ToM. Literal probes are diagnostic and must not replace the functional benchmark score.

## Paper Pipeline

The retained pipeline is:

1. Load a Habitat scene with `new_scene`.
2. Select sampled seed tasks from the existing pool using calibrated pass/fail outcomes for the target model.
3. Author a cooperative or mixed task grounded in that scene.
4. Validate the task JSON and PDDL goal.
5. Verify the golden trajectory in Habitat.
6. Judge whether the task genuinely requires ToM reasoning.
7. Run the empirical `test_task` gate, including standard-mode failure and baseline solvability calibration.
8. Submit accepted tasks.
9. Benchmark agents on the accepted task set.

Bulk task generation is native to `./enacttom/run.sh generate --bulk`. Bulk workers still run the same one-task generation path; the launcher only assigns worker IDs, GPU slots, categories, and shared run/output directories.

There is no separate campaign, migration, salvage, reinforcement-learning, or evolution pipeline. Difficulty shaping is part of normal generation:

- `standard` generation targets a 10% pass rate for the target model
- `hard` generation caps target-model pass rate at 5%
- seed examples are sampled with mostly target-model failures and a small number of passes
- sampled examples are inspiration only and are never copied directly into a new task

## Categories

Paper reproduction uses only:

- `cooperative`
- `mixed`

Competitive task generation and competitive team-vs-team benchmarking are out of scope for the minimal paper code path.

## Agent Counts

Paper reproduction uses the retained 2-, 3-, and 4-agent Habitat presets. All retained presets use Spot robots with the same paper benchmark action surface: `Navigate`, `Pick`, `Place`, `Open`, `Close`, `Communicate`, `Wait`, and the scene lookup tools.

## Mechanics

Task authoring supports exactly these mechanics:

- `room_restriction`
- `limited_bandwidth`
- `restricted_communication`
- `remote_control`
- `state_mirroring`
- `inverse_state`

Task-added item schemas, locked-container key logic, stun actions, and legacy one-off mechanics are out of scope.

## Benchmark Modes

- `standard`: task secrets remain private, partial observability is enabled, and agents only observe through normal benchmark channels
- `baseline`: task secrets are shared and the information bottleneck is removed to check whether the task is solvable when ToM pressure is relaxed

Functional success ignores `K()` and evaluates the projected non-epistemic goal. PDDL solvability, deterministic planning, and golden-trajectory verification solve the same functional goal.

## Code Ownership

- `enacttom/pddl/`: goal language, epistemic compilation, and solvability checks
- `enacttom/task_gen/`: task authoring surface, seed selection, validation, judging, calibration, and submission gates
- `enacttom/runner/`: Habitat execution runtime
- `enacttom/cli/`: stable command surfaces used by `./enacttom/run.sh` and external authoring agents
- `docs/*.md`: conceptual behavior and architecture

## Invariants

- Keep `./enacttom/run.sh` as the main entry point.
- Keep one clear implementation path for each paper flow.
- Keep `problem_pddl` as the single authored source of epistemic structure and goals.
- Generate scene objects and mechanic init facts deterministically from the loaded scene snapshot and `mechanic_bindings`.
- Keep public task text non-leaking; exact hidden object IDs belong only in private secrets and PDDL.
- Runtime handlers and planner compilation must agree on mechanic semantics.
- End-of-episode literal-ToM probes are derived deterministically from `K()` formulas and reported separately from physical success.
- Update this file whenever the benchmark structure or invariants change.
