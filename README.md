# EnactToM

EnactToM is a research benchmark for embodied Theory of Mind. Agents act in
Habitat scenes under asymmetric information, communicate through restricted
channels, and are evaluated on whether they can complete tasks that require
reasoning about what other agents know.

The benchmark design is summarized in
[docs/benchmark-architecture.md](docs/benchmark-architecture.md). The command
line entry point is always `./enacttom/run.sh`.

## Installation

Create the default authoring environment and install EnactToM:

```bash
conda create -n enacttom python=3.10 cmake=3.14.0 -y
conda activate enacttom
python -m pip install -r requirements.txt
python -m pip install -e .
```

`mamba` can be used in place of `conda`. The requirements install the Python
packages used by local validation, PDDL solving, tests, and the default `mini`
task-generation agent. Habitat scene execution additionally requires the
simulator env and assets described in [docs/installation.md](docs/installation.md).
The split is required because `mini-swe-agent` is Python 3.10+ while
`habitat-sim==0.3.3` is installed in a Python 3.9 Habitat env.

Run smoke checks:

```bash
bash -n enacttom/run.sh
python -m compileall -q enacttom habitat_llm tests
python -m pytest
./enacttom/run.sh --help
```

## Credentials

Task generation, judging, and benchmarking use model APIs. Configure keys in
the shell or in a repo-root `.env` file:

```bash
OPENAI_API_KEY=...
```

## Quick Start

Generate benchmark tasks after the Habitat setup is complete:

```bash
conda activate enacttom-habitat
export PATH="$(conda info --base)/envs/enacttom/bin:$PATH"
./enacttom/run.sh generate --num-tasks 3 --difficulty standard
./enacttom/run.sh generate --num-tasks 3 --difficulty hard
```

`new-scene` and `generate` require real Habitat episodes and fail if Habitat
dependencies or assets are missing.

Validate and solve a generated task:

```bash
TASK=path/to/task.json
./enacttom/run.sh validate-task --task "$TASK"
./enacttom/run.sh verify-pddl --task "$TASK"
./enacttom/run.sh verify --task "$TASK"
./enacttom/run.sh judge --task "$TASK"
```

Benchmark a task set:

```bash
./enacttom/run.sh benchmark \
  --tasks-dir data/enacttom/tasks \
  --model gpt-5.4-mini \
  --num-times 3
```

Repeated benchmark runs report mean pass rate, pass-rate standard deviation,
`pass@k`, and `pass^k` for `k = --num-times`.

## Scope

This release contains the EnactToM paper pipeline: scene exploration, task
generation, validation, PDDL solvability checks, Habitat replay, ToM judging,
and agent benchmarking. Supported Habitat presets are the paper-scale 2-, 3-,
and 4-agent Spot robot configurations.

Supported task mechanics are `room_restriction`, `limited_bandwidth`,
`restricted_communication`, `remote_control`, `state_mirroring`, and
`inverse_state`.
