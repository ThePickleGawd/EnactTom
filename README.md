# EnactToM

EnactToM is a minimal research benchmark for embodied Theory of Mind. Agents act in Habitat scenes with asymmetric information, communicate through restricted channels, and are scored on whether they complete tasks that require reasoning about what other agents know.

The conceptual source of truth is [docs/benchmark-architecture.md](docs/benchmark-architecture.md). Use `./enacttom/run_enacttom.sh` as the operator entry point.

## Setup

Create the EnactToM environment:

```bash
mamba create -n enacttom python=3.9.2 cmake=3.14.0 -y
mamba activate enacttom
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use `conda` in place of `mamba` if needed. Habitat execution requires the full simulator and dataset setup described in [docs/installation.md](docs/installation.md).

Configure model credentials through environment variables or a repo-root `.env` file:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

## Quick Start

Generate paper-style tasks:

```bash
./enacttom/run_enacttom.sh generate --num-tasks 3 --difficulty standard
./enacttom/run_enacttom.sh generate --num-tasks 3 --difficulty hard
./enacttom/run_enacttom.sh generate --bulk --num-tasks 24 --per-gpu 3
```

Validate, solve, replay, and judge a task:

```bash
./enacttom/run_enacttom.sh validate-task --task data/enacttom/tasks/example.json
./enacttom/run_enacttom.sh verify-pddl --task data/enacttom/tasks/example.json
./enacttom/run_enacttom.sh verify --task data/enacttom/tasks/example.json
./enacttom/run_enacttom.sh judge --task data/enacttom/tasks/example.json
```

Benchmark a task set:

```bash
./enacttom/run_enacttom.sh benchmark --tasks-dir data/enacttom/tasks --model gpt-5.4 --num-times 3
```

Repeated benchmark runs report mean pass rate, pass-rate standard deviation, `pass@k`, and `pass^k` with `k = --num-times`.

## Scope

This repository keeps only the code needed to recreate the EnactToM paper pipeline:

- task generation through sampled scenes, external authoring agents, validation, judging, and submission
- deterministic PDDL and golden-trajectory solvability checks
- standard and hard benchmark calibration
- functional task success and separate literal-ToM probe reporting
- benchmark execution for cooperative and mixed tasks

Supported Habitat presets are the paper-scale 2-, 3-, and 4-agent Spot robot configurations.

Supported authoring mechanics are `room_restriction`, `limited_bandwidth`, `restricted_communication`, `remote_control`, `state_mirroring`, and `inverse_state`.
