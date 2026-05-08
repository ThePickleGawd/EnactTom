# AGENTS.md

## Conventions

- Use `./enacttom/run_enacttom.sh` as the main entry point. Do not call internal Python entrypoints directly unless there is a clear reason.
- Do not commit unless explicitly asked.
- Keep the implementation dead simple. Prefer one correct path over extra flags, compatibility layers, or speculative abstractions.
- Do not hardcode logic that should scale with the benchmark, especially prompt content, action descriptions, mechanic handling, and agent-specific behavior.
- Keep `README.md` brief: setup, quick start, and pointers.
- Keep `docs/*.md` as the single source of truth for benchmark architecture and conceptual behavior.
- When the benchmark architecture changes, update `docs/*.md` in the same change.

## Architecture

- The benchmark has a simple pipeline: explore scenes, generate tasks, verify solvability, judge ToM quality, then benchmark agents.
- `enacttom/pddl/` owns goal syntax, epistemic compilation, and solvability checks.
- `enacttom/task_gen/` owns task authoring, validation, and calibration flow.
- `enacttom/runner/` and `enacttom/cli/` own execution surfaces and user-facing commands.
- `docs/*.md` should describe the intended system shape. If the code and docs disagree, fix one immediately.
- When asked about functional or literal ToM, the definition comes from here: https://arxiv.org/html/2412.19726v4