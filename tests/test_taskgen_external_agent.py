from datetime import datetime
import json
from pathlib import Path

from enacttom.task_gen.external_agent import ExternalAgentLauncher
from enacttom.task_gen.prompts import build_external_taskgen_prompt
from enacttom.task_gen.runner import (
    _build_run_manifest_update,
    _copy_sample,
    _write_bootstrap_files,
    build_workspace_id,
)
from enacttom.task_gen.session import TaskGenSession, default_state


def test_external_agent_launcher_builds_backend_commands(tmp_path):
    launcher = ExternalAgentLauncher(tmp_path)

    mini_cmd = launcher.build_command(
        agent_name="mini",
        executable="/tmp/mini",
        workspace_dir=tmp_path,
        bootstrap_prompt="read the prompt",
        model="gpt-5",
    )
    claude_cmd = launcher.build_command(
        agent_name="claude",
        executable="/tmp/claude",
        workspace_dir=tmp_path,
        bootstrap_prompt="read the prompt",
        model="sonnet",
    )
    codex_cmd = launcher.build_command(
        agent_name="codex",
        executable="/tmp/codex",
        workspace_dir=tmp_path,
        bootstrap_prompt="read the prompt",
        model="o3",
    )

    assert mini_cmd[:5] == ["/tmp/mini", "-c", "mini.yaml", "-c", "environment.timeout=1200"]
    assert "-y" in mini_cmd
    assert "--exit-immediately" in mini_cmd
    assert "-m" in mini_cmd
    assert "openai/gpt-5" in mini_cmd
    assert str(tmp_path / "mini_trajectory.json") in mini_cmd
    assert mini_cmd[-1] == "read the prompt"
    assert "--model" in claude_cmd and "sonnet" in claude_cmd
    assert claude_cmd[-1] == "read the prompt"
    assert codex_cmd[:5] == [
        "/tmp/codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(tmp_path),
    ]
    assert "--model" in codex_cmd and "o3" in codex_cmd


def test_build_agent_env_clears_conda_and_prefers_workspace(tmp_path):
    launcher = ExternalAgentLauncher(tmp_path)
    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "bin").mkdir(parents=True)
    launcher.agent_env_dir(workspace_dir).mkdir(parents=True)

    env = launcher.build_agent_env(
        workspace_dir=workspace_dir,
        inherit_env={
            "PATH": "/usr/bin",
            "CONDA_PREFIX": "/conda",
            "CONDA_DEFAULT_ENV": "habitat-llm",
        },
    )

    assert "CONDA_PREFIX" not in env
    assert "CONDA_DEFAULT_ENV" not in env
    assert env["PATH"].split(":")[0] == str(workspace_dir / "bin")
    assert env["VIRTUAL_ENV"] == str(launcher.agent_env_dir(workspace_dir))


def test_ensure_agent_environment_only_creates_sandbox_env(monkeypatch, tmp_path):
    launcher = ExternalAgentLauncher(tmp_path)
    calls = []
    workspace_dir = tmp_path / "workspace"

    def fake_run_bootstrap(cmd, description):
        calls.append((cmd, description))
        env_dir = launcher.agent_env_dir(workspace_dir)
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "bin").mkdir(exist_ok=True)
        (env_dir / "bin" / "python").write_text("")

    monkeypatch.setattr(launcher, "_run_bootstrap", fake_run_bootstrap)
    launcher.ensure_agent_environment(workspace_dir)

    assert len(calls) == 1
    assert calls[0][1] == "create task-gen agent environment"
    assert str(launcher.agent_env_dir(workspace_dir)) in calls[0][0]


def test_build_external_prompt_rewrites_taskgen_commands():
    prompt = build_external_taskgen_prompt(
        working_dir="/repo/tmp/task_gen/run",
        task_file="/repo/tmp/task_gen/run/working_task.json",
        category="cooperative",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=4,
        current_k_level=2,
    )

    assert "`taskgen new_scene N`" in prompt
    assert "`taskgen judge`" in prompt
    assert "`taskgen submit_task`" in prompt
    assert "`taskgen finish`" in prompt
    assert "judge[]" not in prompt
    assert "submit_task[]" not in prompt
    assert "sampled_trajectories" not in prompt
    assert "## Good ToM" in prompt
    assert "## Required K-Level: 2" in prompt


def test_bootstrap_prompt_contains_full_taskgen_prompt(tmp_path):
    prompt = "full task instructions"

    _write_bootstrap_files(
        working_dir=tmp_path,
        prompt_text=prompt,
        authoring_constraints="scene objects only",
        available_mechanics="room_restriction",
        available_predicates="is_open",
        action_descriptions="Navigate",
    )

    assert (tmp_path / "bootstrap_prompt.txt").read_text() == prompt
    assert (tmp_path / "taskgen_prompt.md").read_text() == prompt


def test_sampled_task_aliases_preserve_original_text(tmp_path):
    source = tmp_path / "seed.json"
    original = {
        "task": "Keep the original task description.",
        "agent_secrets": {"agent_0": ["Original secret text."]},
    }
    source.write_text(json.dumps(original))

    _copy_sample(source, tmp_path, 1)

    copied = json.loads((tmp_path / "task_1.json").read_text())
    assert copied == original


def test_build_workspace_id_starts_with_timestamp():
    workspace_id = build_workspace_id("mini", now=datetime(2026, 3, 20, 16, 30, 45))

    assert workspace_id.startswith("2026-03-20_16-30-45-mini-")


def test_build_run_manifest_update_preserves_launcher_owned_fields():
    existing = {
        "run_id": "2026-04-08_08-23-16-generation",
        "started_at": "2026-04-08T08:23:16",
        "mode": "bulk",
        "total_workers": 24,
        "requested_tasks": 50,
        "output_dir": "data/enacttom/tasks",
        "task_gen_agent": "mini",
        "model": "gpt-5.2",
    }

    updated = _build_run_manifest_update(
        existing,
        run_id="wrong-run-id",
        generation_mode="worker-local-mode",
        generation_total_workers=1,
        generation_requested_tasks=1,
        output_dir="wrong/output",
        task_gen_agent="codex",
        model="wrong-model",
    )

    assert updated == existing


def test_taskgen_session_finish_and_fail(tmp_path):
    state = default_state(
        working_dir=str(tmp_path),
        output_dir="data/enacttom/tasks",
        num_tasks_target=2,
        agents_min=2,
        agents_max=2,
        subtasks_min=2,
        subtasks_max=4,
        category=None,
        seed_tasks_dir=None,
        seed_pass_ratio=0.2,
        seed_fail_ratio=0.8,
        judge_threshold=None,
        difficulty=None,
        test_model=None,
        calibration_stats={"model": "gpt-5.2", "target_rate": 0.20},
        calibration_tasks_dirs=[],
        task_gen_agent="mini",
        allowed_k_levels=[2],
    )
    with open(tmp_path / "taskgen_state.json", "w") as f:
        json.dump(state, f, indent=2)

    session = TaskGenSession(str(tmp_path))
    incomplete = session.finish()
    assert incomplete["success"] is False

    session.state["submitted_tasks"] = ["a.json", "b.json"]
    session._write_state()
    complete = session.finish()
    assert complete["success"] is True

    failed = session.fail("boom")
    assert failed["success"] is True

    saved = json.loads((tmp_path / "taskgen_state.json").read_text())
    assert saved["finished"] is True
    assert saved["failed"] is True
    assert saved["fail_reason"] == "boom"
