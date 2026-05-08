from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import enacttom.scripts.benchmark_repeat as benchmark_repeat


def _repeat_args(**overrides):
    defaults = {
        "tasks_dir": "data/enacttom/tasks",
        "task": None,
        "model": "gpt-5.4",
        "output_dir": "outputs/enacttom/test",
        "num_times": 3,
        "max_sim_steps": 200000,
        "max_llm_calls": None,
        "max_workers": 8,
        "num_gpus": None,
        "category": None,
        "observation_mode": "text",
        "benchmark_run_mode": "standard",
        "selector_min_frames": 1,
        "selector_max_frames": 5,
        "selector_max_candidates": 12,
        "video": False,
        "no_calibration": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_launch_run_redirects_child_output_to_per_run_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_output_dir = tmp_path / "run_1"
    run_output_dir.mkdir(parents=True)
    args = _repeat_args()
    captured = {}

    class DummyProc:
        def wait(self) -> int:
            return 0

    def fake_popen(cmd, cwd, stdout=None, stderr=None, text=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        return DummyProc()

    monkeypatch.setattr(benchmark_repeat.subprocess, "Popen", fake_popen)

    active_run = benchmark_repeat._launch_run(args, 1, run_output_dir)

    assert active_run.log_path == run_output_dir / "benchmark.log"
    assert Path(captured["stdout"].name) == active_run.log_path
    assert captured["stderr"] is benchmark_repeat.subprocess.STDOUT
    assert captured["text"] is True
    active_run.log_handle.close()
