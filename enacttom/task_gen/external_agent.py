from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class ExternalAgentError(RuntimeError):
    pass


class ExternalAgentLauncher:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.base_tmp_dir = project_root / "tmp" / "task_gen"
        self.bootstrap_cache_dir = project_root / "tmp" / "uv-cache"

    def agent_env_dir(self, workspace_dir: Path) -> Path:
        return workspace_dir / ".venv"

    def agent_python(self, workspace_dir: Path) -> Path:
        return self.agent_env_dir(workspace_dir) / "bin" / "python"

    def mini_cli_env_dir(self, workspace_dir: Path) -> Path:
        return workspace_dir / ".mini-cli"

    def mini_cli_executable(self, workspace_dir: Path) -> Path:
        return self.mini_cli_env_dir(workspace_dir) / "bin" / "mini"

    def ensure_agent_environment(self, workspace_dir: Path) -> Path:
        self.base_tmp_dir.mkdir(parents=True, exist_ok=True)
        env_dir = self.agent_env_dir(workspace_dir)
        if not self.agent_python(workspace_dir).exists():
            cmd = ["uv", "venv", str(env_dir), "--python", sys.executable]
            self._run_bootstrap(cmd, "create task-gen agent environment")

        return env_dir

    def _run_bootstrap(self, cmd: List[str], description: str) -> None:
        env = dict(os.environ)
        self.bootstrap_cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("UV_CACHE_DIR", str(self.bootstrap_cache_dir))
        env.setdefault("XDG_CACHE_HOME", str(self.project_root / "tmp" / "xdg-cache"))
        try:
            subprocess.run(cmd, cwd=str(self.project_root), check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise ExternalAgentError(f"Failed to {description}: {exc}") from exc

    def build_agent_env(
        self,
        *,
        workspace_dir: Path,
        inherit_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        env_dir = self.agent_env_dir(workspace_dir)
        base_env = dict(inherit_env or os.environ)
        for key in [
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
            "CONDA_PROMPT_MODIFIER",
            "CONDA_SHLVL",
            "PYTHONHOME",
            "PYTHONPATH",
        ]:
            base_env.pop(key, None)

        env = dict(base_env)
        path_parts = [
            str(workspace_dir / "bin"),
            str(env_dir / "bin"),
            env.get("PATH", ""),
        ]
        env["PATH"] = os.pathsep.join(part for part in path_parts if part)
        env["VIRTUAL_ENV"] = str(env_dir)
        env["PAGER"] = "cat"
        env["MANPAGER"] = "cat"
        env["LESS"] = "-R"
        env["PIP_PROGRESS_BAR"] = "off"
        env["TQDM_DISABLE"] = "1"
        env["MSWEA_CONFIGURED"] = "1"
        return env

    def _normalize_mini_model(self, model: Optional[str]) -> Optional[str]:
        if not model or "/" in model:
            return model
        if model.startswith("gpt-") or model.startswith("o"):
            return f"openai/{model}"
        anthropic_aliases = {
            "sonnet": "claude-sonnet-4-6",
            "sonnet-4.6": "claude-sonnet-4-6",
            "sonnet4.6": "claude-sonnet-4-6",
            "sonnet-4.5": "claude-sonnet-4-5-20250929",
            "sonnet4.5": "claude-sonnet-4-5-20250929",
            "haiku": "claude-haiku-4-5-20251001",
            "haiku-4.5": "claude-haiku-4-5-20251001",
            "haiku4.5": "claude-haiku-4-5-20251001",
            "opus": "claude-opus-4-6",
            "opus-4.6": "claude-opus-4-6",
            "opus4.6": "claude-opus-4-6",
            "opus-4.5": "claude-opus-4-5-20251101",
            "opus4.5": "claude-opus-4-5-20251101",
        }
        if model in anthropic_aliases:
            return f"anthropic/{anthropic_aliases[model]}"
        if model.startswith("claude-"):
            return f"anthropic/{model}"
        return model

    def resolve_executable(self, agent_name: str, env: Dict[str, str]) -> str:
        executable_names = {
            "mini": "mini",
            "claude": "claude",
            "codex": "codex",
        }
        executable = executable_names[agent_name]
        resolved = shutil.which(executable, path=env.get("PATH"))
        if not resolved and agent_name == "mini":
            resolved = self._ensure_mini_cli(workspace_dir=Path(env["VIRTUAL_ENV"]).parent)
        if not resolved:
            raise ExternalAgentError(
                f"Could not find executable '{executable}' for task-gen agent '{agent_name}'. "
                f"For mini, either install mini-swe-agent in a Python 3.10+ operator environment "
                f"or let the launcher provision it with uv."
            )
        return resolved

    def _ensure_mini_cli(self, workspace_dir: Path) -> Optional[str]:
        mini_executable = self.mini_cli_executable(workspace_dir)
        if mini_executable.exists():
            return str(mini_executable)

        mini_env_dir = self.mini_cli_env_dir(workspace_dir)
        try:
            self._run_bootstrap(
                ["uv", "venv", "--python", "3.11", str(mini_env_dir)],
                "create mini CLI environment",
            )
            self._run_bootstrap(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(mini_env_dir / "bin" / "python"),
                    "mini-swe-agent",
                ],
                "install mini-swe-agent in mini CLI environment",
            )
        except ExternalAgentError:
            return None

        if mini_executable.exists():
            return str(mini_executable)
        return None

    def build_command(
        self,
        *,
        agent_name: str,
        executable: str,
        workspace_dir: Path,
        bootstrap_prompt: str,
        model: Optional[str] = None,
        trace_output_path: Optional[Path] = None,
    ) -> List[str]:
        if agent_name == "mini":
            model = self._normalize_mini_model(model)
            cmd = [
                executable,
                "-c",
                "mini.yaml",
                "-c",
                "environment.timeout=1200",
                "-y",
            ]
            if model:
                cmd.extend(["-m", model])
            cmd.extend(
                [
                    "--exit-immediately",
                    "-o",
                    str(trace_output_path or (workspace_dir / "mini_trajectory.json")),
                    "-t",
                    bootstrap_prompt,
                ]
            )
            return cmd
        if agent_name == "claude":
            cmd = [
                executable,
                "--bare",
                "--dangerously-skip-permissions",
                "--print",
                "--output-format",
                "text",
                "--add-dir",
                str(workspace_dir),
                "--tools",
                "Bash",
            ]
            if model:
                cmd.extend(["--model", model])
            cmd.append(bootstrap_prompt)
            return cmd
        if agent_name == "codex":
            cmd = [
                executable,
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--cd",
                str(workspace_dir),
            ]
            if model:
                cmd.extend(["--model", model])
            cmd.append(bootstrap_prompt)
            return cmd
        raise ExternalAgentError(f"Unsupported task-gen agent: {agent_name}")

    def run(
        self,
        *,
        agent_name: str,
        workspace_dir: Path,
        bootstrap_prompt: str,
        model: Optional[str] = None,
        trace_output_path: Optional[Path] = None,
    ) -> int:
        self.ensure_agent_environment(workspace_dir)
        env = self.build_agent_env(workspace_dir=workspace_dir)
        executable = self.resolve_executable(agent_name, env)
        cmd = self.build_command(
            agent_name=agent_name,
            executable=executable,
            workspace_dir=workspace_dir,
            bootstrap_prompt=bootstrap_prompt,
            model=model,
            trace_output_path=trace_output_path,
        )
        proc = subprocess.run(cmd, cwd=str(workspace_dir), env=env)
        return proc.returncode
