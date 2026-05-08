from __future__ import annotations

from omegaconf import OmegaConf

from enacttom.examples.run_habitat_benchmark import ensure_benchmark_observation_config


def test_ensure_benchmark_observation_config_forces_private_partial_obs() -> None:
    config = OmegaConf.create(
        {
            "world_model": {"partial_obs": False},
            "agent_asymmetry": False,
        }
    )

    ensure_benchmark_observation_config(config)

    # Standard mode (default) gets private per-agent world graphs.
    assert config.world_model.partial_obs is True
    assert config.agent_asymmetry is True
    assert config.benchmark_observation_mode == "text"
    assert config.benchmark_run_mode == "standard"
    assert config.benchmark_vision.selector_prompt_name == "enacttom_frame_selector"


def test_baseline_gets_shared_world_graph() -> None:
    config = OmegaConf.create(
        {
            "world_model": {"partial_obs": True},
            "agent_asymmetry": True,
            "benchmark_run_mode": "baseline",
        }
    )

    ensure_benchmark_observation_config(config)

    # Baseline mode uses the full-observability benchmark setting.
    assert config.world_model.partial_obs is False
    assert config.agent_asymmetry is False
