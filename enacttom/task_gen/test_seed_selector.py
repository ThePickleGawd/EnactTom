import json
import random

from enacttom.task_gen.seed_selector import (
    SeedSelectionConfig,
    build_seed_candidates,
    select_seed_tasks,
)


def _write_task(path, title, calibration=None, category="cooperative", tom_level=2):
    data = {
        "title": title,
        "task": f"Task for {title}",
        "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
        "category": category,
        "tom_level": tom_level,
        "calibration": calibration or [],
    }
    path.write_text(json.dumps(data))


def _standard_calibration(model, passed, progress):
    return [
        {
            "model": model,
            "run_mode": "standard",
            "agent_models": {"agent_0": model, "agent_1": model},
            "results": {
                "passed": passed,
                "progress": progress,
            },
        }
    ]


def test_seed_selector_can_force_fail_bucket(tmp_path):
    model = "gpt-5.2"
    _write_task(tmp_path / "hard.json", "hard", _standard_calibration(model, False, 0.35))
    _write_task(tmp_path / "easy.json", "easy", _standard_calibration(model, True, 1.0))

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
        pass_seed_ratio=0.0,
        fail_seed_ratio=1.0,
    )
    selected = select_seed_tasks(config, count=1, rng=random.Random(0))

    assert selected[0].path.name == "hard.json"


def test_seed_selector_can_force_pass_bucket(tmp_path):
    model = "gpt-5.2"
    _write_task(tmp_path / "hard.json", "hard", _standard_calibration(model, False, 0.10))
    _write_task(tmp_path / "easy.json", "easy", _standard_calibration(model, True, 1.0))

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
        pass_seed_ratio=1.0,
        fail_seed_ratio=0.0,
    )
    selected = select_seed_tasks(config, count=1, rng=random.Random(0))

    assert selected[0].path.name == "easy.json"


def test_seed_selector_applies_category_and_tom_filters_as_soft_biases(tmp_path):
    model = "gpt-5.2"
    calibration = _standard_calibration(model, False, 0.25)
    _write_task(tmp_path / "match.json", "match", calibration, category="mixed", tom_level=3)
    _write_task(tmp_path / "mismatch.json", "mismatch", calibration, category="cooperative", tom_level=1)

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
        target_pass_rate=0.20,
        current_pass_rate=0.40,
        category="mixed",
        tom_level=3,
    )
    weights = {candidate.path.name: candidate.weight for candidate in build_seed_candidates(config)}

    assert weights["match.json"] > weights["mismatch.json"]


def test_seed_selector_falls_back_to_untested_when_no_calibrated_buckets_exist(tmp_path):
    model = "gpt-5.2"
    _write_task(tmp_path / "unknown_a.json", "unknown_a")
    _write_task(tmp_path / "unknown_b.json", "unknown_b")

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
        pass_seed_ratio=0.2,
        fail_seed_ratio=0.8,
    )
    selected = select_seed_tasks(config, count=1, rng=random.Random(0))

    assert selected[0].path.name in {"unknown_a.json", "unknown_b.json"}


def test_seed_selector_excludes_k_zero_tasks(tmp_path):
    model = "gpt-5.2"
    _write_task(tmp_path / "k0.json", "k0", tom_level=0)
    _write_task(tmp_path / "k1.json", "k1", tom_level=1)

    config = SeedSelectionConfig(
        tasks_dir=tmp_path,
        target_model=model,
    )
    selected = build_seed_candidates(config)

    assert [candidate.path.name for candidate in selected] == ["k1.json"]
