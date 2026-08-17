from __future__ import annotations

from src.metrics import compute_metrics
import sys
from types import SimpleNamespace

from src.tracking import NullRun, init_wandb


def test_metrics_are_computed_over_complete_case_records():
    records = [
        {
            "gold_p": [1, 0],
            "pred_p": [1, 1],
            "gold_d": [0],
            "pred_d": [0],
            "gold_tort": 1,
            "pred_tort": 1,
        },
        {
            "gold_p": [],
            "pred_p": [],
            "gold_d": [1],
            "pred_d": [0],
            "gold_tort": 0,
            "pred_tort": 1,
        },
    ]
    metrics = compute_metrics(records)
    assert metrics["official_re_f1"] == 1.0 / 3.0
    assert metrics["tort_accuracy"] == 0.5
    assert metrics["selection_score"] == (1.0 / 3.0 + 0.5) / 2.0


def test_disabled_wandb_does_not_import_dependency():
    config = {
        "logging": {"wandb": {"enabled": False}},
        "_config_path": "/tmp/configs/base.yaml",
    }
    assert isinstance(init_wandb(config), NullRun)


def test_enabled_wandb_receives_public_config_without_secret(monkeypatch, tmp_path):
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return NullRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda *args, **kwargs: True)
    )
    config = {
        "experiment_name": "tracked",
        "logging": {
            "wandb": {
                "enabled": True,
                "project": "legal-mtl",
                "entity": None,
                "run_name": None,
                "mode": "offline",
            }
        },
        "model": {"model_dim": 16},
        "_config_path": str(tmp_path / "configs" / "base.yaml"),
    }
    init_wandb(config)
    assert captured["project"] == "legal-mtl"
    assert captured["name"] == "tracked"
    assert "_config_path" not in captured["config"]
    assert "WANDB_API_KEY" not in str(captured["config"])
