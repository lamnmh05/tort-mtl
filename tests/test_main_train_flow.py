from __future__ import annotations

import argparse

import src.main as main_module


def test_run_train_evaluates_best_checkpoint_on_test(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    best_dir = output_dir / "best"

    config = {
        "output_dir": str(output_dir),
        "data": {
            "train_path": "train.jsonl",
            "dev_path": "dev.jsonl",
            "test_path": "test.jsonl",
        },
        "system": {"seed": 1, "deterministic": False, "device": "cpu"},
        "training": {
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "warmup_ratio": 0.0,
        },
    }
    calls = {"load": [], "evaluate": []}

    class DummyRun:
        def finish(self):
            pass

    class DummyModel:
        def to(self, device):
            return self

        def parameters(self):
            return iter(())

    monkeypatch.setattr(main_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        main_module,
        "resolve_project_path",
        lambda config, path: output_dir if path == str(output_dir) else tmp_path / path,
    )
    monkeypatch.setattr(main_module, "set_seed", lambda *args: None)
    monkeypatch.setattr(main_module, "resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(main_module, "build_tokenizer", lambda config: object())
    monkeypatch.setattr(main_module, "build_model", lambda config: DummyModel())
    monkeypatch.setattr(main_module, "save_resolved_config", lambda *args: None)
    monkeypatch.setattr(main_module, "write_json", lambda *args: None)
    monkeypatch.setattr(main_module, "build_loader", lambda *args: [object()])
    monkeypatch.setattr(main_module, "build_optimizer", lambda *args: object())
    monkeypatch.setattr(main_module, "build_scheduler", lambda *args: object())
    monkeypatch.setattr(main_module, "init_wandb", lambda config: DummyRun())

    def fake_train_model(*args):
        best_dir.mkdir(parents=True)
        (best_dir / "model.pt").touch()
        return {}

    monkeypatch.setattr(main_module, "train_model", fake_train_model)
    monkeypatch.setattr(
        main_module,
        "load_model_state",
        lambda checkpoint, model, device: calls["load"].append(checkpoint),
    )
    monkeypatch.setattr(
        main_module,
        "evaluate_split",
        lambda model, tokenizer, config, device, checkpoint, split: calls[
            "evaluate"
        ].append((checkpoint, split)),
    )

    main_module.run_train(argparse.Namespace(config="base.yaml", resume=None))

    assert calls["load"] == [best_dir]
    assert calls["evaluate"] == [(best_dir, "test")]
