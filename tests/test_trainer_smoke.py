from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data import JsonlLegalDataset, LegalBatchCollator
from src.main import build_optimizer, build_scheduler
from src.model import LegalCrossAttentionMTL, LegalMTLConfig, TinyEncoder
from src.tracking import NullRun
from src.trainer import evaluate_model, train_model
import src.trainer as trainer_module

from conftest import FakeTokenizer, legal_case


def _config(tmp_path):
    return {
        "experiment_name": "smoke",
        "output_dir": str(tmp_path / "output"),
        "model": {"encoder_hidden_size": 16},
        "data": {},
        "loss": {"plaintiff": 1.0, "defendant": 1.0, "tort": 1.0},
        "training": {
            "optimizer": "adamw",
            "scheduler": "linear",
            "num_train_epochs": 1,
            "gradient_accumulation_steps": 2,
            "encoder_learning_rate": 1e-3,
            "task_learning_rate": 1e-3,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "bf16": False,
            "early_stopping_patience": 2,
        },
        "evaluation": {"threshold": 0.5, "selection_metric": "selection_score"},
        "logging": {"tqdm": False, "wandb": {"enabled": False}},
        "system": {},
        "_config_path": str(tmp_path / "configs" / "base.yaml"),
    }


def _model():
    return LegalCrossAttentionMTL(
        TinyEncoder(vocab_size=128, hidden_size=16),
        LegalMTLConfig(16, 16, 4, 32, 0.0, 4, "cls", 8, 16),
    )


def test_train_checkpoint_and_evaluate_roundtrip(tmp_path):
    tokenizer = FakeTokenizer()
    dataset = JsonlLegalDataset([legal_case("0"), legal_case("1", empty=True)])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=LegalBatchCollator(tokenizer, 8, 4, 4, 4),
    )
    config = _config(tmp_path)
    model = _model()
    before = model.span_encoder.projection.weight.detach().clone()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, 1, 0.0)
    result = train_model(
        model,
        loader,
        loader,
        optimizer,
        scheduler,
        tokenizer,
        config,
        torch.device("cpu"),
        config["output_dir"],
        NullRun(),
    )

    assert result["global_step"] == 1
    assert not torch.equal(before, model.span_encoder.projection.weight)
    assert (tmp_path / "output" / "last" / "model.pt").exists()
    assert (tmp_path / "output" / "best" / "model.pt").exists()
    metrics, records = evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        config["loss"],
        0.5,
        False,
        False,
    )
    assert metrics["loss"] is not None
    assert len(records) == 2

    resumed_config = _config(tmp_path)
    resumed_config["training"]["num_train_epochs"] = 2
    resumed_model = _model()
    resumed_optimizer = build_optimizer(resumed_model, resumed_config)
    resumed_scheduler = build_scheduler(resumed_optimizer, 2, 0.0)
    resumed = train_model(
        resumed_model,
        loader,
        loader,
        resumed_optimizer,
        resumed_scheduler,
        tokenizer,
        resumed_config,
        torch.device("cpu"),
        resumed_config["output_dir"],
        NullRun(),
        tmp_path / "output" / "last",
    )
    assert resumed["global_step"] == 2
    assert len(resumed["history"]) == 2


def test_progress_wrapper_configures_tqdm(monkeypatch):
    captured = {}

    class FakeProgress:
        def __init__(self, iterable, **kwargs):
            captured.update(kwargs)
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

    monkeypatch.setattr(trainer_module, "tqdm", FakeProgress)
    progress = trainer_module._progress([1, 2], "Train", enabled=True)
    assert list(progress) == [1, 2]
    assert captured == {
        "total": 2,
        "desc": "Train",
        "disable": False,
        "dynamic_ncols": True,
    }
