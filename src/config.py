from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate YAML, resolving paths relative to the config file."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {config_path}")

    base_ref = config.pop("_base_", None)
    if base_ref is not None:
        config = deep_merge(load_config(config_path.parent / str(base_ref)), config)
        config.pop("_config_path", None)

    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "experiment_name",
        "output_dir",
        "model",
        "data",
        "loss",
        "training",
        "evaluation",
        "logging",
        "system",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError("Missing config sections: " + ", ".join(missing))

    model = config["model"]
    positive_model_values = (
        "model_dim",
        "num_heads",
        "ff_dim",
        "max_spans",
        "re_hidden_dim",
        "tp_hidden_dim",
    )
    for key in positive_model_values:
        if int(model.get(key, 0)) <= 0:
            raise ValueError(f"model.{key} must be positive")
    if int(model["model_dim"]) % int(model["num_heads"]) != 0:
        raise ValueError("model.model_dim must be divisible by model.num_heads")
    if model.get("pooling") not in {"cls", "mean"}:
        raise ValueError("model.pooling must be 'cls' or 'mean'")
    if not 0.0 <= float(model.get("dropout", -1.0)) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    data = config["data"]
    if int(data.get("max_length", 0)) <= 0:
        raise ValueError("data.max_length must be positive")
    for key in ("max_u_spans", "max_p_spans", "max_d_spans"):
        value = data.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"data.{key} must be positive or null")
        if value is not None and int(value) > int(model["max_spans"]):
            raise ValueError(f"data.{key} cannot exceed model.max_spans")

    training = config["training"]
    positive_training_values = (
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "encoder_learning_rate",
        "task_learning_rate",
        "max_grad_norm",
        "early_stopping_patience",
    )
    for key in positive_training_values:
        if float(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if not 0.0 <= float(training.get("warmup_ratio", -1.0)) <= 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1]")
    if training.get("optimizer") != "adamw":
        raise ValueError("training.optimizer must be 'adamw'")
    if training.get("scheduler") != "linear":
        raise ValueError("training.scheduler must be 'linear'")
    for key in ("adam_beta1", "adam_beta2"):
        if not 0.0 <= float(training.get(key, -1.0)) < 1.0:
            raise ValueError(f"training.{key} must be in [0, 1)")
    if float(training.get("adam_epsilon", 0.0)) <= 0:
        raise ValueError("training.adam_epsilon must be positive")

    for name, weight in config["loss"].items():
        if float(weight) < 0:
            raise ValueError(f"loss.{name} must be non-negative")
    threshold = float(config["evaluation"].get("threshold", -1.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("evaluation.threshold must be in [0, 1]")
    if config["evaluation"].get("selection_metric") != "selection_score":
        raise ValueError("evaluation.selection_metric must be 'selection_score'")
    if config["logging"].get("wandb", {}).get("mode", "online") not in {
        "online",
        "offline",
        "disabled",
    }:
        raise ValueError("logging.wandb.mode must be online, offline, or disabled")
    # if config["system"].get("device", "auto") not in {"auto", "cuda", "cpu"}:
    #     raise ValueError("system.device must be auto, cuda, or cpu")


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return config fields safe to persist or send to experiment tracking."""
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve runtime paths relative to the directory containing the YAML file."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_config_path"]).parent.parent / path


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(public_config(config), handle, sort_keys=False, allow_unicode=True)
