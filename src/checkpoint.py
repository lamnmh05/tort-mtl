from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .config import save_resolved_config


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    tokenizer: Any,
    config: dict[str, Any],
    trainer_state: dict[str, Any],
) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model.pt")
    state = dict(trainer_state)
    state.update(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _rng_state(),
        }
    )
    torch.save(state, destination / "trainer_state.pt")
    save_resolved_config(config, destination / "resolved_config.yaml")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(destination / "tokenizer")


def load_model_state(directory: str | Path, model: torch.nn.Module, device: torch.device) -> None:
    state = torch.load(Path(directory) / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)


def load_training_state(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
) -> dict[str, Any]:
    load_model_state(directory, model, device)
    state = torch.load(
        Path(directory) / "trainer_state.pt", map_location=device, weights_only=False
    )
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler.load_state_dict(state.pop("scheduler"))
    scaler.load_state_dict(state.pop("scaler"))
    _restore_rng_state(state.pop("rng"))
    return state
