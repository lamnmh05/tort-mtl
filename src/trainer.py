from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import load_training_state, save_checkpoint
from .losses import compute_multitask_loss
from .metrics import compute_metrics


def _progress(iterable: Any, description: str, enabled: bool = True):
    return tqdm(iterable, total=len(iterable), desc=description, disable=not enabled, dynamic_ncols=True)


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def _grad_scaler(enabled: bool):
    grad_scaler = getattr(torch.amp, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover - old PyTorch


def _move_to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in values.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_predictions(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    threshold: float,
    bf16: bool,
    progress_enabled: bool = True,
    description: str = "Evaluate",
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    total_loss = 0.0
    total_examples = 0
    with _progress(loader, description, progress_enabled) as progress:
        for batch in progress:
            inputs = _move_to_device(batch["inputs"], device)
            labels = _move_to_device(batch["labels"], device)
            with _autocast(device, bf16):
                outputs = model(**inputs)
                losses = compute_multitask_loss(outputs, labels, weights)
            batch_size = labels["tort"].shape[0]
            total_loss += float(losses["loss"].detach()) * batch_size
            total_examples += batch_size

            probabilities = {
                "p": torch.sigmoid(outputs["plaintiff_logits"].float()).cpu(),
                "d": torch.sigmoid(outputs["defendant_logits"].float()).cpu(),
                "tort": torch.sigmoid(outputs["tort_logits"].float()).cpu(),
            }
            cpu_labels = {key: value.cpu() for key, value in labels.items()}
            for row, metadata in enumerate(batch["metadata"]):
                num_p, num_d = metadata["num_p"], metadata["num_d"]
                prob_p = probabilities["p"][row, :num_p].tolist()
                prob_d = probabilities["d"][row, :num_d].tolist()
                prob_tort = float(probabilities["tort"][row])
                records.append(
                    {
                        **metadata,
                        "gold_p": cpu_labels["plaintiff"][row, :num_p].int().tolist(),
                        "prob_p": prob_p,
                        "pred_p": [int(value >= threshold) for value in prob_p],
                        "gold_d": cpu_labels["defendant"][row, :num_d].int().tolist(),
                        "prob_d": prob_d,
                        "pred_d": [int(value >= threshold) for value in prob_d],
                        "gold_tort": int(cpu_labels["tort"][row]),
                        "prob_tort": prob_tort,
                        "pred_tort": int(prob_tort >= threshold),
                    }
                )

    metrics = compute_metrics(records)
    metrics["loss"] = None if not total_examples else total_loss / total_examples
    return metrics, records


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
    wandb_run: Any,
    resume_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    weights = {key: float(value) for key, value in config["loss"].items()}
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["num_train_epochs"])
    patience = int(training["early_stopping_patience"])
    threshold = float(config["evaluation"]["threshold"])
    progress_enabled = bool(config["logging"].get("tqdm", True))
    bf16 = bool(training["bf16"]) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = _grad_scaler(bf16)

    start_epoch = 0
    global_step = 0
    best_score = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if resume_dir is not None:
        state = load_training_state(
            resume_dir, model, optimizer, scheduler, scaler, device
        )
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_score = float(state["best_score"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state["history"])

    model.to(device)
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        example_count = 0
        window_batches = 0
        with _progress(
            train_loader, f"Train epoch {epoch + 1}/{epochs}", progress_enabled
        ) as progress:
            for batch_index, batch in enumerate(progress):
                inputs = _move_to_device(batch["inputs"], device)
                labels = _move_to_device(batch["labels"], device)
                with _autocast(device, bf16):
                    outputs = model(**inputs)
                    losses = compute_multitask_loss(outputs, labels, weights)
                scaler.scale(losses["loss"]).backward()
                window_batches += 1
                batch_size = labels["tort"].shape[0]
                epoch_loss += float(losses["loss"].detach()) * batch_size
                example_count += batch_size

                should_step = window_batches == accumulation or batch_index + 1 == len(train_loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    if window_batches > 1:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(window_batches)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    window_batches = 0
                    global_step += 1
                    current_loss = epoch_loss / max(example_count, 1)
                    learning_rate = float(scheduler.get_last_lr()[0])
                    progress.set_postfix(loss=f"{current_loss:.4f}", step=global_step, lr=f"{learning_rate:.3g}")
                    wandb_run.log(
                        {"train/loss": current_loss, "train/learning_rate": learning_rate, "train/epoch": epoch + 1},
                        step=global_step,
                    )

        train_loss = epoch_loss / max(example_count, 1)
        validation_metrics, predictions = evaluate_model(
            model,
            dev_loader,
            device,
            weights,
            threshold,
            bf16,
            progress_enabled,
            f"Validation epoch {epoch + 1}/{epochs}",
        )
        score_value = validation_metrics[config["evaluation"]["selection_metric"]]
        score = -math.inf if score_value is None else float(score_value)
        improved = math.isfinite(score) and score > best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation": validation_metrics,
            "best_score": best_score,
        }
        history.append(epoch_record)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        save_checkpoint(
            output_dir / "last", model, optimizer, scheduler, scaler, tokenizer, config, state
        )
        if improved:
            save_checkpoint(
                output_dir / "best", model, optimizer, scheduler, scaler, tokenizer, config, state
            )
            write_predictions(output_dir / "best" / "dev_predictions.jsonl", predictions)
        write_json(output_dir / "history.json", history)
        wandb_run.log(
            {
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                **{f"validation/{key}": value for key, value in validation_metrics.items() if value is not None},
                "validation/best_score": best_score,
            },
            step=global_step,
        )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f} selection_score={score:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs", flush=True)
            break
    return {"history": history, "best_score": best_score, "global_step": global_step}
