from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_model_state
from .config import load_config, resolve_project_path, save_resolved_config
from .data import JsonlLegalDataset, LegalBatchCollator
from .losses import compute_multitask_loss
from .model import LegalCrossAttentionMTL, LegalMTLConfig, TinyEncoder
from .tracking import init_wandb
from .trainer import evaluate_model, train_model, write_json, write_predictions


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("system.device is cuda but CUDA is unavailable")
    return torch.device(requested)


def build_model(config: dict[str, Any]) -> LegalCrossAttentionMTL:
    settings = config["model"]
    model = LegalCrossAttentionMTL.from_pretrained(
        settings["name"],
        revision=settings.get("revision"),
        model_dim=int(settings["model_dim"]),
        num_heads=int(settings["num_heads"]),
        ff_dim=int(settings["ff_dim"]),
        dropout=float(settings["dropout"]),
        max_spans=int(settings["max_spans"]),
        pooling=settings["pooling"],
        re_hidden_dim=int(settings["re_hidden_dim"]),
        tp_hidden_dim=int(settings["tp_hidden_dim"]),
        freeze_encoder=bool(settings["freeze_encoder"]),
    )
    config["model"]["encoder_hidden_size"] = model.config.encoder_hidden_size
    return model


def build_tokenizer(config: dict[str, Any], checkpoint: Path | None = None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to create the tokenizer") from exc
    saved = None if checkpoint is None else checkpoint / "tokenizer"
    source = saved if saved is not None and saved.exists() else config["model"].get(
        "tokenizer_name", config["model"]["name"]
    )
    kwargs: dict[str, Any] = {"use_fast": bool(config["model"].get("use_fast_tokenizer", True))}
    if saved is None or not saved.exists():
        revision = config["model"].get("revision")
        if revision:
            kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(source, **kwargs)


def build_collator(config: dict[str, Any], tokenizer: Any) -> LegalBatchCollator:
    data = config["data"]
    return LegalBatchCollator(
        tokenizer,
        int(data["max_length"]),
        data.get("max_u_spans"),
        data.get("max_p_spans"),
        data.get("max_d_spans"),
    )


def build_loader(
    config: dict[str, Any], path: Path, tokenizer: Any, train: bool
) -> DataLoader:
    training = config["training"]
    system = config["system"]
    return DataLoader(
        JsonlLegalDataset(path),
        batch_size=int(
            training["per_device_train_batch_size"]
            if train
            else training["per_device_eval_batch_size"]
        ),
        shuffle=train,
        num_workers=int(system["num_workers"]),
        pin_memory=bool(system["pin_memory"]) and torch.cuda.is_available(),
        collate_fn=build_collator(config, tokenizer),
    )


def build_optimizer(model: LegalCrossAttentionMTL, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder_parameters = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids and parameter.requires_grad]
    task_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids and parameter.requires_grad]
    groups = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": float(training["encoder_learning_rate"]), "name": "encoder"})
    groups.append({"params": task_parameters, "lr": float(training["task_learning_rate"]), "name": "task_modules"})
    return torch.optim.AdamW(
        groups,
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def run_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = resolve_project_path(config, config["output_dir"])
    if args.resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --resume or choose another output_dir."
        )
    config["output_dir"] = str(output_dir)
    for key in ("train_path", "dev_path", "test_path"):
        config["data"][key] = str(resolve_project_path(config, config["data"][key]))
    set_seed(int(config["system"]["seed"]), bool(config["system"]["deterministic"]))
    device = resolve_device(config["system"]["device"])
    tokenizer = build_tokenizer(config)
    model = build_model(config)
    model.to(device)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "trainable_ratio": trainable_parameters / max(total_parameters, 1),
        },
    )
    print(
        f"device={device} total_parameters={total_parameters:,} "
        f"trainable_parameters={trainable_parameters:,}",
        flush=True,
    )

    train_loader = build_loader(
        config, Path(config["data"]["train_path"]), tokenizer, True
    )
    dev_loader = build_loader(
        config, Path(config["data"]["dev_path"]), tokenizer, False
    )
    optimizer = build_optimizer(model, config)
    steps_per_epoch = math.ceil(
        len(train_loader) / int(config["training"]["gradient_accumulation_steps"])
    )
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch * int(config["training"]["num_train_epochs"]),
        float(config["training"]["warmup_ratio"]),
    )
    resume_dir = None if args.resume is None else Path(args.resume).resolve()
    wandb_run = init_wandb(config)
    try:
        result = train_model(
            model,
            train_loader,
            dev_loader,
            optimizer,
            scheduler,
            tokenizer,
            config,
            device,
            output_dir,
            wandb_run,
            resume_dir,
        )
        write_json(output_dir / "training_summary.json", result)
    finally:
        wandb_run.finish()

    best_checkpoint = output_dir / "best"
    if not (best_checkpoint / "model.pt").exists():
        raise RuntimeError(
            f"Training finished without a best checkpoint: {best_checkpoint}"
        )
    print(f"Evaluating best checkpoint on test split: {best_checkpoint}", flush=True)
    load_model_state(best_checkpoint, model, device)
    evaluate_split(model, tokenizer, config, device, best_checkpoint, "test")


def evaluate_split(
    model: LegalCrossAttentionMTL,
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
    checkpoint: Path,
    split: str,
) -> dict[str, float | None]:
    data_key = f"{split}_path"
    if data_key not in config["data"]:
        raise ValueError(f"Config has no data.{data_key}")
    loader = build_loader(
        config, resolve_project_path(config, config["data"][data_key]), tokenizer, False
    )
    bf16 = (
        bool(config["training"]["bf16"])
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    metrics, records = evaluate_model(
        model,
        loader,
        device,
        {key: float(value) for key, value in config["loss"].items()},
        float(config["evaluation"]["threshold"]),
        bf16,
        bool(config["logging"].get("tqdm", True)),
        f"Evaluate {split}",
    )
    write_json(checkpoint / f"{split}_metrics.json", metrics)
    write_predictions(checkpoint / f"{split}_predictions.jsonl", records)
    print(json_display(metrics))
    return metrics


def run_evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    config = load_config(checkpoint / "resolved_config.yaml")
    set_seed(int(config["system"]["seed"]), bool(config["system"]["deterministic"]))
    device = resolve_device(config["system"]["device"])
    tokenizer = build_tokenizer(config, checkpoint)
    model = build_model(config).to(device)
    load_model_state(checkpoint, model, device)
    evaluate_split(model, tokenizer, config, device, checkpoint, args.split)


def json_display(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def run_smoke_test() -> None:
    torch.manual_seed(7)
    config = LegalMTLConfig(
        encoder_hidden_size=32,
        model_dim=32,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        max_spans=4,
        re_hidden_dim=16,
        tp_hidden_dim=32,
    )
    model = LegalCrossAttentionMTL(TinyEncoder(hidden_size=32), config)
    batch_size, length = 2, 6
    inputs: dict[str, torch.Tensor] = {}
    for group, mask in {
        "u": torch.tensor([[1, 0], [0, 0]], dtype=torch.bool),
        "p": torch.tensor([[1, 1], [0, 0]], dtype=torch.bool),
        "d": torch.tensor([[1, 0], [0, 0]], dtype=torch.bool),
    }.items():
        inputs[f"{group}_input_ids"] = torch.randint(1, 128, (batch_size, 2, length))
        inputs[f"{group}_attention_mask"] = mask[:, :, None].expand(-1, -1, length).long()
        inputs[f"{group}_span_mask"] = mask
    outputs = model(**inputs)
    losses = compute_multitask_loss(
        outputs,
        {
            "plaintiff": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            "defendant": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "tort": torch.tensor([1.0, 0.0]),
        },
        {"plaintiff": 1.0, "defendant": 1.0, "tort": 1.0},
    )
    losses["loss"].backward()
    assert outputs["plaintiff_logits"].shape == (2, 2)
    assert outputs["defendant_logits"].shape == (2, 2)
    assert outputs["tort_logits"].shape == (2,)
    assert torch.isfinite(losses["loss"])
    print(f"Smoke test passed; loss={float(losses['loss'].detach()):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legal cross-attention multi-task learning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train and validate a model")
    train.add_argument("--config", required=True)
    train.add_argument("--resume")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("train", "dev", "test"), default="test")
    subparsers.add_parser("smoke-test", help="Run a download-free forward/backward test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_smoke_test()


if __name__ == "__main__":
    main()
