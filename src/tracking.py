from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import public_config


class NullRun:
    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        del data, step

    def finish(self) -> None:
        pass


def init_wandb(config: dict[str, Any]) -> Any:
    settings = config["logging"]["wandb"]
    if not bool(settings.get("enabled", False)):
        return NullRun()

    try:
        from dotenv import load_dotenv
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled; install wandb and python-dotenv first"
        ) from exc

    project_root = Path(config["_config_path"]).parent.parent
    load_dotenv(project_root / ".env", override=False)
    return wandb.init(
        project=settings["project"],
        entity=settings.get("entity") or None,
        name=settings.get("run_name") or config["experiment_name"],
        mode=settings.get("mode", "online"),
        config=public_config(config),
        dir=str(project_root),
    )
