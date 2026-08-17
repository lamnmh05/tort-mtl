from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def masked_bce_with_logits(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    element_loss = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    mask_f = mask.to(element_loss.dtype)
    return (element_loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def compute_multitask_loss(
    outputs: Mapping[str, Any],
    labels: Mapping[str, Tensor],
    weights: Mapping[str, float],
) -> dict[str, Tensor]:
    loss_p = masked_bce_with_logits(
        outputs["plaintiff_logits"], labels["plaintiff"], outputs["masks"]["p"]
    )
    loss_d = masked_bce_with_logits(
        outputs["defendant_logits"], labels["defendant"], outputs["masks"]["d"]
    )
    loss_tort = F.binary_cross_entropy_with_logits(
        outputs["tort_logits"], labels["tort"].to(outputs["tort_logits"].dtype)
    )
    total = (
        float(weights["plaintiff"]) * loss_p
        + float(weights["defendant"]) * loss_d
        + float(weights["tort"]) * loss_tort
    )
    return {
        "loss": total,
        "loss_re_plaintiff": loss_p,
        "loss_re_defendant": loss_d,
        "loss_tort": loss_tort,
    }
