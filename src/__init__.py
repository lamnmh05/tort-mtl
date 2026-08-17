"""Legal multi-task learning package."""

from .data import JsonlLegalDataset, LegalBatchCollator, load_jsonl_cases
from .losses import compute_multitask_loss, masked_bce_with_logits
from .model import LegalCrossAttentionMTL, LegalMTLConfig

__all__ = [
    "JsonlLegalDataset",
    "LegalBatchCollator",
    "LegalCrossAttentionMTL",
    "LegalMTLConfig",
    "compute_multitask_loss",
    "load_jsonl_cases",
    "masked_bce_with_logits",
]
