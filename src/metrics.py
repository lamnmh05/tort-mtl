from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def positive_f1(gold: Sequence[int], predicted: Sequence[int]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    true_positive = sum(g == 1 and p == 1 for g, p in zip(gold, predicted))
    false_positive = sum(g == 0 and p == 1 for g, p in zip(gold, predicted))
    false_negative = sum(g == 1 and p == 0 for g, p in zip(gold, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def binary_accuracy(gold: Sequence[int], predicted: Sequence[int]) -> float | None:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    return None if not gold else sum(g == p for g, p in zip(gold, predicted)) / len(gold)


def compute_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    per_case_re_f1: list[float] = []
    all_gold_p: list[int] = []
    all_pred_p: list[int] = []
    all_gold_d: list[int] = []
    all_pred_d: list[int] = []
    gold_tort: list[int] = []
    pred_tort: list[int] = []

    for record in records:
        gold_re = list(record["gold_p"]) + list(record["gold_d"])
        pred_re = list(record["pred_p"]) + list(record["pred_d"])
        if gold_re:
            per_case_re_f1.append(positive_f1(gold_re, pred_re))
        all_gold_p.extend(record["gold_p"])
        all_pred_p.extend(record["pred_p"])
        all_gold_d.extend(record["gold_d"])
        all_pred_d.extend(record["pred_d"])
        gold_tort.append(int(record["gold_tort"]))
        pred_tort.append(int(record["pred_tort"]))

    official_re_f1 = (
        None if not per_case_re_f1 else sum(per_case_re_f1) / len(per_case_re_f1)
    )
    tort_accuracy = binary_accuracy(gold_tort, pred_tort)
    metrics: dict[str, float | None] = {
        "official_re_f1": official_re_f1,
        "tort_accuracy": tort_accuracy,
        "plaintiff_accuracy": binary_accuracy(all_gold_p, all_pred_p),
        "plaintiff_f1": None if not all_gold_p else positive_f1(all_gold_p, all_pred_p),
        "defendant_accuracy": binary_accuracy(all_gold_d, all_pred_d),
        "defendant_f1": None if not all_gold_d else positive_f1(all_gold_d, all_pred_d),
    }
    metrics["selection_score"] = (
        None
        if official_re_f1 is None or tort_accuracy is None
        else 0.5 * official_re_f1 + 0.5 * tort_accuracy
    )
    return metrics
