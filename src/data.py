from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


GROUP_KEYS = {
    "u": "undisputed_facts",
    "p": "plaintiff_claims",
    "d": "defendant_claims",
}


def _validate_case(case: Any, identity: str) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError(f"Expected an object at {identity}")
    for key in ("tort_id", *GROUP_KEYS.values()):
        if key not in case:
            raise ValueError(f"Missing {key!r} at {identity}")
    for key in GROUP_KEYS.values():
        if not isinstance(case[key], list):
            raise ValueError(f"{key!r} must be a list at {identity}")
        for index, span in enumerate(case[key]):
            if not isinstance(span, (str, Mapping)):
                raise ValueError(f"Invalid span {key}[{index}] at {identity}")
            if key != "undisputed_facts" and (
                not isinstance(span, Mapping) or "is_accepted" not in span
            ):
                raise ValueError(f"Missing is_accepted in {key}[{index}] at {identity}")
            if key != "undisputed_facts" and type(span["is_accepted"]) is not bool:
                raise ValueError(f"is_accepted must be boolean in {key}[{index}] at {identity}")
    if "court_decision" not in case:
        raise ValueError(f"Missing 'court_decision' at {identity}")
    if type(case["court_decision"]) is not bool:
        raise ValueError(f"court_decision must be boolean at {identity}")
    return case


def load_jsonl_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc.msg}") from exc
            cases.append(_validate_case(item, f"{source}:{line_number}"))
    if not cases:
        raise ValueError(f"No cases found in {source}")
    return cases


class JsonlLegalDataset(Dataset[dict[str, Any]]):
    def __init__(self, path_or_cases: str | Path | Sequence[Mapping[str, Any]]) -> None:
        if isinstance(path_or_cases, (str, Path)):
            self.cases = load_jsonl_cases(path_or_cases)
        else:
            self.cases = [
                _validate_case(dict(case), f"dataset index {index}")
                for index, case in enumerate(path_or_cases)
            ]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.cases[index]


def _extract_text(span: Any) -> str:
    if isinstance(span, str):
        return span
    for key in ("description", "text", "content"):
        if key in span:
            return str(span[key])
    raise ValueError("A span mapping must contain description, text, or content")


class LegalBatchCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        max_u_spans: int | None,
        max_p_spans: int | None,
        max_d_spans: int | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.limits = {"u": max_u_spans, "p": max_p_spans, "d": max_d_spans}

    def _tokenize_group(self, cases: Sequence[Mapping[str, Any]], group: str) -> dict[str, Tensor]:
        key = GROUP_KEYS[group]
        limit = self.limits[group]
        spans_by_case = [list(case[key])[:limit] if limit is not None else list(case[key]) for case in cases]
        width = max(1, max((len(spans) for spans in spans_by_case), default=0))
        texts: list[str] = []
        span_mask = torch.zeros(len(cases), width, dtype=torch.bool)
        for row, spans in enumerate(spans_by_case):
            for column in range(width):
                if column < len(spans):
                    texts.append(_extract_text(spans[column]))
                    span_mask[row, column] = True
                else:
                    texts.append("")
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].reshape(len(cases), width, self.max_length),
            "attention_mask": tokens["attention_mask"].reshape(len(cases), width, self.max_length),
            "span_mask": span_mask,
        }

    def __call__(self, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not cases:
            raise ValueError("Cannot collate an empty batch")
        inputs: dict[str, Tensor] = {}
        groups: dict[str, dict[str, Tensor]] = {}
        for group in GROUP_KEYS:
            groups[group] = self._tokenize_group(cases, group)
            for name, tensor in groups[group].items():
                inputs[f"{group}_{name}"] = tensor

        labels = {
            "plaintiff": torch.zeros_like(groups["p"]["span_mask"], dtype=torch.float32),
            "defendant": torch.zeros_like(groups["d"]["span_mask"], dtype=torch.float32),
            "tort": torch.tensor([bool(case["court_decision"]) for case in cases], dtype=torch.float32),
        }
        for row, case in enumerate(cases):
            for column, claim in enumerate(case["plaintiff_claims"][: self.limits["p"]]):
                labels["plaintiff"][row, column] = float(bool(claim["is_accepted"]))
            for column, claim in enumerate(case["defendant_claims"][: self.limits["d"]]):
                labels["defendant"][row, column] = float(bool(claim["is_accepted"]))

        metadata = [
            {
                "version": str(case.get("version", "")),
                "tort_id": str(case["tort_id"]),
                "num_u": int(groups["u"]["span_mask"][row].sum()),
                "num_p": int(groups["p"]["span_mask"][row].sum()),
                "num_d": int(groups["d"]["span_mask"][row].sum()),
            }
            for row, case in enumerate(cases)
        ]
        return {"inputs": inputs, "labels": labels, "metadata": metadata}
