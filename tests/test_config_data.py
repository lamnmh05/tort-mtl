from __future__ import annotations

import json

import pytest

from src.config import load_config
from src.data import JsonlLegalDataset, LegalBatchCollator, load_jsonl_cases

from conftest import FakeTokenizer, legal_case


def test_base_config_is_valid_and_explicit():
    config = load_config("configs/base.yaml")
    assert config["model"]["freeze_encoder"] is False
    assert config["logging"]["wandb"]["enabled"] is False
    assert config["training"]["encoder_learning_rate"] == pytest.approx(2e-5)


def test_loader_skips_blank_lines_and_validates_labels(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(legal_case(), ensure_ascii=False) + "\n\n", encoding="utf-8"
    )
    assert len(load_jsonl_cases(path)) == 1

    invalid = legal_case()
    invalid["court_decision"] = "true"
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="court_decision must be boolean"):
        load_jsonl_cases(path)


def test_collator_preserves_masks_labels_metadata_and_clips():
    first = legal_case("a")
    first["plaintiff_claims"].append(
        {"description": "clipped", "is_accepted": False}
    )
    dataset = JsonlLegalDataset([first, legal_case("b", empty=True)])
    collator = LegalBatchCollator(FakeTokenizer(), 8, 2, 1, 2)
    batch = collator([dataset[0], dataset[1]])

    assert batch["inputs"]["p_input_ids"].shape == (2, 1, 8)
    assert batch["inputs"]["u_span_mask"].tolist() == [[True], [False]]
    assert batch["labels"]["plaintiff"].tolist() == [[1.0], [0.0]]
    assert batch["metadata"][0]["num_p"] == 1
    assert batch["metadata"][1]["num_p"] == 0
