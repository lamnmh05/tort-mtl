from __future__ import annotations

from pathlib import Path

import torch


class FakeTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            values = [1 + (ord(character) % 63) for character in text][:max_length]
            if values:
                ids[row, : len(values)] = torch.tensor(values)
                mask[row, : len(values)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def save_pretrained(self, path):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "fake_tokenizer.txt").write_text("fake", encoding="utf-8")


def legal_case(case_id="1", *, empty=False):
    if empty:
        u, p, d = [], [], []
    else:
        u = [{"description": "fact"}]
        p = [{"description": "plaintiff", "is_accepted": True}]
        d = [{"description": "defendant", "is_accepted": False}]
    return {
        "version": "test",
        "tort_id": str(case_id),
        "undisputed_facts": u,
        "plaintiff_claims": p,
        "defendant_claims": d,
        "court_decision": True,
    }
