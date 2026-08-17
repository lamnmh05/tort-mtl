from __future__ import annotations

import torch

from src.data import LegalBatchCollator
from src.losses import compute_multitask_loss, masked_bce_with_logits
from src.model import LegalCrossAttentionMTL, LegalMTLConfig, SharedSpanEncoder, TinyEncoder

from conftest import FakeTokenizer, legal_case


def tiny_model() -> LegalCrossAttentionMTL:
    config = LegalMTLConfig(
        encoder_hidden_size=16,
        model_dim=16,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        max_spans=4,
        re_hidden_dim=8,
        tp_hidden_dim=16,
    )
    return LegalCrossAttentionMTL(TinyEncoder(vocab_size=128, hidden_size=16), config)


def test_forward_backward_with_empty_groups_and_entirely_empty_case():
    batch = LegalBatchCollator(FakeTokenizer(), 8, 4, 4, 4)(
        [legal_case("normal"), legal_case("empty", empty=True)]
    )
    model = tiny_model()
    outputs = model(**batch["inputs"])
    losses = compute_multitask_loss(
        outputs,
        batch["labels"],
        {"plaintiff": 1.0, "defendant": 1.0, "tort": 1.0},
    )
    losses["loss"].backward()

    assert outputs["tort_logits"].shape == (2,)
    assert torch.isfinite(outputs["tort_logits"]).all()
    assert torch.isfinite(losses["loss"])
    assert model.span_encoder.encoder.embedding.weight.grad is not None


def test_masked_loss_is_zero_for_an_empty_group():
    loss = masked_bce_with_logits(
        torch.tensor([[100.0, -100.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[False, False]]),
    )
    assert loss.item() == 0.0


def test_span_encoder_supports_bfloat16_autocast():
    config = LegalMTLConfig(
        encoder_hidden_size=16,
        model_dim=16,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    )
    encoder = SharedSpanEncoder(TinyEncoder(vocab_size=128, hidden_size=16), config)
    input_ids = torch.randint(0, 128, (2, 3, 8))
    attention_mask = torch.ones_like(input_ids)
    span_mask = torch.tensor([[True, False, True], [False, True, False]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(input_ids, attention_mask, span_mask)

    assert output.dtype == torch.bfloat16
    assert output.shape == (2, 3, 16)
    assert torch.isfinite(output).all()
