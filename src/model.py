from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


@dataclass
class LegalMTLConfig:
    encoder_hidden_size: int
    model_dim: int = 768
    num_heads: int = 12
    ff_dim: int = 2048
    dropout: float = 0.1
    max_spans: int = 128
    pooling: str = "cls"
    re_hidden_dim: int = 384
    tp_hidden_dim: int = 768
    freeze_encoder: bool = False

    def __post_init__(self) -> None:
        if self.encoder_hidden_size <= 0:
            raise ValueError("encoder_hidden_size must be positive")
        if self.model_dim <= 0 or self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        if self.max_spans < 1:
            raise ValueError("max_spans must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SharedSpanEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, config: LegalMTLConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = config.pooling
        self.projection = nn.Linear(config.encoder_hidden_size, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        if config.freeze_encoder:
            self.encoder.requires_grad_(False)

    def _pool(self, token_states: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling == "cls":
            return token_states[:, 0]
        return masked_mean(token_states, attention_mask.bool(), dim=1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, span_mask: Tensor) -> Tensor:
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [B, S, L]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if span_mask.shape != input_ids.shape[:2]:
            raise ValueError("span_mask must have shape [B, S]")

        batch_size, num_spans, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(-1, seq_len)
        flat_attention = attention_mask.reshape(-1, seq_len)
        valid_indices = span_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel():
            valid_ids = flat_ids.index_select(0, valid_indices)
            valid_attention = flat_attention.index_select(0, valid_indices)
            encoded = self.encoder(input_ids=valid_ids, attention_mask=valid_attention)
            token_states = (
                encoded.last_hidden_state
                if hasattr(encoded, "last_hidden_state")
                else encoded[0]
            )
            pooled = self.dropout(self.projection(self._pool(token_states, valid_attention)))
            # Autocast can produce bf16 activations while the projection weights
            # remain fp32. index_copy requires its destination and source to have
            # exactly the same dtype, so allocate from the computed activation.
            output = pooled.new_zeros(
                (batch_size * num_spans, self.projection.out_features)
            )
            output = output.index_copy(0, valid_indices, pooled)
        else:
            output = self.projection.weight.new_zeros(
                (batch_size * num_spans, self.projection.out_features)
            )
        return output.reshape(batch_size, num_spans, -1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.key_value_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, query: Tensor, key_value: Tensor, query_mask: Tensor, key_value_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        query_mask = query_mask.bool()
        key_value_mask = key_value_mask.bool()
        has_key = key_value_mask.any(dim=1)
        safe_key_value = key_value.clone()
        safe_key_mask = key_value_mask.clone()
        missing = ~has_key
        if missing.any():
            safe_key_value[missing, 0] = 0.0
            safe_key_mask[missing, 0] = True

        attended, weights = self.attention(
            query=self.query_norm(query),
            key=self.key_value_norm(safe_key_value),
            value=self.key_value_norm(safe_key_value),
            key_padding_mask=~safe_key_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = attended * has_key[:, None, None].to(attended.dtype)
        hidden = query + self.attention_dropout(attended)
        hidden = hidden + self.ffn(self.ff_norm(hidden))
        hidden = hidden * query_mask.unsqueeze(-1).to(hidden.dtype)
        weights = weights * has_key[:, None, None, None].to(weights.dtype)
        weights = weights * query_mask[:, None, :, None].to(weights.dtype)
        return hidden, weights


class ClassificationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.network(hidden)


class LegalCrossAttentionMTL(nn.Module):
    U_TYPE, P_TYPE, D_TYPE = 0, 1, 2

    def __init__(self, encoder: nn.Module, config: LegalMTLConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.span_encoder = SharedSpanEncoder(encoder, config)
        self.type_embeddings = nn.Embedding(3, dim)
        self.position_embeddings = nn.Embedding(config.max_spans, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.p_from_u = CrossAttentionBlock(dim, config.num_heads, config.ff_dim, config.dropout)
        self.d_from_u = CrossAttentionBlock(dim, config.num_heads, config.ff_dim, config.dropout)
        self.p_from_d = CrossAttentionBlock(dim, config.num_heads, config.ff_dim, config.dropout)
        self.d_from_p = CrossAttentionBlock(dim, config.num_heads, config.ff_dim, config.dropout)
        self.plaintiff_re_head = ClassificationMLP(dim, config.re_hidden_dim, config.dropout)
        self.defendant_re_head = ClassificationMLP(dim, config.re_hidden_dim, config.dropout)
        self.tort_head = ClassificationMLP(4 * dim, config.tp_hidden_dim, config.dropout)

    @property
    def encoder(self) -> nn.Module:
        return self.span_encoder.encoder

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> "LegalCrossAttentionMTL":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to load a pretrained encoder") from exc

        revision = kwargs.pop("revision", None)
        encoder_kwargs = kwargs.pop("encoder_kwargs", {})
        if revision:
            encoder_kwargs["revision"] = revision
        encoder = AutoModel.from_pretrained(model_name_or_path, **encoder_kwargs)
        hidden_size = int(encoder.config.hidden_size)
        kwargs["encoder_hidden_size"] = hidden_size
        if kwargs.get("model_dim") is None:
            kwargs["model_dim"] = hidden_size
        return cls(encoder, LegalMTLConfig(**kwargs))

    def _encode_group(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        span_mask: Tensor,
        type_id: int,
    ) -> Tensor:
        hidden = self.span_encoder(input_ids, attention_mask, span_mask)
        batch_size, num_spans, _ = hidden.shape
        if num_spans > self.config.max_spans:
            raise ValueError(f"num_spans={num_spans} exceeds max_spans={self.config.max_spans}")
        positions = torch.arange(num_spans, device=hidden.device).expand(batch_size, -1)
        types = torch.full_like(positions, type_id)
        hidden = hidden + self.position_embeddings(positions) + self.type_embeddings(types)
        hidden = self.input_dropout(self.input_norm(hidden))
        return hidden * span_mask.unsqueeze(-1).to(hidden.dtype)

    def forward(self, **inputs: Tensor) -> dict[str, Any]:
        masks = {name: inputs[f"{name}_span_mask"].bool() for name in ("u", "p", "d")}
        hidden = {
            name: self._encode_group(
                inputs[f"{name}_input_ids"],
                inputs[f"{name}_attention_mask"],
                masks[name],
                type_id,
            )
            for name, type_id in (("u", self.U_TYPE), ("p", self.P_TYPE), ("d", self.D_TYPE))
        }
        p_grounded, attn_pu = self.p_from_u(hidden["p"], hidden["u"], masks["p"], masks["u"])
        d_grounded, attn_du = self.d_from_u(hidden["d"], hidden["u"], masks["d"], masks["u"])
        p_final, attn_pd = self.p_from_d(p_grounded, d_grounded, masks["p"], masks["d"])
        d_final, attn_dp = self.d_from_p(d_grounded, p_grounded, masks["d"], masks["p"])

        plaintiff_logits = self.plaintiff_re_head(p_final).squeeze(-1)
        defendant_logits = self.defendant_re_head(d_final).squeeze(-1)
        all_hidden = torch.cat([hidden["u"], p_final, d_final], dim=1)
        all_mask = torch.cat([masks["u"], masks["p"], masks["d"]], dim=1)
        case_vector = torch.cat(
            [
                masked_mean(all_hidden, all_mask, 1),
                masked_mean(hidden["u"], masks["u"], 1),
                masked_mean(p_final, masks["p"], 1),
                masked_mean(d_final, masks["d"], 1),
            ],
            dim=-1,
        )
        tort_logits = self.tort_head(case_vector).squeeze(-1)
        return {
            "plaintiff_logits": plaintiff_logits,
            "defendant_logits": defendant_logits,
            "tort_logits": tort_logits,
            "masks": masks,
            "attention_maps": {
                "p_to_u": attn_pu,
                "d_to_u": attn_du,
                "p_to_d": attn_pd,
                "d_to_p": attn_dp,
            },
            "hidden_states": {"u_anchor": hidden["u"], "p_final": p_final, "d_final": d_final, "case_vector": case_vector},
        }


class TinyEncoder(nn.Module):
    """Download-free encoder for tests and the CLI smoke test."""

    def __init__(self, vocab_size: int = 128, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size, 4, 2 * hidden_size, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Any:
        hidden = self.transformer(
            self.embedding(input_ids), src_key_padding_mask=~attention_mask.bool()
        )
        return SimpleNamespace(last_hidden_state=hidden)
