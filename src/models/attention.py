"""Attention modules implemented from scratch: scaled dot-product, MHA, and GQA."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.positional import RotaryPositionalEmbedding


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional boolean masking.

    Computes softmax(Q K^T / sqrt(d_k)) V. Masks are boolean tensors where
    True marks positions that may be attended and False marks blocked ones.
    """

    def __init__(self, attention_dropout: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {attention_dropout}")
        self.attention_dropout = attention_dropout

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None):
        """Apply attention to shaped head tensors.

        Args:
            query: [batch, n_heads, q_len, head_dim]
            key: [batch, n_heads, k_len, head_dim]
            value: [batch, n_heads, k_len, head_dim]
            mask: broadcastable boolean mask, True = attend

        Returns:
            Tuple of (output, attention_weights).
        """
        if mask is not None and mask.dtype != torch.bool:
            raise ValueError("Attention mask must be a boolean tensor with True = attend")
        head_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = torch.matmul(weights, value)
        return output, weights


def prepare_attention_mask(mask: Tensor | None) -> Tensor | None:
    """Normalize padding/causal masks to 4D form [batch, 1, q_len, k_len]."""
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise ValueError("Attention mask must be a boolean tensor with True = attend")
    if mask.dim() == 2:
        return mask[:, None, None, :]
    if mask.dim() == 3:
        return mask[:, None, :, :]
    if mask.dim() == 4:
        return mask
    raise ValueError(f"Unsupported mask dimensionality: {mask.dim()}")


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    """Reshape [batch, seq_len, d_model] into [batch, n_heads, seq_len, head_dim]."""
    batch_size, sequence_length, _ = x.shape
    return x.view(batch_size, sequence_length, n_heads, -1).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    """Reshape [batch, n_heads, seq_len, head_dim] back to [batch, seq_len, d_model]."""
    batch_size, n_heads, sequence_length, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, n_heads * head_dim)


class MultiHeadAttention(nn.Module):
    """Multi-head attention built from manual projections.

    When a RotaryPositionalEmbedding is supplied (configuration C2), rotary
    embeddings are applied to the projected queries and keys per head instead
    of adding absolute positional encodings at the model input.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rotary: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(attention_dropout=attention_dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute multi-head attention.

        Args:
            query: [batch, q_len, d_model]
            key: [batch, k_len, d_model]
            value: [batch, k_len, d_model]
            mask: boolean mask broadcastable to [batch, n_heads, q_len, k_len]

        Returns:
            Output tensor of shape [batch, q_len, d_model].
        """
        batch_size = query.size(0)
        q = _split_heads(self.q_proj(query), self.n_heads)
        k = _split_heads(self.k_proj(key), self.n_heads)
        v = _split_heads(self.v_proj(value), self.n_heads)
        if self.rotary is not None:
            q, k = self.rotary.apply_to_pair(q, k)
        attended, _weights = self.attention(q, k, v, prepare_attention_mask(mask))
        merged = _merge_heads(attended)
        return self.out_dropout(self.out_proj(merged))


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention: multiple query heads share each key/value head.

    With ``n_heads`` query heads and ``n_kv_heads`` key/value heads, every
    ``n_heads / n_kv_heads`` consecutive query heads reuse one KV head,
    shrinking the KV projection parameters and cache while keeping the
    interface identical to :class:`MultiHeadAttention`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rotary: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads} "
                "with n_kv_heads >= 1"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model)
        self.attention = ScaledDotProductAttention(attention_dropout=attention_dropout)
        self.out_dropout = nn.Dropout(dropout)

    @staticmethod
    def _expand_kv(x: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads so they line up one-to-one with the query heads."""
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
        """Compute grouped-query attention; shapes match MultiHeadAttention."""
        q = _split_heads(self.q_proj(query), self.n_heads)
        k = _split_heads(self.k_proj(key), self.n_kv_heads)
        v = _split_heads(self.v_proj(value), self.n_kv_heads)
        if self.rotary is not None:
            q, k = self.rotary.apply_to_pair(q, k)
        k = self._expand_kv(k, self.n_rep)
        v = self._expand_kv(v, self.n_rep)
        attended, _weights = self.attention(q, k, v, prepare_attention_mask(mask))
        merged = _merge_heads(attended)
        return self.out_dropout(self.out_proj(merged))


def build_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    dropout: float,
    attention_dropout: float,
    rotary: RotaryPositionalEmbedding | None = None,
    n_kv_heads: int | None = None,
) -> nn.Module:
    """Create an attention module from a config string."""
    if attention_type == "mha":
        return MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            rotary=rotary,
        )
    if attention_type == "gqa":
        groups = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 2)
        return GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=groups,
            dropout=dropout,
            attention_dropout=attention_dropout,
            rotary=rotary,
        )
    raise ValueError(f"Unknown attention_type {attention_type!r}; expected 'mha' or 'gqa'")
