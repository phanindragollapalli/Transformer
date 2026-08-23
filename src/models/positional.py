"""Positional encoding modules implemented from scratch: sinusoidal and RoPE."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine absolute positional encodings added to embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The buffer is registered as non-persistent so it follows the module to
    any device without being part of saved checkpoints.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        division_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * division_term)
        encoding[:, 1::2] = torch.cos(position * division_term)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings for positions [0, seq_len) to x."""
        sequence_length = x.size(1)
        if sequence_length > self.encoding.size(0):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum positional length "
                f"{self.encoding.size(0)}"
            )
        return x + self.encoding[:sequence_length].to(dtype=x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate feature pairs by swapping halves and negating the second half."""
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) applied to query/key head tensors.

    RoPE rotates each pair of adjacent (half-split) features of queries and
    keys by position-dependent angles, so that attention scores depend only
    on relative offsets between positions.
    """

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        inv_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_frequency)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def apply_to_pair(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings to shaped head tensors.

        Args:
            query: [batch, n_heads, q_len, head_dim]
            key: [batch, n_heads, k_len, head_dim]

        Returns:
            Rotated (query, key) with identical shapes.
        """
        q_len = query.size(-2)
        k_len = key.size(-2)
        cos_q = torch.cat(
            [self.cos_cached[:q_len], self.cos_cached[:q_len]], dim=-1
        ).view(1, 1, q_len, self.head_dim).to(dtype=query.dtype)
        sin_q = torch.cat(
            [self.sin_cached[:q_len], self.sin_cached[:q_len]], dim=-1
        ).view(1, 1, q_len, self.head_dim).to(dtype=query.dtype)
        cos_k = torch.cat(
            [self.cos_cached[:k_len], self.cos_cached[:k_len]], dim=-1
        ).view(1, 1, k_len, self.head_dim).to(dtype=key.dtype)
        sin_k = torch.cat(
            [self.sin_cached[:k_len], self.sin_cached[:k_len]], dim=-1
        ).view(1, 1, k_len, self.head_dim).to(dtype=key.dtype)
        rotated_query = query * cos_q + rotate_half(query) * sin_q
        rotated_key = key * cos_k + rotate_half(key) * sin_k
        return rotated_query, rotated_key


def build_positional_encoding(position_encoding_type: str, d_model: int, max_len: int) -> nn.Module | None:
    """Create a positional encoding module from a config string.

    Returns None for 'rope' because rotary embeddings are applied inside the
    attention modules rather than added to the input embeddings.
    """
    if position_encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    if position_encoding_type == "rope":
        return None
    raise ValueError(
        f"Unknown position_encoding_type {position_encoding_type!r}; expected 'sinusoidal' or 'rope'"
    )
