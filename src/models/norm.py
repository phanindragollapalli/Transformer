"""Normalization layers implemented from scratch: LayerNorm and RMSNorm."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Manual LayerNorm over the last dimension.

    Normalizes each feature vector to zero mean and unit variance, then
    applies a learned per-feature scale and shift:

        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError(f"normalized_shape must be positive, got {normalized_shape}")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


class RMSNorm(nn.Module):
    """Root Mean Square normalization with a learned scale.

    Unlike LayerNorm, no mean subtraction and no bias are used:

        y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError(f"normalized_shape must be positive, got {normalized_shape}")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return x / torch.sqrt(mean_square + self.eps) * self.weight

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


def build_normalization(normalization_type: str, d_model: int, eps: float | None = None) -> nn.Module:
    """Create a normalization layer from a config string."""
    if normalization_type == "layernorm":
        kwargs = {} if eps is None else {"eps": eps}
        return LayerNorm(d_model, **kwargs)
    if normalization_type == "rmsnorm":
        kwargs = {} if eps is None else {"eps": eps}
        return RMSNorm(d_model, **kwargs)
    raise ValueError(f"Unknown normalization_type {normalization_type!r}; expected 'layernorm' or 'rmsnorm'")
