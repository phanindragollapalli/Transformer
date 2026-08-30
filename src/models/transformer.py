"""Baseline encoder-decoder transformer assembled from from-scratch modules.

This module contains the position-wise feed-forward network, pre-norm
encoder/decoder blocks, the stacked encoder-decoder model used by
configurations C1-C4, and the shared greedy decoding helper.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.attention import GroupedQueryAttention, MultiHeadAttention, build_attention
from src.models.norm import build_normalization
from src.models.positional import (
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
    build_positional_encoding,
)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up_projection = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.down_projection = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_projection(self.dropout(self.activation(self.up_projection(x))))


class EncoderBlock(nn.Module):
    """Pre-layer-norm encoder block: masked self-attention plus FFN residuals."""

    def __init__(
        self,
        attention_module: nn.Module,
        normalization_module: nn.Module,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = attention_module
        self.norm_self_attention = normalization_module
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm_feed_forward = normalization_module
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, self_attention_mask: Tensor | None) -> Tensor:
        attended = self.self_attention(
            self.norm_self_attention(x),
            self.norm_self_attention(x),
            self.norm_self_attention(x),
            mask=self_attention_mask,
        )
        x = x + self.residual_dropout(attended)
        x = x + self.residual_dropout(self.feed_forward(self.norm_feed_forward(x)))
        return x


class DecoderBlock(nn.Module):
    """Pre-layer-norm decoder block: causal self-attention, cross-attention, FFN."""

    def __init__(
        self,
        self_attention_module: nn.Module,
        cross_attention_module: nn.Module,
        normalization_module: nn.Module,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention_module
        self.cross_attention = cross_attention_module
        self.norm_self_attention = normalization_module
        self.norm_cross_attention = normalization_module
        self.norm_feed_forward = normalization_module
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attention_mask: Tensor | None,
        cross_attention_mask: Tensor | None,
    ) -> Tensor:
        normalized_x = self.norm_self_attention(x)
        attended = self.self_attention(normalized_x, normalized_x, normalized_x, mask=self_attention_mask)
        x = x + self.residual_dropout(attended)

        normalized_x = self.norm_cross_attention(x)
        cross_attended = self.cross_attention(normalized_x, memory, memory, mask=cross_attention_mask)
        x = x + self.residual_dropout(cross_attended)

        x = x + self.residual_dropout(self.feed_forward(self.norm_feed_forward(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of encoder blocks with a final normalization layer."""

    def __init__(
        self,
        n_layers: int,
        attention_factory,
        normalization_factory,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    attention_module=attention_factory(),
                    normalization_module=normalization_factory(),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = normalization_factory()

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    """Stack of decoder blocks with a final normalization layer."""

    def __init__(
        self,
        n_layers: int,
        attention_factory,
        normalization_factory,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    self_attention_module=attention_factory(),
                    cross_attention_module=attention_factory(),
                    normalization_module=normalization_factory(),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = normalization_factory()

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attention_mask: Tensor | None,
        cross_attention_mask: Tensor | None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, self_attention_mask, cross_attention_mask)
        return self.final_norm(x)


def combine_causal_and_padding(causal_mask: Tensor, padding_mask_4d: Tensor) -> Tensor:
    """AND a [q_len, k_len] causal mask with a [batch, 1, 1, k_len] padding mask."""
    return padding_mask_4d & causal_mask


@torch.no_grad()
def greedy_decode(
    model,
    source_ids: Tensor,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
) -> Tensor:
    """Greedy autoregressive decoding for any model exposing encode/decode/project.

    Args:
        model: encoder-decoder model with ``encode``, ``decode``, ``project``.
        source_ids: [batch, src_len] padded source token ids.
        max_length: maximum number of generated tokens including BOS.

    Returns:
        Token tensor of shape [batch, <= max_length]; padding after EOS.
    """
    model.eval()
    device = source_ids.device
    batch_size = source_ids.size(0)
    source_padding_mask = source_ids.ne(pad_id)
    memory, memory_padding_mask = model.encode(source_ids, source_padding_mask)

    sequences = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(max_length - 1):
        target_padding_mask = sequences.ne(pad_id)
        hidden = model.decode(sequences, memory, memory_padding_mask, target_padding_mask)
        logits = model.project(hidden[:, -1])
        next_tokens = logits.argmax(dim=-1)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)
        sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
        finished = finished | next_tokens.eq(eos_id)
        if bool(finished.all()):
            break
    return sequences


def trim_after_eos(sequences: Tensor, eos_id: int, pad_id: int) -> list[list[int]]:
    """Trim each decoded row at its first EOS token, dropping padding."""
    trimmed: list[list[int]] = []
    for row in sequences.tolist():
        tokens: list[int] = []
        for token_id in row:
            if token_id == eos_id:
                break
            if token_id == pad_id:
                continue
            tokens.append(token_id)
        trimmed.append(tokens)
    return trimmed


class EncoderDecoderTransformer(nn.Module):
    """Full encoder-decoder transformer for configurations C1-C4.

    The same class realizes all four configurations through its factory
    arguments; only one component differs between C1 and any of C2-C4:

    - positional_encoding='sinusoidal' | 'rope'
    - attention_type='mha' | 'gqa' (n_kv_heads used when 'gqa')
    - normalization_type='layernorm' | 'rmsnorm'

    With RoPE enabled no additive positional encoding is applied at the
    input embeddings; instead rotary embeddings are applied inside every
    attention module.
    """

    def __init__(
        self,
        *,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        dropout: float = 0.1,
        attention_type: str = "mha",
        normalization_type: str = "layernorm",
        positional_encoding: str = "sinusoidal",
        n_kv_heads: int | None = None,
        rope_base: float = 10000.0,
        max_position_length: int = 4096,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.positional_encoding_type = positional_encoding
        self.embedding_scale = float(d_model) ** 0.5

        rotary = None
        if positional_encoding == "rope":
            rotary = RotaryPositionalEmbedding(
                head_dim=d_model // n_heads, max_len=max_position_length, base=rope_base
            )

        def attention_factory() -> nn.Module:
            return build_attention(
                attention_type=attention_type,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                attention_dropout=0.0,
                rotary=rotary,
                n_kv_heads=n_kv_heads,
            )

        def normalization_factory() -> nn.Module:
            return build_normalization(normalization_type, d_model)

        self.source_embedding = nn.Embedding(src_vocab_size, d_model)
        self.target_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.positional_encoding: SinusoidalPositionalEncoding | None = (
            build_positional_encoding(positional_encoding, d_model, max_position_length)
        )

        self.encoder = TransformerEncoder(
            n_layers=n_encoder_layers,
            attention_factory=attention_factory,
            normalization_factory=normalization_factory,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            n_layers=n_decoder_layers,
            attention_factory=attention_factory,
            normalization_factory=normalization_factory,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.source_embedding.weight, mean=0.0, std=self.d_model**-0.5)
        nn.init.normal_(self.target_embedding.weight, mean=0.0, std=self.d_model**-0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _embed_source(self, source_ids: Tensor) -> Tensor:
        x = self.source_embedding(source_ids) * self.embedding_scale
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        return self.embedding_dropout(x)

    def _embed_target(self, decoder_input_ids: Tensor) -> Tensor:
        x = self.target_embedding(decoder_input_ids) * self.embedding_scale
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        return self.embedding_dropout(x)

    @staticmethod
    def _key_padding_mask(padding_mask: Tensor) -> Tensor:
        return padding_mask[:, None, None, :]

    def encode(self, source_ids: Tensor, source_padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode source token ids into contextual memory states.

        Returns:
            Tuple of (memory, memory_padding_mask). The memory padding mask is
            returned separately because compressed architectures such as the
            BLT variant operate on a different number of positions than the
            raw input tokens.
        """
        embedded = self._embed_source(source_ids)
        return self.encoder(embedded, self._key_padding_mask(source_padding_mask)), source_padding_mask

    def decode(
        self,
        decoder_input_ids: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ) -> Tensor:
        """Run the decoder over shifted-right target ids against encoder memory."""
        sequence_length = decoder_input_ids.size(1)
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=decoder_input_ids.device,
            )
        )
        combined_self_mask = combine_causal_and_padding(
            causal_mask, self._key_padding_mask(target_padding_mask)
        )
        embedded = self._embed_target(decoder_input_ids)
        return self.decoder(
            embedded,
            memory,
            self_attention_mask=combined_self_mask,
            cross_attention_mask=self._key_padding_mask(memory_padding_mask),
        )

    def project(self, hidden_states: Tensor) -> Tensor:
        """Project decoder hidden states to vocabulary logits."""
        return self.output_projection(hidden_states)

    def forward(
        self,
        source_ids: Tensor,
        decoder_input_ids: Tensor,
        source_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ) -> Tensor:
        """Teacher-forced forward pass returning vocabulary logits."""
        memory, memory_padding_mask = self.encode(source_ids, source_padding_mask)
        hidden = self.decode(
            decoder_input_ids, memory, memory_padding_mask, target_padding_mask
        )
        return self.project(hidden)


def build_model(config, *, src_vocab_size: int, tgt_vocab_size: int) -> nn.Module:
    """Build the model for an experiment configuration.

    C1-C4 share :class:`EncoderDecoderTransformer` with exactly one component
    swapped per configuration; C5 uses the token-free
    :class:`ByteLatentTransformer`. The BLT backbone intentionally keeps every
    baseline hyperparameter so comparisons isolate the tokenization change.
    """
    if config.tokenization_type == "blt":
        from src.models.blt import ByteLatentTransformer

        return ByteLatentTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config.embedding_dim,
            n_heads=config.attention_heads,
            d_ff=config.ff_hidden_dim,
            n_encoder_layers=config.encoder_layers,
            n_decoder_layers=config.decoder_layers,
            dropout=config.dropout,
            patch_size=config.blt_patch_size,
            max_position_length=max(config.max_source_length, config.max_target_length) + 8,
        )
    return EncoderDecoderTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.embedding_dim,
        n_heads=config.attention_heads,
        d_ff=config.ff_hidden_dim,
        n_encoder_layers=config.encoder_layers,
        n_decoder_layers=config.decoder_layers,
        dropout=config.dropout,
        attention_type=config.attention_type,
        normalization_type=config.normalization_type,
        positional_encoding=config.positional_encoding,
        n_kv_heads=config.query_groups,
        max_position_length=max(config.max_source_length, config.max_target_length) + 8,
    )


__all__ = [
    "FeedForward",
    "EncoderBlock",
    "DecoderBlock",
    "TransformerEncoder",
    "TransformerDecoder",
    "EncoderDecoderTransformer",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "greedy_decode",
    "trim_after_eos",
    "build_model",
]
