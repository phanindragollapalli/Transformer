"""Simplified Byte Latent Transformer (BLT) modules for configuration C5.

The simplified BLT pipeline replaces only the tokenization strategy of the
baseline model; every other architectural choice matches C1 (sinusoidal
positional encoding, multi-head attention, LayerNorm, identical depth and
width):

1. Local byte encoder: raw bytes are grouped into fixed-size patches and
   compressed into one latent vector per patch using patch-local attention.
2. Global latent transformer: the standard pre-norm encoder-decoder backbone
   processes patch latents on the source side and byte-level autoregressive
   states on the target side with cross-attention to the latents.
3. Local byte decoder: a causal local-window attention refinement stage over
   the global decoder states produces the final hidden states used to predict
   raw byte ids, so no learned subword vocabulary is involved anywhere.

Causality note: the global decoder is fully causal at the byte level and the
local decoder uses a banded-causal mask, so no future byte can influence an
earlier prediction at any stage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.attention import build_attention
from src.models.norm import LayerNorm
from src.models.positional import SinusoidalPositionalEncoding
from src.models.transformer import (
    EncoderBlock,
    TransformerDecoder,
    TransformerEncoder,
    combine_causal_and_padding,
)


class LocalEncoder(nn.Module):
    """Compresses raw byte sequences into fixed-size patch representations.

    Bytes are embedded, reshaped into non-overlapping patches of
    ``patch_size`` positions, refined by patch-local pre-norm attention
    blocks (the reshape guarantees no information crosses patch boundaries),
    and mean-pooled over valid byte positions into one latent per patch.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        patch_size: int,
        n_local_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.patch_size = patch_size
        self.pad_id = pad_id
        self.embedding_scale = float(d_model) ** 0.5

        def attention_factory() -> nn.Module:
            return build_attention(
                attention_type="mha",
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                attention_dropout=0.0,
                rotary=None,
            )

        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        self.local_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    attention_module=attention_factory(),
                    normalization_module=LayerNorm(d_model),
                    d_model=d_model,
                    d_ff=d_model * 2,
                    dropout=dropout,
                )
                for _ in range(n_local_layers)
            ]
        )
        self.final_norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=d_model**-0.5)

    def forward(
        self, byte_ids: Tensor, byte_padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode padded raw-byte ids into patch latents.

        Args:
            byte_ids: [batch, src_len] byte token ids.
            byte_padding_mask: [batch, src_len] True at valid byte positions.

        Returns:
            Tuple of (patch_latents [batch, n_patches, d_model],
            patch_padding_mask [batch, n_patches]).
        """
        batch_size, sequence_length = byte_ids.shape
        padding_needed = (-sequence_length) % self.patch_size
        if padding_needed:
            byte_ids = F.pad(byte_ids, (0, padding_needed), value=self.pad_id)
            byte_padding_mask = F.pad(byte_padding_mask, (0, padding_needed), value=False)

        n_patches = byte_ids.size(1) // self.patch_size
        valid = byte_padding_mask.view(batch_size, n_patches, self.patch_size)
        key_mask = valid.reshape(batch_size * n_patches, 1, 1, self.patch_size)

        x = self.byte_embedding(byte_ids) * self.embedding_scale
        x = x.view(batch_size * n_patches, self.patch_size, -1)
        for block in self.local_blocks:
            x = block(x, key_mask)
        x = self.final_norm(x)

        valid_float = valid.reshape(batch_size * n_patches, self.patch_size, 1).to(x.dtype)
        pooled = (x * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp(min=1.0)
        latents = self.output_projection(pooled).view(batch_size, n_patches, -1)
        patch_padding_mask = valid.any(dim=-1)
        return latents, patch_padding_mask


def build_banded_causal_mask(sequence_length: int, window_size: int, device: torch.device) -> Tensor:
    """Build a causal mask limited to a local window of past positions."""
    base = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
    causal = torch.tril(base)
    local_window = torch.triu(base, diagonal=-(window_size - 1))
    return causal & local_window


class LocalDecoder(nn.Module):
    """Causal local-window refinement over global decoder states.

    Each position attends only to itself and the previous
    ``window_size - 1`` positions, mirroring BLT's local decoder that turns
    globally contextualized latents into final byte-level hidden states.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = window_size

        def attention_factory() -> nn.Module:
            return build_attention(
                attention_type="mha",
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                attention_dropout=0.0,
                rotary=None,
            )

        self.local_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    attention_module=attention_factory(),
                    normalization_module=LayerNorm(d_model),
                    d_model=d_model,
                    d_ff=d_model * 2,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        sequence_length = hidden_states.size(1)
        if sequence_length == 0:
            return hidden_states
        mask = build_banded_causal_mask(
            sequence_length, self.window_size, hidden_states.device
        )
        for block in self.local_blocks:
            hidden_states = block(hidden_states, mask)
        return hidden_states


class ByteLatentTransformer(nn.Module):
    """Configuration C5 model: token-free byte pipeline with BLT-style stages.

    Exposes the same ``encode``/``decode``/``project`` interface as
    :class:`EncoderDecoderTransformer` so training and greedy decoding are
    shared between all configurations.
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
        patch_size: int = 16,
        n_local_encoder_layers: int = 1,
        n_local_decoder_layers: int = 1,
        max_position_length: int = 4096,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding_scale = float(d_model) ** 0.5

        def attention_factory() -> nn.Module:
            return build_attention(
                attention_type="mha",
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                attention_dropout=0.0,
                rotary=None,
            )

        def normalization_factory() -> nn.Module:
            return LayerNorm(d_model)

        self.local_encoder = LocalEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            patch_size=patch_size,
            n_local_layers=n_local_encoder_layers,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.source_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_position_length
        )
        self.global_encoder = TransformerEncoder(
            n_layers=n_encoder_layers,
            attention_factory=attention_factory,
            normalization_factory=normalization_factory,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.target_byte_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.target_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_position_length
        )
        self.global_decoder = TransformerDecoder(
            n_layers=n_decoder_layers,
            attention_factory=attention_factory,
            normalization_factory=normalization_factory,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.local_decoder = LocalDecoder(
            d_model=d_model,
            n_heads=n_heads,
            window_size=patch_size,
            n_layers=n_local_decoder_layers,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.target_byte_embedding.weight, mean=0.0, std=self.d_model**-0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _key_padding_mask(padding_mask: Tensor) -> Tensor:
        return padding_mask[:, None, None, :]

    def encode(self, source_ids: Tensor, source_padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Compress source bytes into patches and run the global encoder."""
        latents, patch_padding_mask = self.local_encoder(source_ids, source_padding_mask)
        latents = self.source_positional_encoding(latents)
        memory = self.global_encoder(latents, self._key_padding_mask(patch_padding_mask))
        return memory, patch_padding_mask

    def decode(
        self,
        decoder_input_ids: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ) -> Tensor:
        """Run byte-level autoregressive decoding followed by local refinement."""
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
        x = self.target_byte_embedding(decoder_input_ids) * self.embedding_scale
        x = self.target_positional_encoding(x)
        x = self.embedding_dropout(x)
        x = self.global_decoder(
            x,
            memory,
            self_attention_mask=combined_self_mask,
            cross_attention_mask=self._key_padding_mask(memory_padding_mask),
        )
        return self.local_decoder(x)

    def project(self, hidden_states: Tensor) -> Tensor:
        return self.output_projection(hidden_states)

    def forward(
        self,
        source_ids: Tensor,
        decoder_input_ids: Tensor,
        source_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ) -> Tensor:
        memory, memory_padding_mask = self.encode(source_ids, source_padding_mask)
        hidden = self.decode(
            decoder_input_ids, memory, memory_padding_mask, target_padding_mask
        )
        return self.project(hidden)


__all__ = [
    "LocalEncoder",
    "LocalDecoder",
    "ByteLatentTransformer",
    "build_banded_causal_mask",
]
