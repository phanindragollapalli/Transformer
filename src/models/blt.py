"""Faithful fixed-patch Byte Latent Transformer (BLT) modules for configuration C5.

The BLT pipeline replaces the subword tokenization strategy of the baseline
model with a patch-latent token-free architecture; every other architectural
choice matches C1 (sinusoidal positional encoding, MHA, LayerNorm, identical
depth and width):

1. Source path:
   - Source cipher bits are grouped into bytes (handled in data loading).
   - Source bytes are embedded and split into fixed-size non-overlapping patches.
   - Lightweight local self-attention runs independently inside each patch.
   - Masked-pooling produces one source patch latent per patch.
   - Global encoder runs over source patch latents with sinusoidal positional encoding.

2. Target path:
   - Plaintext is encoded as raw UTF-8 byte values (PAD=0, BOS=1, EOS=2, UNK=3).
   - During training, model input is [BOS] + target_bytes; labels are target_bytes + [EOS].
   - Target inputs are temporarily padded to a multiple of patch_size for patch operations.
   - A separate target local byte encoder produces one latent per target patch.
   - A learned global BOS patch latent is prepended, and target patch latents are shifted right by 1:
     global_decoder_input = [global_bos_patch, target_patch_latents[:, :-1]]
   - Global decoder runs causally over this patch sequence with cross-attention to source patch latents.

3. Local byte decoder:
   - Byte-level autoregressive decoder taking the unpadded target byte sequence plus the
     global decoder's target-patch states.
   - Uses causal self-attention over target bytes.
   - Uses cross-attention from each byte position to permitted global patch states subject
     to the patch-shift causality:
     * predictions in the first output patch (t < patch_size) attend only to global BOS patch state;
     * predictions in later output patches attend only to global states derived from fully completed
       earlier target patches (j <= t // patch_size);
     * no predicted byte may access a patch representation containing itself or any future byte.
   - Output states are projected to the byte vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.attention import build_attention
from src.models.norm import LayerNorm
from src.models.positional import SinusoidalPositionalEncoding
from src.models.transformer import (
    DecoderBlock,
    EncoderBlock,
    TransformerDecoder,
    TransformerEncoder,
)


class LocalEncoder(nn.Module):
    """Compresses raw byte sequences into fixed-size patch representations.

    Bytes are embedded, reshaped into non-overlapping patches of
    ``patch_size`` positions, refined by patch-local pre-norm attention
    blocks (the reshape guarantees no information crosses patch boundaries),
    and masked-pooled over valid byte positions into one latent per patch.
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
        self.intra_patch_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=patch_size + 4
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
                for _ in range(n_local_layers)
            ]
        )
        self.final_norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=d_model**-0.5)

    def forward(
        self, byte_ids: Tensor, byte_padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode raw byte ids into patch latents.

        Args:
            byte_ids: [batch, seq_len] byte token ids.
            byte_padding_mask: [batch, seq_len] True at valid byte positions.

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
        patch_padding_mask = valid.any(dim=-1)
        key_mask = valid.reshape(batch_size * n_patches, 1, 1, self.patch_size)
        has_valid_keys = patch_padding_mask.view(batch_size * n_patches, 1, 1, 1)
        safe_key_mask = torch.where(has_valid_keys, key_mask, torch.ones_like(key_mask))

        x = self.byte_embedding(byte_ids) * self.embedding_scale
        x = x.view(batch_size * n_patches, self.patch_size, -1)
        x = self.intra_patch_positional_encoding(x)
        for block in self.local_blocks:
            x = block(x, safe_key_mask)
        x = self.final_norm(x)

        valid_float = valid.reshape(batch_size * n_patches, self.patch_size, 1).to(x.dtype)
        sum_valid = valid_float.sum(dim=1).clamp(min=1.0)
        pooled = (x * valid_float).sum(dim=1) / sum_valid
        latents = self.output_projection(pooled).view(batch_size, n_patches, -1)
        latents = torch.where(patch_padding_mask.unsqueeze(-1), latents, torch.zeros_like(latents))
        return latents, patch_padding_mask


class LocalByteDecoder(nn.Module):
    """Byte-level autoregressive decoder with shift-enforcing cross-attention.

    Inputs:
        byte_embeddings: [batch, seq_len, d_model] target byte sequence embeddings.
        global_patch_states: [batch, n_patches, d_model] global decoder patch representations.
        target_padding_mask: [batch, seq_len] True at valid target byte positions.
        global_patch_padding_mask: [batch, n_patches] True at valid global patch positions.

    Causality guarantee:
        Byte position t in patch p = t // patch_size attends only to global patch
        states j <= p. Because global patch state 0 represents the learned BOS patch
        and state j >= 1 represents target patch j - 1, no predicted byte accesses a
        patch representation containing itself or any future byte.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        patch_size: int,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

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

        self.blocks = nn.ModuleList(
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
        self.final_norm = LayerNorm(d_model)

    def forward(
        self,
        byte_embeddings: Tensor,
        global_patch_states: Tensor,
        target_padding_mask: Tensor,
        global_patch_padding_mask: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, _ = byte_embeddings.shape
        if sequence_length == 0:
            return byte_embeddings
        n_patches = global_patch_states.size(1)
        device = byte_embeddings.device

        # 1. Causal self-attention mask over target bytes: [batch, 1, seq_len, seq_len]
        byte_causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
        ).unsqueeze(0).unsqueeze(0)
        self_attn_mask = byte_causal_mask & target_padding_mask[:, None, None, :]
        has_valid_self = self_attn_mask.any(dim=-1, keepdim=True)
        safe_self_mask = torch.where(has_valid_self, self_attn_mask, torch.ones_like(self_attn_mask))

        # 2. Shift-enforcing cross-attention mask from byte position t to global patch state j:
        # Byte position t in patch p = t // patch_size may attend only to global patch states j <= p.
        t_indices = torch.arange(sequence_length, device=device)[:, None]
        j_indices = torch.arange(n_patches, device=device)[None, :]
        patch_causal_mask = (j_indices <= (t_indices // self.patch_size)).unsqueeze(0).unsqueeze(0)
        cross_attn_mask = patch_causal_mask & global_patch_padding_mask[:, None, None, :]
        has_valid_cross = cross_attn_mask.any(dim=-1, keepdim=True)
        safe_cross_mask = torch.where(has_valid_cross, cross_attn_mask, torch.ones_like(cross_attn_mask))

        x = byte_embeddings
        for block in self.blocks:
            x = block(
                x,
                global_patch_states,
                self_attention_mask=safe_self_mask,
                cross_attention_mask=safe_cross_mask,
            )
        return self.final_norm(x)


# Backward-compatibility alias
LocalDecoder = LocalByteDecoder


class ByteLatentTransformer(nn.Module):
    """Configuration C5 model: Fixed-patch Byte Latent Transformer (BLT) architecture.

    Exposes standard ``encode``, ``decode``, and ``project`` APIs matching
    :class:`EncoderDecoderTransformer` so training and greedy decoding remain
    shared across all configurations.
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

        def normalization_factory() -> nn.Module:
            return LayerNorm(d_model)

        # 1. Source local encoder & global encoder
        self.source_local_encoder = LocalEncoder(
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

        # 2. Target local encoder & global decoder
        self.target_local_encoder = LocalEncoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            patch_size=patch_size,
            n_local_layers=n_local_encoder_layers,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.global_bos_patch = nn.Parameter(torch.empty(1, 1, d_model))
        self.target_patch_positional_encoding = SinusoidalPositionalEncoding(
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

        # 3. Target byte embedding & Local byte decoder
        self.target_byte_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.target_byte_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=max_position_length
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.local_decoder = LocalByteDecoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            patch_size=patch_size,
            n_layers=n_local_decoder_layers,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.target_byte_embedding.weight, mean=0.0, std=self.d_model**-0.5)
        nn.init.normal_(self.global_bos_patch, mean=0.0, std=self.d_model**-0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, source_ids: Tensor, source_padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Compress source bytes into patches and run the global encoder.

        Args:
            source_ids: [batch, src_len] source byte token ids.
            source_padding_mask: [batch, src_len] True at valid positions.

        Returns:
            Tuple of (memory [batch, n_src_patches, d_model],
            patch_padding_mask [batch, n_src_patches]).
        """
        latents, patch_padding_mask = self.source_local_encoder(source_ids, source_padding_mask)
        latents = self.source_positional_encoding(latents)
        memory = self.global_encoder(latents, patch_padding_mask[:, None, None, :])
        return memory, patch_padding_mask

    def decode(
        self,
        decoder_input_ids: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ) -> Tensor:
        """Run target local patch encoding, global decoding, and local byte decoding.

        Args:
            decoder_input_ids: [batch, tgt_len] target byte token ids (e.g. [BOS] + bytes).
            memory: [batch, n_src_patches, d_model] source patch states from global encoder.
            memory_padding_mask: [batch, n_src_patches] True at valid source patch positions.
            target_padding_mask: [batch, tgt_len] True at valid target byte positions.

        Returns:
            Hidden states tensor of shape [batch, tgt_len, d_model].
        """
        batch_size, sequence_length = decoder_input_ids.shape
        device = decoder_input_ids.device

        # 1. Pad target inputs to a multiple of patch_size for local patch encoding
        padding_needed = (-sequence_length) % self.patch_size
        if padding_needed:
            padded_target_ids = F.pad(decoder_input_ids, (0, padding_needed), value=self.pad_id)
            padded_target_mask = F.pad(target_padding_mask, (0, padding_needed), value=False)
        else:
            padded_target_ids = decoder_input_ids
            padded_target_mask = target_padding_mask

        # 2. Run target local byte encoder to produce one latent per target patch
        target_patch_latents, target_patch_mask = self.target_local_encoder(
            padded_target_ids, padded_target_mask
        )

        # 3. Prepend learned global BOS patch latent and shift target patch-latents right by 1
        # global_decoder_input = [global_bos_patch, target_patch_latents[:, :-1]]
        bos_patch = self.global_bos_patch.expand(batch_size, 1, -1)
        global_decoder_input = torch.cat([bos_patch, target_patch_latents[:, :-1, :]], dim=1)
        bos_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        global_decoder_padding_mask = torch.cat([bos_mask, target_patch_mask[:, :-1]], dim=1)

        # 4. Run global decoder causally over shifted patch latents with cross-attention to memory
        n_patches = global_decoder_input.size(1)
        patch_causal_mask = torch.tril(
            torch.ones(n_patches, n_patches, dtype=torch.bool, device=device)
        )
        patch_self_mask = (
            patch_causal_mask.unsqueeze(0).unsqueeze(0)
            & global_decoder_padding_mask[:, None, None, :]
        )
        has_valid_patch = patch_self_mask.any(dim=-1, keepdim=True)
        safe_patch_self_mask = torch.where(
            has_valid_patch, patch_self_mask, torch.ones_like(patch_self_mask)
        )

        mem_key_mask = memory_padding_mask[:, None, None, :]
        has_valid_mem = memory_padding_mask.any(dim=-1, keepdim=True)[:, None, None, :]
        safe_mem_cross_mask = torch.where(
            has_valid_mem, mem_key_mask, torch.ones_like(mem_key_mask)
        )

        pos_global_input = self.target_patch_positional_encoding(global_decoder_input)
        global_patch_states = self.global_decoder(
            pos_global_input,
            memory,
            self_attention_mask=safe_patch_self_mask,
            cross_attention_mask=safe_mem_cross_mask,
        )

        # 5. Local byte decoder runs over unpadded target byte sequence with cross-attention
        # to permitted global patch states
        byte_emb = self.target_byte_embedding(decoder_input_ids) * self.embedding_scale
        byte_emb = self.target_byte_positional_encoding(byte_emb)
        byte_emb = self.embedding_dropout(byte_emb)

        hidden = self.local_decoder(
            byte_embeddings=byte_emb,
            global_patch_states=global_patch_states,
            target_padding_mask=target_padding_mask,
            global_patch_padding_mask=global_decoder_padding_mask,
        )
        return hidden

    def project(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to target byte vocabulary logits."""
        return self.output_projection(hidden_states)

    def forward(
        self,
        source_ids: Tensor,
        decoder_input_ids: Tensor,
        source_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ) -> Tensor:
        """Teacher-forced forward pass returning target byte logits."""
        memory, memory_padding_mask = self.encode(source_ids, source_padding_mask)
        hidden = self.decode(
            decoder_input_ids, memory, memory_padding_mask, target_padding_mask
        )
        return self.project(hidden)


__all__ = [
    "LocalEncoder",
    "LocalByteDecoder",
    "LocalDecoder",
    "ByteLatentTransformer",
]
