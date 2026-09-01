"""Dataset and from-scratch Byte-level BPE tokenization utilities for Assignment 1."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils import ascii_byte_values, bits_to_byte_values

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4  # 0..255 mapped to 4..259
BASE_VOCAB_SIZE = 260


def load_parallel_corpus(data_dir: str | Path = "dataset") -> tuple[list[str], list[str]]:
    """Load the aligned cipher/plaintext corpus from disk."""
    data_dir = Path(data_dir)
    cipher_path = data_dir / "brown_cipher.txt"
    plain_path = data_dir / "brown_plain.txt"
    if not cipher_path.exists() or not plain_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset files in {data_dir}. Expected brown_cipher.txt and brown_plain.txt."
        )
    cipher_lines = cipher_path.read_text(encoding="utf-8").splitlines()
    plain_lines = plain_path.read_text(encoding="utf-8").splitlines()
    if len(cipher_lines) != len(plain_lines):
        raise ValueError(
            "Dataset files are misaligned: "
            f"{cipher_path.name} has {len(cipher_lines)} lines while "
            f"{plain_path.name} has {len(plain_lines)} lines."
        )
    return cipher_lines, plain_lines


def build_dataset_splits(
    num_examples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Create deterministic train/validation/test splits."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_examples, generator=generator).tolist()

    train_end = int(num_examples * train_ratio)
    val_end = train_end + int(num_examples * val_ratio)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


class ByteTokenizer:
    """Direct unmerged byte tokenizer used by the token-free BLT model (C5)."""

    def __init__(self) -> None:
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self.byte_offset = BYTE_OFFSET
        self.vocab_size = BASE_VOCAB_SIZE

    def byte_to_id(self, value: int) -> int:
        if not 0 <= value <= 255:
            return self.unk_id
        return value + self.byte_offset

    def id_to_byte(self, value: int) -> int:
        if value < self.byte_offset:
            raise ValueError("Special token cannot be converted back to a byte")
        return value - self.byte_offset

    def encode_bytes(
        self,
        values: Sequence[int],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        encoded = [self.byte_to_id(value) for value in values]
        if add_bos:
            encoded.insert(0, self.bos_id)
        if add_eos:
            encoded.append(self.eos_id)
        return encoded

    def encode_text(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        return self.encode_bytes(
            ascii_byte_values(text),
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def encode_binary_string(
        self,
        bit_string: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        return self.encode_bytes(
            bits_to_byte_values(bit_string),
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def decode_to_text(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        byte_values: list[int] = []
        for token_id in token_ids:
            if token_id < self.byte_offset:
                if skip_special:
                    continue
                raise ValueError(f"Cannot decode special token id {token_id} to text")
            byte_values.append(self.id_to_byte(token_id))
        return bytes(byte_values).decode("utf-8", errors="replace")


class ByteBPETokenizer:
    """100% From-scratch Byte-Level BPE Tokenizer.

    Learns subword merges over byte sequences (starting from 256 individual byte
    tokens + 4 special tokens). Supports both source ciphertext bytes and target
    plaintext UTF-8 bytes.
    """

    def __init__(self, vocab_size: int = 512, min_frequency: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self.byte_offset = BYTE_OFFSET

        # Mapping from token_id -> bytes
        self.token_to_bytes: dict[int, bytes] = {
            i + self.byte_offset: bytes([i]) for i in range(256)
        }
        # Learned merges: (token_a, token_b) -> new_token_id
        self.merges: dict[tuple[int, int], int] = {}
        # Merge priority ranks: (token_a, token_b) -> rank_order (0, 1, 2, ...)
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.trained = False

    @property
    def learned_vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    def fit(self, byte_sequences: Iterable[Sequence[int] | bytes], chunk_size: int = 64) -> None:
        """Learn BPE merge operations fast using a chunk-frequency table."""
        chunk_counts: Counter[tuple[int, ...]] = Counter()
        for raw in byte_sequences:
            if isinstance(raw, (bytes, bytearray)):
                byte_list = list(raw)
            else:
                byte_list = list(raw)
            if not byte_list:
                continue
            token_list = [b + self.byte_offset for b in byte_list]
            # Split long sequences into manageable chunks for fast pair counting
            for start_idx in range(0, len(token_list), chunk_size):
                chunk = tuple(token_list[start_idx : start_idx + chunk_size])
                if len(chunk) >= 2:
                    chunk_counts[chunk] += 1

        num_merges_needed = self.vocab_size - BASE_VOCAB_SIZE
        if num_merges_needed <= 0:
            self.trained = True
            return

        for _ in range(num_merges_needed):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for chunk, freq in chunk_counts.items():
                for i in range(len(chunk) - 1):
                    pair_counts[(chunk[i], chunk[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < self.min_frequency:
                break

            new_token_id = BASE_VOCAB_SIZE + len(self.merges)
            self.merges[best_pair] = new_token_id
            self.merge_ranks[best_pair] = len(self.merge_ranks)
            self.token_to_bytes[new_token_id] = (
                self.token_to_bytes[best_pair[0]] + self.token_to_bytes[best_pair[1]]
            )

            # Apply this merge to unique chunks
            p0, p1 = best_pair
            updated_chunk_counts: Counter[tuple[int, ...]] = Counter()
            for chunk, freq in chunk_counts.items():
                new_chunk: list[int] = []
                idx = 0
                while idx < len(chunk):
                    if idx < len(chunk) - 1 and chunk[idx] == p0 and chunk[idx + 1] == p1:
                        new_chunk.append(new_token_id)
                        idx += 2
                    else:
                        new_chunk.append(chunk[idx])
                        idx += 1
                updated_chunk_counts[tuple(new_chunk)] += freq
    def _ensure_pattern(self) -> None:
        if not hasattr(self, "_compiled_pattern") or self._compiled_pattern is None:
            import re
            # Sort byte tokens by length descending so longest subwords match first
            patterns = sorted(self.token_to_bytes.items(), key=lambda x: len(x[1]), reverse=True)
            self._byte_to_id = {b: tid for tid, b in self.token_to_bytes.items()}
            self._compiled_pattern = re.compile(b"|".join(re.escape(b) for _, b in patterns))

    def encode_bytes(
        self,
        values: Sequence[int],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Tokenize a sequence of byte values (0-255) into subword token IDs."""
        self._ensure_pattern()
        raw_bytes = bytes(values)
        if not raw_bytes:
            tokens: list[int] = []
        else:
            matches = self._compiled_pattern.findall(raw_bytes)
            tokens = [self._byte_to_id[m] for m in matches]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def encode_text(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Tokenize UTF-8 text using learned BPE merges."""
        return self.encode_bytes(
            ascii_byte_values(text),
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def encode_binary_string(
        self,
        bit_string: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Tokenize binary string by grouping into bytes and applying BPE."""
        return self.encode_bytes(
            bits_to_byte_values(bit_string),
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def decode_to_bytes(self, token_ids: Sequence[int], *, skip_special: bool = True) -> bytes:
        """Decode a list of subword token IDs back into raw bytes."""
        byte_chunks: list[bytes] = []
        for token_id in token_ids:
            if token_id < self.byte_offset:
                if skip_special:
                    continue
                continue
            byte_val = self.token_to_bytes.get(token_id)
            if byte_val is not None:
                byte_chunks.append(byte_val)
        return b"".join(byte_chunks)

    def decode_to_text(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        """Decode a list of subword token IDs back to a UTF-8 string."""
        return self.decode_to_bytes(token_ids, skip_special=skip_special).decode(
            "utf-8", errors="replace"
        )

    def save(self, path: str | Path) -> Path:
        """Persist the tokenizer to a clean JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "merges": [
                [p[0], p[1], new_id] for p, new_id in self.merges.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        """Restore a ByteBPETokenizer from a JSON file."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        tok = cls(
            vocab_size=payload["vocab_size"],
            min_frequency=payload.get("min_frequency", 2),
        )
        for p0, p1, new_id in payload["merges"]:
            pair = (p0, p1)
            tok.merges[pair] = new_id
            tok.merge_ranks[pair] = len(tok.merge_ranks)
            tok.token_to_bytes[new_id] = (
                tok.token_to_bytes[p0] + tok.token_to_bytes[p1]
            )
        tok.trained = True
        return tok


# Alias for backward compatibility if referenced elsewhere
SimpleBPETokenizer = ByteBPETokenizer


@dataclass
class DatasetConfig:
    """Configuration for the tokenized and BLT-style datasets."""

    max_source_length: int | None = None
    max_target_length: int | None = None


class ParallelTextDataset(Dataset[dict[str, object]]):
    """Tokenized subword dataset used by configurations C1 to C4."""

    def __init__(
        self,
        cipher_lines: Sequence[str],
        plain_lines: Sequence[str],
        source_tokenizer: ByteBPETokenizer,
        target_tokenizer: ByteBPETokenizer,
        config: DatasetConfig | None = None,
    ) -> None:
        if len(cipher_lines) != len(plain_lines):
            raise ValueError("cipher_lines and plain_lines must have the same length")
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.config = config or DatasetConfig()
        self.examples: list[dict[str, object]] = []
        for cipher_text, plain_text in zip(cipher_lines, plain_lines):
            source_ids = source_tokenizer.encode_binary_string(cipher_text)
            target_ids = target_tokenizer.encode_text(plain_text, add_bos=True, add_eos=True)
            if self.config.max_source_length is not None:
                source_ids = source_ids[: self.config.max_source_length]
            if self.config.max_target_length is not None:
                target_ids = target_ids[: self.config.max_target_length]
            self.examples.append(
                {
                    "source_ids": source_ids,
                    "target_ids": target_ids,
                    "source_text": cipher_text,
                    "target_text": plain_text,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


class ByteLatentParallelDataset(Dataset[dict[str, object]]):
    """Byte-level dataset used by the BLT-style configuration (C5)."""

    def __init__(
        self,
        cipher_lines: Sequence[str],
        plain_lines: Sequence[str],
        config: DatasetConfig | None = None,
    ) -> None:
        if len(cipher_lines) != len(plain_lines):
            raise ValueError("cipher_lines and plain_lines must have the same length")
        self.config = config or DatasetConfig()
        self.examples: list[dict[str, object]] = []
        for cipher_text, plain_text in zip(cipher_lines, plain_lines):
            source_bytes = bits_to_byte_values(cipher_text)
            target_bytes = ascii_byte_values(plain_text)
            if self.config.max_source_length is not None:
                source_bytes = source_bytes[: self.config.max_source_length]
            if self.config.max_target_length is not None:
                target_bytes = target_bytes[: self.config.max_target_length]
            self.examples.append(
                {
                    "source_bytes": source_bytes,
                    "target_bytes": target_bytes,
                    "source_text": cipher_text,
                    "target_text": plain_text,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


def collate_tokenized_batch(
    batch: Sequence[dict[str, object]],
    *,
    pad_id: int = PAD_ID,
) -> dict[str, object]:
    """Pad a tokenized mini-batch and build decoder inputs and labels."""
    source_tensors = [torch.tensor(item["source_ids"], dtype=torch.long) for item in batch]
    target_tensors = [torch.tensor(item["target_ids"], dtype=torch.long) for item in batch]
    source_ids = pad_sequence(source_tensors, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(target_tensors, batch_first=True, padding_value=pad_id)
    return {
        "source_ids": source_ids,
        "decoder_input_ids": target_ids[:, :-1],
        "labels": target_ids[:, 1:],
        "target_ids": target_ids,
        "source_text": [item["source_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
    }


def collate_blt_batch(
    batch: Sequence[dict[str, object]],
    *,
    tokenizer: ByteTokenizer,
) -> dict[str, object]:
    """Pad a byte-level mini-batch for BLT-style experiments (C5)."""
    source_tensors = [
        torch.tensor(tokenizer.encode_bytes(item["source_bytes"]), dtype=torch.long)
        for item in batch
    ]
    target_tensors = [
        torch.tensor(
            tokenizer.encode_bytes(item["target_bytes"], add_bos=True, add_eos=True),
            dtype=torch.long,
        )
        for item in batch
    ]
    source_ids = pad_sequence(source_tensors, batch_first=True, padding_value=tokenizer.pad_id)
    target_ids = pad_sequence(target_tensors, batch_first=True, padding_value=tokenizer.pad_id)
    return {
        "source_ids": source_ids,
        "decoder_input_ids": target_ids[:, :-1],
        "labels": target_ids[:, 1:],
        "target_ids": target_ids,
        "source_text": [item["source_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
    }


def select_lines(lines: Sequence[str], indices: Sequence[int]) -> list[str]:
    """Select a subset of corpus lines by index."""
    return [lines[index] for index in indices]


def build_batch_schedule(
    lengths: Sequence[tuple[int, int]],
    *,
    batch_size: int,
    max_tokens_per_batch: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[list[int]]:
    """Create length-bucketed mini-batch schedules for efficient padding."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = random.Random(seed)
    indices = list(range(len(lengths)))

    def sort_key(index: int) -> float:
        return float(max(lengths[index]))

    if shuffle:
        jittered_key = {index: sort_key(index) + rng.random() * 8.0 for index in indices}
        order = sorted(indices, key=lambda index: jittered_key[index])
    else:
        order = sorted(indices, key=sort_key)

    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_max_length = 0
    for index in order:
        candidate_max = max(current_max_length, max(lengths[index]))
        would_exceed_examples = len(current_batch) + 1 > batch_size
        would_exceed_tokens = (len(current_batch) + 1) * candidate_max > max_tokens_per_batch
        if current_batch and (would_exceed_examples or would_exceed_tokens):
            batches.append(current_batch)
            current_batch = [index]
            current_max_length = max(lengths[index])
        else:
            current_batch.append(index)
            current_max_length = candidate_max
    if current_batch:
        batches.append(current_batch)
    if shuffle:
        rng.shuffle(batches)
    return batches


def save_target_tokenizer(tokenizer: ByteBPETokenizer, path: str | Path) -> Path:
    return tokenizer.save(path)


def load_target_tokenizer(path: str | Path) -> ByteBPETokenizer:
    return ByteBPETokenizer.load(path)
