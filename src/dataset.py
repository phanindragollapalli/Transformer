"""Dataset and tokenization utilities for Assignment 1."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils import ascii_byte_values, bits_to_byte_values

try:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    from tokenizers.trainers import BpeTrainer

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
WORD_END = "</w>"
TOKEN_PATTERN = re.compile(r"\S+|\s+")


def load_parallel_corpus(data_dir: str | Path) -> tuple[list[str], list[str]]:
    """Load the aligned cipher/plaintext corpus from disk."""
    data_dir = Path(data_dir)
    cipher_path = data_dir / "brown_cipher.txt"
    plain_path = data_dir / "brown_plain.txt"
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
    """Direct byte tokenizer with a tiny set of control tokens."""

    def __init__(self) -> None:
        self.special_tokens = list(SPECIAL_TOKENS)
        self.token_to_id = {token: index for index, token in enumerate(self.special_tokens)}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.byte_offset = len(self.special_tokens)
        self.vocab_size = self.byte_offset + 256

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


class SimpleBPETokenizer:
    """A lightweight whitespace-preserving BPE tokenizer for the plaintext side."""

    def __init__(self, vocab_size: int = 512, min_pair_frequency: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_pair_frequency = min_pair_frequency
        self.special_tokens = list(SPECIAL_TOKENS)
        self.token_to_id = {token: index for index, token in enumerate(self.special_tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.merges: list[tuple[str, str]] = []
        self.backend_tokenizer: Tokenizer | None = None
        self.trained = False

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    @staticmethod
    def _apply_merge(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        merged: list[str] = []
        index = 0
        while index < len(symbols):
            if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
                merged.append(symbols[index] + symbols[index + 1])
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        return tuple(merged)

    def fit(self, texts: Iterable[str]) -> None:
        texts = list(texts)
        if TOKENIZERS_AVAILABLE:
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
            tokenizer.decoder = ByteLevelDecoder()
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_pair_frequency,
                special_tokens=self.special_tokens,
            )
            tokenizer.train_from_iterator(texts, trainer=trainer)
            vocab = tokenizer.get_vocab()
            self.token_to_id = dict(vocab)
            self.id_to_token = {index: token for token, index in self.token_to_id.items()}
            self.pad_id = self.token_to_id["<pad>"]
            self.bos_id = self.token_to_id["<bos>"]
            self.eos_id = self.token_to_id["<eos>"]
            self.unk_id = self.token_to_id["<unk>"]
            self.backend_tokenizer = tokenizer
            self.trained = True
            return

        word_frequencies: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for piece in self._pretokenize(text):
                word_frequencies[tuple(list(piece) + [WORD_END])] += 1

        symbols = {
            symbol
            for word in word_frequencies
            for symbol in word
        }
        while len(self.special_tokens) + len(symbols) < self.vocab_size:
            pair_frequencies: Counter[tuple[str, str]] = Counter()
            for word, frequency in word_frequencies.items():
                for index in range(len(word) - 1):
                    pair_frequencies[(word[index], word[index + 1])] += frequency
            if not pair_frequencies:
                break
            best_pair, best_frequency = pair_frequencies.most_common(1)[0]
            if best_frequency < self.min_pair_frequency:
                break
            self.merges.append(best_pair)
            merged_frequencies: Counter[tuple[str, ...]] = Counter()
            for word, frequency in word_frequencies.items():
                merged_frequencies[self._apply_merge(word, best_pair)] += frequency
            word_frequencies = merged_frequencies
            symbols = {symbol for word in word_frequencies for symbol in word}

        next_id = len(self.special_tokens)
        for symbol in sorted(symbols):
            if symbol not in self.token_to_id:
                self.token_to_id[symbol] = next_id
                self.id_to_token[next_id] = symbol
                next_id += 1
        self.trained = True

    def _ensure_trained(self) -> None:
        if not self.trained:
            raise RuntimeError("SimpleBPETokenizer.fit must be called before encoding")

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        self._ensure_trained()
        if self.backend_tokenizer is not None:
            token_ids = self.backend_tokenizer.encode(text).ids
            if add_bos:
                token_ids.insert(0, self.bos_id)
            if add_eos:
                token_ids.append(self.eos_id)
            return token_ids

        token_ids: list[int] = []
        for piece in self._pretokenize(text):
            symbols: tuple[str, ...] = tuple(list(piece) + [WORD_END])
            for pair in self.merges:
                symbols = self._apply_merge(symbols, pair)
            for symbol in symbols:
                token_ids.append(self.token_to_id.get(symbol, self.unk_id))
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        if self.backend_tokenizer is not None:
            filtered_ids: list[int] = []
            for token_id in token_ids:
                token = self.id_to_token.get(int(token_id), "<unk>")
                if token in self.special_tokens and skip_special:
                    continue
                filtered_ids.append(int(token_id))
            return self.backend_tokenizer.decode(filtered_ids)

        pieces: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if token in self.special_tokens:
                if skip_special:
                    continue
                pieces.append(token)
                continue
            pieces.append(token.replace(WORD_END, ""))
        return "".join(pieces)

    @property
    def learned_vocab_size(self) -> int:
        return len(self.token_to_id)


@dataclass
class DatasetConfig:
    """Configuration for the tokenized and BLT-style datasets."""

    max_source_length: int | None = None
    max_target_length: int | None = None


class ParallelTextDataset(Dataset[dict[str, object]]):
    """Tokenized dataset used by configurations C1 to C4."""

    def __init__(
        self,
        cipher_lines: Sequence[str],
        plain_lines: Sequence[str],
        source_tokenizer: ByteTokenizer,
        target_tokenizer: SimpleBPETokenizer,
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
            target_ids = target_tokenizer.encode(plain_text, add_bos=True, add_eos=True)
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
    """Byte-level dataset used by the BLT-style configuration."""

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
    pad_id: int,
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
    """Pad a byte-level mini-batch for BLT-style experiments."""
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
    """Create length-bucketed mini-batch schedules for efficient padding.

    Indices are sorted by sequence length (with optional jitter when
    shuffling so bucket composition varies across epochs), greedily packed
    under both a per-batch example cap and a token budget cap, and the
    resulting batch order is shuffled for training runs.
    """
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


def save_target_tokenizer(tokenizer: SimpleBPETokenizer, path: str | Path) -> Path:
    """Persist a trained target tokenizer to disk for reproducible runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer.backend_tokenizer is not None:
        tokenizer.backend_tokenizer.save(str(path))
        return path
    payload = {
        "format": "simple_bpe_fallback",
        "vocab_size": tokenizer.vocab_size,
        "min_pair_frequency": tokenizer.min_pair_frequency,
        "merges": [list(pair) for pair in tokenizer.merges],
        "token_to_id": tokenizer.token_to_id,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_target_tokenizer(path: str | Path) -> SimpleBPETokenizer:
    """Restore a previously trained target tokenizer from disk."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, dict) and payload.get("format") == "simple_bpe_fallback":
        tokenizer = SimpleBPETokenizer(
            vocab_size=payload["vocab_size"],
            min_pair_frequency=payload["min_pair_frequency"],
        )
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        tokenizer.token_to_id = dict(payload["token_to_id"])
        tokenizer.id_to_token = {index: token for token, index in tokenizer.token_to_id.items()}
        tokenizer.pad_id = tokenizer.token_to_id["<pad>"]
        tokenizer.bos_id = tokenizer.token_to_id["<bos>"]
        tokenizer.eos_id = tokenizer.token_to_id["<eos>"]
        tokenizer.unk_id = tokenizer.token_to_id["<unk>"]
        tokenizer.trained = True
        return tokenizer
    tokenizer = SimpleBPETokenizer()
    tokenizer.backend_tokenizer = Tokenizer.from_file(str(path))
    vocab = tokenizer.backend_tokenizer.get_vocab()
    tokenizer.token_to_id = dict(vocab)
    tokenizer.id_to_token = {index: token for token, index in tokenizer.token_to_id.items()}
    tokenizer.pad_id = tokenizer.token_to_id["<pad>"]
    tokenizer.bos_id = tokenizer.token_to_id["<bos>"]
    tokenizer.eos_id = tokenizer.token_to_id["<eos>"]
    tokenizer.unk_id = tokenizer.token_to_id["<unk>"]
    tokenizer.trained = True
    return tokenizer
