"""Phase 1 dataset inspection utility."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def summarize_lengths(lines: list[str]) -> dict[str, float | int]:
    lengths = [len(line) for line in lines]
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(sum(lengths) / len(lengths), 3),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "Dataset_A1"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cipher_lines = load_lines(data_dir / "brown_cipher.txt")
    plain_lines = load_lines(data_dir / "brown_plain.txt")

    if len(cipher_lines) != len(plain_lines):
        raise ValueError("Cipher and plaintext files are not line-aligned.")

    cipher_counter = Counter(cipher_lines)
    mapping: dict[str, set[str]] = defaultdict(set)
    for cipher, plain in zip(cipher_lines, plain_lines):
        mapping[cipher].add(plain)

    cipher_lengths = [len(line) for line in cipher_lines]
    plain_lengths = [len(line) for line in plain_lines]
    ratios = [c_len / p_len for c_len, p_len in zip(cipher_lengths, plain_lengths) if p_len > 0]

    summary = {
        "data_dir": str(data_dir.relative_to(root)),
        "num_pairs": len(cipher_lines),
        "alignment_verified": True,
        "cipher_characters": sorted(set("".join(cipher_lines))),
        "plain_unique_characters": len(set("".join(plain_lines))),
        "cipher_length_stats": summarize_lengths(cipher_lines),
        "plain_length_stats": summarize_lengths(plain_lines),
        "cipher_to_plain_length_ratio": {
            "min": round(min(ratios), 3),
            "max": round(max(ratios), 3),
            "avg": round(sum(ratios) / len(ratios), 3),
        },
        "unique_cipher_sequences": len(set(cipher_lines)),
        "unique_plain_sequences": len(set(plain_lines)),
        "duplicate_cipher_sequences": sum(1 for count in cipher_counter.values() if count > 1),
        "conflicting_cipher_mappings": sum(1 for plains in mapping.values() if len(plains) > 1),
        "suggested_split": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 42,
        },
        "phase_1_decisions": {
            "binary_to_byte_conversion": "Group each contiguous 8 bits into one byte before standard tokenization.",
            "sequence_handling": "Keep full sequences and rely on bucketing or dynamic batching to manage memory.",
        },
    }

    output_path = outputs_dir / "dataset_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote dataset summary to {output_path}")


if __name__ == "__main__":
    main()
