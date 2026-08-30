"""Shared utilities for the Assignment 1 codebase."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

import Levenshtein
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu as nltk_corpus_bleu
from rouge_score import rouge_scorer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_serializable(payload: Any) -> Any:
    """Convert common training objects into JSON-serializable structures."""
    if is_dataclass(payload):
        return to_serializable(asdict(payload))
    if isinstance(payload, dict):
        return {key: to_serializable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [to_serializable(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with stable formatting for experiment metadata."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_serializable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a Python dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bits_to_byte_values(bit_string: str) -> list[int]:
    """Convert a binary string into byte values."""
    if len(bit_string) % 8 != 0:
        raise ValueError(
            f"Expected bit string length to be divisible by 8, got {len(bit_string)}"
        )
    if set(bit_string) - {"0", "1"}:
        raise ValueError("Bit string contains characters other than 0 and 1")
    return [int(bit_string[index : index + 8], 2) for index in range(0, len(bit_string), 8)]


def ascii_byte_values(text: str) -> list[int]:
    """Encode text as UTF-8 bytes."""
    return list(text.encode("utf-8"))


def make_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Create a boolean padding mask with True at valid positions."""
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D token tensor, got shape {tuple(tokens.shape)}")
    return tokens.ne(pad_id)


def make_causal_mask(sequence_length: int, device: torch.device | None = None) -> torch.Tensor:
    """Create a lower-triangular causal mask for autoregressive decoding."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
    )


def combine_attention_masks(
    padding_mask: torch.Tensor | None,
    causal_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Combine broadcastable padding and causal masks into one boolean mask."""
    if padding_mask is None:
        return causal_mask
    if causal_mask is None:
        return padding_mask
    return padding_mask & causal_mask


def shift_right(target_tokens: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Prepare decoder inputs by shifting the target tokens to the right."""
    if target_tokens.ndim != 2:
        raise ValueError(f"Expected 2D target tensor, got shape {tuple(target_tokens.shape)}")
    shifted = target_tokens.clone()
    shifted[:, 1:] = target_tokens[:, :-1]
    shifted[:, 0] = bos_id
    return shifted


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Choose the execution device for training and evaluation."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch dictionary onto the selected device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    *,
    epoch: int,
    config: Any | None = None,
    metrics: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> None:
    """Persist model and optimizer state for later resumption."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": to_serializable(config) if config is not None else None,
        "metrics": to_serializable(metrics) if metrics is not None else None,
    }
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model and optimizer state."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


@dataclass
class MetricTracker:
    """Track scalar metrics across multiple updates."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, **metrics: float) -> None:
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value)
            self.counts[name] = self.counts.get(name, 0) + 1

    def average(self, name: str) -> float:
        count = self.counts.get(name, 0)
        if count == 0:
            raise KeyError(f"Metric {name!r} has not been recorded")
        return self.totals[name] / count

    def averages(self) -> dict[str, float]:
        return {name: self.totals[name] / self.counts[name] for name in self.totals}

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()


@dataclass
class HistoryTracker:
    """Store per-epoch metric histories for later reporting."""

    history: dict[str, list[float]] = field(default_factory=dict)

    def log(self, **metrics: float) -> None:
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(float(value))

    def latest(self) -> dict[str, float]:
        return {name: values[-1] for name, values in self.history.items() if values}


class Timer(AbstractContextManager["Timer"]):
    """Simple wall-clock timer context manager."""

    def __init__(self) -> None:
        self.started_at = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        self.elapsed = time.perf_counter() - self.started_at
        return None


def get_peak_memory_mb(device: torch.device | None = None) -> float:
    """Return peak CUDA memory usage in megabytes when available."""
    if device is None:
        device = resolve_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_peak_memory_stats(device: torch.device | None = None) -> None:
    """Reset CUDA peak memory counters when running on GPU."""
    if device is None:
        device = resolve_device()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


class NullWandbRun:
    """Fallback object matching the minimal WandB API used by this project."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.url = None

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


def init_wandb_run(
    *,
    project: str,
    config: dict[str, Any],
    run_name: str,
    enabled: bool = True,
    output_dir: str | Path | None = None,
) -> Any:
    """Create a WandB run or return a no-op fallback if unavailable."""
    if not enabled:
        return NullWandbRun("wandb disabled in config")
    if not WANDB_AVAILABLE:
        return NullWandbRun("wandb package not installed")
    init_kwargs: dict[str, Any] = {
        "project": project,
        "config": to_serializable(config),
        "name": run_name,
    }
    if output_dir is not None:
        init_kwargs["dir"] = str(output_dir)
    return wandb.init(**init_kwargs)


# ---------------------------------------------------------------------------
# Evaluation metrics (Phase 7)
# ---------------------------------------------------------------------------


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Edit distance between the reference and predicted strings using standard Levenshtein library."""
    return Levenshtein.distance(reference, hypothesis)


def text_to_bits(text: str) -> tuple[int, ...]:
    """Convert text into its UTF-8 bit tuple for bit-level comparisons."""
    byte_values = text.encode("utf-8")
    bits: list[int] = []
    for value in byte_values:
        bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    return tuple(bits)


def bit_level_accuracy(hypothesis: str, reference: str) -> float:
    """Fraction of matching bits between prediction and target.

    Both strings are encoded as UTF-8 bytes and expanded to bits. Positions
    beyond the shorter bit string count as mismatches, so length errors are
    penalized in addition to content errors.
    """
    hypothesis_bits = text_to_bits(hypothesis)
    reference_bits = text_to_bits(reference)
    total_length = max(len(hypothesis_bits), len(reference_bits))
    if total_length == 0:
        return 1.0
    matched = sum(
        1 for a, b in zip(hypothesis_bits, reference_bits) if a == b
    )
    return matched / total_length


def sequence_accuracy(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Fraction of examples whose entire prediction matches the target exactly."""
    if len(references) == 0:
        raise ValueError("sequence_accuracy requires at least one example")
    correct = sum(1 for hyp, ref in zip(hypotheses, references) if hyp == ref)
    return correct / len(references)


def mean_bit_level_accuracy(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Average bit-level accuracy across paired predictions and targets."""
    if len(references) == 0:
        raise ValueError("mean_bit_level_accuracy requires at least one example")
    return sum(
        bit_level_accuracy(hyp, ref) for hyp, ref in zip(hypotheses, references)
    ) / len(references)


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> float:
    """Corpus-level BLEU using NLTK's corpus_bleu with smoothing."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not references:
        return 0.0
    list_of_references = [[ref.split()] for ref in references]
    list_of_hypotheses = [hyp.split() for hyp in hypotheses]
    smoothing = SmoothingFunction().method1
    return float(
        nltk_corpus_bleu(
            list_of_references,
            list_of_hypotheses,
            smoothing_function=smoothing,
        )
    )


def rouge_scores(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores using rouge-score library."""
    if not references:
        return {"rouge_1_f1": 0.0, "rouge_2_f1": 0.0, "rouge_l_f1": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {"rouge_1_f1": 0.0, "rouge_2_f1": 0.0, "rouge_l_f1": 0.0}
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        totals["rouge_1_f1"] += scores["rouge1"].fmeasure
        totals["rouge_2_f1"] += scores["rouge2"].fmeasure
        totals["rouge_l_f1"] += scores["rougeL"].fmeasure
    num_examples = len(references)
    return {name: val / num_examples for name, val in totals.items()}


def evaluate_text_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    include_nlp_metrics: bool = True,
) -> dict[str, float]:
    """Compute the assignment metric suite over decoded text pairs using standard libraries.

    BLEU and ROUGE apply to tokenized configurations only; set
    ``include_nlp_metrics=False`` for the token-free BLT path.
    """
    metrics: dict[str, float] = {
        "bit_level_accuracy": mean_bit_level_accuracy(hypotheses, references),
        "sequence_accuracy": sequence_accuracy(hypotheses, references),
        "levenshtein_distance": (
            sum(levenshtein_distance(ref, hyp) for hyp, ref in zip(hypotheses, references))
            / max(len(references), 1)
        ),
    }
    if include_nlp_metrics:
        metrics["bleu"] = corpus_bleu(hypotheses, references)
        metrics.update(rouge_scores(hypotheses, references))
    return metrics


# ---------------------------------------------------------------------------
# Plots (Phase 7)
# ---------------------------------------------------------------------------

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path, title: str = "") -> Path | None:
    """Plot train/val loss curves (with optional epoch-time bars) from history."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    if not any(history.get(key) for key in ("train_loss", "val_loss")):
        return None
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    figure, axis = plt.subplots(figsize=(7, 4))
    legend_handles = []
    legend_labels = []
    if history.get("train_loss"):
        handle = axis.plot(
            range(1, len(history["train_loss"]) + 1),
            history["train_loss"],
            label="train loss",
        )[0]
        legend_handles.append(handle)
        legend_labels.append("train loss")
    if history.get("val_loss"):
        handle = axis.plot(
            range(1, len(history["val_loss"]) + 1),
            history["val_loss"],
            label="val loss",
        )[0]
        legend_handles.append(handle)
        legend_labels.append("val loss")
    if history.get("epoch_time_seconds"):
        time_axis = axis.twinx()
        time_axis.bar(
            range(1, len(history["epoch_time_seconds"]) + 1),
            history["epoch_time_seconds"],
            alpha=0.25,
            color="tab:green",
        )
        time_axis.set_ylabel("epoch time (s)", color="tab:green")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    if title:
        axis.set_title(title)
    axis.legend(legend_handles, legend_labels, loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_config_comparison(
    stats_by_config: dict[str, dict[str, float]],
    metric_names: Sequence[str],
    output_path: str | Path,
    title: str = "Configuration comparison",
) -> Path | None:
    """Grouped bar chart comparing scalar stats across configurations C1-C5."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    config_ids = sorted(stats_by_config)
    usable_metrics = [
        name
        for name in metric_names
        if all(name in stats_by_config[config_id] for config_id in config_ids)
    ]
    if not usable_metrics or not config_ids:
        return None
    width = 0.8 / len(usable_metrics)
    x_positions = np.arange(len(config_ids))
    figure, axis = plt.subplots(figsize=(max(6.0, 1.5 * len(config_ids)), 4))
    for index, metric_name in enumerate(usable_metrics):
        values = [stats_by_config[config_id][metric_name] for config_id in config_ids]
        axis.bar(x_positions + index * width, values, width, label=metric_name)
    axis.set_xticks(x_positions + width * (len(usable_metrics) - 1) / 2)
    axis.set_xticklabels(config_ids)
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
