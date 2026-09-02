"""Shared experiment configuration and the main training entrypoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import (
    ByteBPETokenizer,
    ByteLatentParallelDataset,
    ByteTokenizer,
    DatasetConfig,
    ParallelTextDataset,
    build_batch_schedule,
    build_dataset_splits,
    chunk_parallel_sentences,
    collate_blt_batch,
    collate_tokenized_batch,
    load_parallel_corpus,
    prepare_chunked_splits,
    select_lines,
)
from src.models.transformer import build_model, greedy_decode, trim_after_eos
from src.utils import (
    HistoryTracker,
    MetricTracker,
    Timer,
    ascii_byte_values,
    bits_to_byte_values,
    ensure_dir,
    evaluate_text_metrics,
    get_peak_memory_mb,
    init_wandb_run,
    move_batch_to_device,
    plot_training_curves,
    reset_peak_memory_stats,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Single source of truth for a configuration run."""

    config_id: str
    run_name: str
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    data_dir: str = "dataset"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    tokenizer_dir: str = "outputs/tokenizers"
    wandb_project: str = "ANLP_A1"
    use_wandb: bool = True
    prefer_cuda: bool = True
    embedding_dim: int = 256
    ff_hidden_dim: int = 1024
    encoder_layers: int = 4
    decoder_layers: int = 4
    attention_heads: int = 8
    query_groups: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_tokens_per_batch: int = 4096
    epochs: int = 30
    chunk_size_bytes: int = 64
    max_source_length: int = 128
    max_target_length: int = 128
    binary_to_byte_compaction: bool = True
    length_bucketing: bool = True
    dynamic_batching: bool = True
    positional_encoding: str = "sinusoidal"
    attention_type: str = "mha"
    normalization_type: str = "layernorm"
    tokenization_type: str = "subword"
    source_tokenizer_type: str = "bpe"
    source_vocab_size: int = 760
    target_vocab_size: int = 760
    blt_patch_size: int = 16
    scheduler_type: str = "none"
    gradient_clip_norm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_base_config() -> ExperimentConfig:
    """Create the baseline configuration for C1."""
    return ExperimentConfig(config_id="C1", run_name="A1_C1_baseline")


def build_experiment_configs() -> dict[str, ExperimentConfig]:
    """Create the five assignment configurations with single-component changes."""
    base = build_base_config()
    return {
        "C1": base,
        "C2": replace(
            base,
            config_id="C2",
            run_name="A1_C2_rope",
            positional_encoding="rope",
        ),
        "C3": replace(
            base,
            config_id="C3",
            run_name="A1_C3_gqa",
            attention_type="gqa",
        ),
        "C4": replace(
            base,
            config_id="C4",
            run_name="A1_C4_rmsnorm",
            normalization_type="rmsnorm",
        ),
        "C5": replace(
            base,
            config_id="C5",
            run_name="A1_C5_blt",
            tokenization_type="blt",
            source_tokenizer_type="byte",
        ),
    }


def export_configs(
    configs: dict[str, ExperimentConfig],
    output_dir: str | Path = "outputs/configs",
) -> Path:
    """Save config metadata to disk for reproducibility and inspection."""
    output_dir = ensure_dir(output_dir)
    payload = {name: config.to_dict() for name, config in configs.items()}
    output_path = output_dir / "experiment_configs.json"
    save_json(payload, output_path)
    return output_path


def get_config(config_id: str) -> ExperimentConfig:
    """Return one configuration by identifier."""
    configs = build_experiment_configs()
    try:
        return configs[config_id]
    except KeyError as exc:
        raise KeyError(f"Unknown config_id {config_id!r}. Expected one of {sorted(configs)}.") from exc


# ---------------------------------------------------------------------------
# Tokenizer preparation
# ---------------------------------------------------------------------------


def prepare_tokenizers(
    config: ExperimentConfig,
    cipher_train_lines: Sequence[str],
    plain_train_lines: Sequence[str],
) -> dict[str, Any]:
    """Build or restore the source and target tokenizers for a run.

    For C1-C4, from-scratch ByteBPETokenizers are trained on the training
    split's cipher bytes and plaintext bytes, respectively.
    For C5, direct ByteTokenizers are used.
    """
    if config.tokenization_type == "blt":
        byte_tok = ByteTokenizer()
        return {"source": byte_tok, "target": byte_tok}

    tokenizer_dir = ensure_dir(config.tokenizer_dir)
    outputs_dir = ensure_dir(config.output_dir)

    # Source ciphertext BPE tokenizer (BPE over 8-bit bytes)
    source_tok_path = tokenizer_dir / f"source_bpe_vocab{config.source_vocab_size}.json"
    source_txt_path = tokenizer_dir / "source_bpe_merges.txt"
    source_json_path = tokenizer_dir / "source_bpe_merges.json"
    if source_tok_path.exists():
        source_tokenizer = ByteBPETokenizer.load(source_tok_path)
    else:
        source_tokenizer = ByteBPETokenizer(vocab_size=config.source_vocab_size)
        fit_lines = cipher_train_lines[: min(5000, len(cipher_train_lines))]
        source_byte_seqs = [bits_to_byte_values(line) for line in fit_lines]
        source_tokenizer.fit(source_byte_seqs)
        source_tokenizer.save(source_tok_path)

    source_tokenizer.export_merge_rules(source_txt_path, source_json_path)
    source_tokenizer.export_merge_rules(
        outputs_dir / "source_bpe_merges.txt", outputs_dir / "source_bpe_merges.json"
    )

    # Target plaintext BPE tokenizer (BPE over UTF-8 bytes)
    target_tok_path = tokenizer_dir / f"target_bpe_vocab{config.target_vocab_size}.json"
    target_txt_path = tokenizer_dir / "target_bpe_merges.txt"
    target_json_path = tokenizer_dir / "target_bpe_merges.json"
    if target_tok_path.exists():
        target_tokenizer = ByteBPETokenizer.load(target_tok_path)
    else:
        target_tokenizer = ByteBPETokenizer(vocab_size=config.target_vocab_size)
        fit_lines = plain_train_lines[: min(5000, len(plain_train_lines))]
        target_byte_seqs = [ascii_byte_values(line) for line in fit_lines]
        target_tokenizer.fit(target_byte_seqs)
        target_tokenizer.save(target_tok_path)

    target_tokenizer.export_merge_rules(target_txt_path, target_json_path)
    target_tokenizer.export_merge_rules(
        outputs_dir / "target_bpe_merges.txt", outputs_dir / "target_bpe_merges.json"
    )

    return {
        "source": source_tokenizer,
        "target": target_tokenizer,
        "source_path": source_tok_path,
        "target_path": target_tok_path,
        "source_merges_txt": source_txt_path,
        "target_merges_txt": target_txt_path,
    }


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------


def make_criterion(pad_id: int) -> torch.nn.CrossEntropyLoss:
    """Cross-entropy ignoring padded label positions."""
    return torch.nn.CrossEntropyLoss(ignore_index=pad_id)


def run_model_forward(model, batch: dict[str, Any]) -> torch.Tensor:
    """Shared teacher-forced forward pass over a collated batch."""
    return model(
        batch["source_ids"],
        batch["decoder_input_ids"],
        batch["source_ids"].ne(0),
        batch["decoder_input_ids"].ne(0),
    )


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
    tracker: MetricTracker,
) -> None:
    """Run one teacher-forced training epoch, recording token-weighted loss."""
    model.train()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = run_model_forward(model, batch)
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        num_tokens = int(batch["labels"].ne(0).sum())
        tracker.update(train_loss=float(loss.detach()) * num_tokens, loss_weight=num_tokens)


def evaluate_loss(
    model,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    tracker: MetricTracker,
) -> None:
    """Compute validation loss without gradients and update the tracker."""
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = run_model_forward(model, batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            num_tokens = int(batch["labels"].ne(0).sum())
            tracker.update(val_loss=float(loss) * num_tokens, loss_weight=num_tokens)


def tracker_average(tracker: MetricTracker, metric_name: str, weight_name: str) -> float:
    """Token-weighted average of a loss accumulated in a tracker."""
    weight_total = tracker.totals.get(weight_name, 0.0)
    if weight_total == 0.0:
        return float("nan")
    return tracker.totals[metric_name] / weight_total


# ---------------------------------------------------------------------------
# Greedy test-set evaluation
# ---------------------------------------------------------------------------


def compute_test_predictions(
    model,
    loader: DataLoader,
    *,
    config: ExperimentConfig,
    target_tokenizer: ByteTokenizer | ByteBPETokenizer,
    device: torch.device,
) -> dict[str, Any]:
    """Greedy-decode a data loader and compute all assignment metrics.

    BLEU/ROUGE are only computed for tokenized configurations; the BLT path
    is genuinely token-free so those n-gram-overlap metrics are skipped.
    """
    is_blt = config.tokenization_type == "blt"
    bos_id = target_tokenizer.bos_id
    eos_id = target_tokenizer.eos_id
    pad_id = target_tokenizer.pad_id
    hypotheses: list[str] = []
    references: list[str] = []
    sources: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            sequences = greedy_decode(
                model,
                batch["source_ids"],
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                max_length=config.max_target_length + 4,
            )
            trimmed_rows = trim_after_eos(sequences, eos_id=eos_id, pad_id=pad_id)
            for row in trimmed_rows:
                text = target_tokenizer.decode_to_text(row)
                hypotheses.append(text)
            references.extend(batch["target_text"])
            sources.extend(batch["source_text"])

    metrics = evaluate_text_metrics(hypotheses, references, include_nlp_metrics=not is_blt)
    samples = [
        {
            "source_cipher": source[:96],
            "prediction": prediction[:160],
            "reference": reference[:160],
        }
        for source, prediction, reference in zip(sources, hypotheses, references)
    ]
    return {"metrics": metrics, "samples": samples}


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def dataset_example_lengths(dataset: Sequence[dict[str, Any]], is_blt: bool) -> list[tuple[int, int]]:
    """Return (source, target) lengths for every example in a dataset."""
    source_key = "source_bytes" if is_blt else "source_ids"
    target_key = "target_bytes" if is_blt else "target_ids"
    return [(len(example[source_key]), len(example[target_key])) for example in dataset]


def run_experiment(
    config: ExperimentConfig,
    *,
    smoke: bool = False,
    device_override: str | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    disable_wandb: bool = False,
) -> dict[str, Any]:
    """Train one configuration end-to-end and persist every artifact."""
    if epochs_override is not None:
        config = replace(config, epochs=epochs_override)
    if batch_size_override is not None:
        config = replace(config, batch_size=batch_size_override)
    if smoke:
        config = replace(config, epochs=min(config.epochs, epochs_override or 2), use_wandb=False)

    set_seed(config.seed)
    device = resolve_device(prefer_cuda=config.prefer_cuda)
    if device_override:
        device = torch.device(device_override)

    chunked_splits = prepare_chunked_splits(
        data_dir=config.data_dir,
        chunk_size_bytes=config.chunk_size_bytes,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    if smoke:
        chunked_splits = {
            "train": (chunked_splits["train"][0][:64], chunked_splits["train"][1][:64]),
            "val": (chunked_splits["val"][0][:16], chunked_splits["val"][1][:16]),
            "test": (chunked_splits["test"][0][:16], chunked_splits["test"][1][:16]),
        }

    tokenizers = prepare_tokenizers(
        config,
        chunked_splits["train"][0],
        chunked_splits["train"][1],
    )
    source_tokenizer = tokenizers["source"]
    target_tokenizer = tokenizers["target"]

    dataset_config = DatasetConfig(
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )
    is_blt = config.tokenization_type == "blt"
    if is_blt:
        datasets = {
            name: ByteLatentParallelDataset(
                c_chunks,
                p_chunks,
                dataset_config,
            )
            for name, (c_chunks, p_chunks) in chunked_splits.items()
        }
        src_vocab_size = source_tokenizer.vocab_size
        tgt_vocab_size = target_tokenizer.vocab_size
        collate_fn = partial(collate_blt_batch, tokenizer=source_tokenizer)
    else:
        datasets = {
            name: ParallelTextDataset(
                c_chunks,
                p_chunks,
                source_tokenizer,
                target_tokenizer,
                dataset_config,
            )
            for name, (c_chunks, p_chunks) in chunked_splits.items()
        }
        src_vocab_size = source_tokenizer.learned_vocab_size
        tgt_vocab_size = target_tokenizer.learned_vocab_size
        collate_fn = partial(collate_tokenized_batch, pad_id=target_tokenizer.pad_id)

    def build_loader(name: str, shuffle_batches: bool, schedule_seed: int | None) -> DataLoader:
        lengths = dataset_example_lengths(datasets[name], is_blt)
        schedule = build_batch_schedule(
            lengths,
            batch_size=config.batch_size,
            max_tokens_per_batch=config.max_tokens_per_batch,
            shuffle=shuffle_batches,
            seed=schedule_seed,
        )
        return DataLoader(datasets[name], batch_sampler=schedule, collate_fn=collate_fn)

    val_loader = build_loader("val", False, None)
    test_loader = build_loader("test", False, None)

    model = build_model(config, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = make_criterion(target_tokenizer.pad_id)

    wandb_run = init_wandb_run(
        project=config.wandb_project,
        config={**config.to_dict(), "smoke": smoke},
        run_name=config.run_name,
        enabled=config.use_wandb and not disable_wandb,
        output_dir=config.output_dir,
    )

    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    plots_dir = ensure_dir(Path(config.output_dir) / "plots")
    logs_dir = ensure_dir(Path(config.output_dir) / "logs")
    history = HistoryTracker()
    best_val_loss = float("inf")
    reset_peak_memory_stats(device)
    total_training_timer = Timer()

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{config.run_name}] training on {device} | {parameter_count:,} parameters (src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size})")
    with total_training_timer:
        for epoch in range(1, config.epochs + 1):
            train_tracker = MetricTracker()
            val_tracker = MetricTracker()
            epoch_timer = Timer()
            with epoch_timer:
                train_loader = build_loader("train", True, schedule_seed=config.seed + epoch)
                train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    config.gradient_clip_norm, train_tracker,
                )
                evaluate_loss(model, val_loader, criterion, device, val_tracker)
            train_loss = tracker_average(train_tracker, "train_loss", "loss_weight")
            val_loss = tracker_average(val_tracker, "val_loss", "loss_weight")
            history.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_time_seconds=epoch_timer.elapsed,
            )
            wandb_run.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_seconds": epoch_timer.elapsed,
                "peak_memory_mb": get_peak_memory_mb(device),
                "epoch": epoch,
            })
            print(
                f"[{config.run_name}] epoch {epoch:02d}/{config.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"time={epoch_timer.elapsed:.1f}s"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, Path(checkpoint_dir) / f"{config.run_name}_best.pt",
                    epoch=epoch, config=config, metrics={"val_loss": val_loss},
                )
            save_checkpoint(
                model, optimizer, Path(checkpoint_dir) / f"{config.run_name}_final.pt",
                epoch=epoch, config=config, metrics={"val_loss": val_loss},
            )

    test_results = compute_test_predictions(
        model,
        test_loader,
        config=config,
        target_tokenizer=target_tokenizer,
        device=device,
    )

    peak_memory_mb = get_peak_memory_mb(device)
    summary = {
        "run_name": config.run_name,
        "config_id": config.config_id,
        "config": config.to_dict(),
        "smoke": smoke,
        "history": history.history,
        "best_val_loss": best_val_loss,
        "total_training_time_seconds": total_training_timer.elapsed,
        "peak_memory_mb": peak_memory_mb,
        "test_metrics": test_results["metrics"],
        "sample_predictions": test_results["samples"][:8],
        "num_parameters": parameter_count,
        "wandb_run_url": getattr(wandb_run, "url", None),
    }
    save_json(summary, Path(logs_dir) / f"{config.run_name}.json")
    plot_training_curves(
        history.history,
        plots_dir / f"{config.run_name}_curves.png",
        title=f"{config.run_name} ({config.config_id})",
    )
    wandb_run.log({"test_metrics": test_results["metrics"], "peak_memory_mb": peak_memory_mb})
    wandb_run.finish()

    print(
        f"[{config.run_name}] done in {total_training_timer.elapsed:.1f}s | "
        f"test metrics: {summary['test_metrics']}"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training entrypoint."""
    parser = argparse.ArgumentParser(description="Train Assignment 1 configurations")
    parser.add_argument("--config", default="C1", help="Configuration id (C1-C5) or 'all'")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--smoke", action="store_true", help="Tiny subset sanity run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--device", default=None, help="cpu, cuda, or leave unset for auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export shared configs and launch requested experiment runs."""
    export_configs(build_experiment_configs())
    args = parse_args(argv)
    config_ids = sorted(build_experiment_configs()) if args.config == "all" else [args.config]
    for config_id in config_ids:
        run_experiment(
            get_config(config_id),
            smoke=args.smoke,
            device_override=args.device,
            epochs_override=args.epochs,
            batch_size_override=args.batch_size,
            disable_wandb=args.no_wandb,
        )


if __name__ == "__main__":
    main()
