"""Utility script to upload or download model checkpoints to/from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed. Please run: pip install huggingface_hub")
    sys.exit(1)

DEFAULT_REPO_ID = "phani4104/Transformer"
CHECKPOINT_FILES = [
    "A1_C1_baseline_best.pt",
    "A1_C1_baseline_final.pt",
    "A1_C2_rope_best.pt",
    "A1_C2_rope_final.pt",
    "A1_C3_gqa_best.pt",
    "A1_C3_gqa_final.pt",
    "A1_C4_rmsnorm_best.pt",
    "A1_C4_rmsnorm_final.pt",
    "A1_C5_blt_best.pt",
    "A1_C5_blt_final.pt",
]


def upload_checkpoints(
    repo_id: str,
    checkpoint_dir: Path,
    token: str | None = None,
) -> None:
    """Upload checkpoint files from local directory to Hugging Face repository."""
    if not checkpoint_dir.exists():
        print(f"Error: Local checkpoint directory '{checkpoint_dir}' does not exist.")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Checking repository '{repo_id}' on Hugging Face...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    except Exception as exc:
        print(f"Note/Warning on repo access/creation: {exc}")

    local_files = [f for f in checkpoint_dir.iterdir() if f.is_file() and f.suffix in {".pt", ".pth", ".json"}]
    if not local_files:
        print(f"No checkpoint files found in '{checkpoint_dir}'.")
        return

    print(f"Uploading {len(local_files)} files from '{checkpoint_dir}' to '{repo_id}'...")
    for file_path in sorted(local_files):
        print(f"  Uploading {file_path.name} ({file_path.stat().st_size / (1024**2):.2f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=f"checkpoints/{file_path.name}",
                repo_id=repo_id,
                token=token,
            )
            print(f"  ✓ Uploaded {file_path.name}")
        except Exception as exc:
            print(f"  ✗ Failed to upload {file_path.name}: {exc}")

    print("\nCheckpoint upload process completed.")


def download_checkpoints(
    repo_id: str,
    target_dir: Path,
    token: str | None = None,
) -> None:
    """Download checkpoints from Hugging Face repository to local directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoints from '{repo_id}' to '{target_dir}'...")

    try:
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=["*.pt", "*.pth", "checkpoints/*"],
            local_dir=target_dir,
        )
        print(f"✓ All checkpoints downloaded to: {downloaded_dir}")
    except Exception as exc:
        print(f"Snapshot download encountered an issue ({exc}). Attempting individual file downloads...")
        api = HfApi(token=token)
        try:
            repo_files = api.list_repo_files(repo_id=repo_id)
        except Exception as list_err:
            print(f"Error accessing repo files: {list_err}")
            return

        for fname in CHECKPOINT_FILES:
            matching = [f for f in repo_files if f.endswith(fname)]
            if not matching:
                continue
            path_in_repo = matching[0]
            print(f"  Downloading {path_in_repo}...")
            try:
                dest = hf_hub_download(
                    repo_id=repo_id,
                    filename=path_in_repo,
                    local_dir=target_dir,
                    token=token,
                )
                print(f"  ✓ Saved to {dest}")
            except Exception as dl_err:
                print(f"  ✗ Failed to download {path_in_repo}: {dl_err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload or download model checkpoints to/from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Local checkpoint directory (default: outputs/checkpoints)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (can also be set via HF_TOKEN env var)",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["auto", "upload", "download"],
        default="auto",
        help="Action to perform: 'upload', 'download', or 'auto' (default: auto)",
    )

    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    if args.action == "upload":
        upload_checkpoints(args.repo_id, checkpoint_dir, token=args.token)
    elif args.action == "download":
        download_checkpoints(args.repo_id, checkpoint_dir, token=args.token)
    else:
        # Auto mode: if checkpoints exist locally, upload them; otherwise download
        has_local_checkpoints = (
            checkpoint_dir.exists()
            and any(f.suffix in {".pt", ".pth"} for f in checkpoint_dir.iterdir() if f.is_file())
        )
        if has_local_checkpoints:
            print(f"Local checkpoints found in '{checkpoint_dir}'. Running upload...")
            upload_checkpoints(args.repo_id, checkpoint_dir, token=args.token)
        else:
            print(f"No local checkpoints found in '{checkpoint_dir}'. Running download...")
            download_checkpoints(args.repo_id, checkpoint_dir, token=args.token)


if __name__ == "__main__":
    main()
