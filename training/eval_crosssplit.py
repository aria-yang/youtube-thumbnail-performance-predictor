"""
Prompt 11 — Cross-Split Generalization Evaluation
Evaluates the trained FusionMLP classification model on all three splits:
  - Random split
  - Channel-heldout split
  - Time-based split

Outputs:
  - AUROC + F1 comparison table (CSV + printed)
  - Bar chart comparing AUROC across splits
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset, read_csv_with_fallback
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from training.train_fusion import (
    compute_auroc,
    compute_macro_f1,
    load_saved_split_ids,
    restore_artifacts,
)


DEFAULT_CSV_PATH = PROCESSED_DATA_DIR / "merged_labeled_data.csv"
DEFAULT_TEXT_PATH = PROCESSED_DATA_DIR / "text_embeddings.npy"
DEFAULT_FACE_PATH = PROCESSED_DATA_DIR / "face_embeddings.npy"
DEFAULT_CHECKPOINT = MODELS_DIR / "fusion_mlp.pt"


def resolve_cnn_path(override: Path | None) -> Path:
    if override is not None:
        return override
    candidates = [
        PROCESSED_DATA_DIR / "merged_cnn_embeddings_resnet50.npy",
        PROCESSED_DATA_DIR / "cnn_embeddings.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def evaluate_split(
    model: FusionMLP,
    dataset: ThumbnailDataset,
    split_df: pd.DataFrame,
    split_dir: Path,
    split_name: str,
    batch_size: int,
    device: str,
) -> dict:
    """Load test indices for a split and evaluate the model on them."""
    try:
        _, _, test_ids = load_saved_split_ids(split_dir, split_name)
    except FileNotFoundError:
        print(f"  [SKIP] Split files not found for '{split_name}' in {split_dir}")
        return None

    id_to_idx = {vid: idx for idx, vid in enumerate(split_df["Id"])}
    test_indices = [id_to_idx[vid] for vid in split_df["Id"] if vid in test_ids]

    if not test_indices:
        print(f"  [SKIP] No test indices found for split '{split_name}'")
        return None

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    auroc = compute_auroc(model, test_loader, num_classes=5, device=device)
    f1 = compute_macro_f1(model, test_loader, device=device)

    print(f"  {split_name:12s} | n_test={len(test_indices):5d} | AUROC={auroc:.4f} | F1={f1:.4f}")
    return {"split": split_name, "n_test": len(test_indices), "AUROC": auroc, "F1": f1}


def plot_cross_split(results: list[dict], output_path: Path) -> None:
    splits = [r["split"] for r in results]
    aurocs = [r["AUROC"] for r in results]
    f1s = [r["F1"] for r in results]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, aurocs, width, label="AUROC", color="#2196F3", edgecolor="black")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#FF9800", edgecolor="black")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in splits], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Cross-Split Generalization: AUROC and Macro F1", fontsize=13, fontweight="bold")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random chance (0.5)")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained FusionMLP on random, channel, and time-based splits."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--cnn_path", type=Path, default=None)
    parser.add_argument("--text_path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--face_path", type=Path, default=DEFAULT_FACE_PATH)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--splits", type=str, nargs="+", default=["random", "channel", "time"],
                        help="Which splits to evaluate. Default: random channel time")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("/content/drive/MyDrive/ECE324/youtube-thumbnail-performance-predictor-artifacts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device

    cnn_path = resolve_cnn_path(args.cnn_path)

    # Optionally restore from Drive
    artifact_root = (
        args.artifact_root
        if args.artifact_root.exists() or "drive" in str(args.artifact_root).lower()
        else None
    )
    restore_artifacts(
        [args.csv_path, cnn_path, args.text_path, args.face_path, args.checkpoint_path]
        + [args.split_dir / f"{s}_{t}.csv" for s in args.splits for t in ("train", "val", "test")],
        artifact_root,
        overwrite=False,
    )

    # Load dataset
    dataset = ThumbnailDataset(
        csv_path=args.csv_path,
        cnn_path=cnn_path,
        text_path=args.text_path,
        face_path=args.face_path,
    )
    split_df = read_csv_with_fallback(args.csv_path)
    split_df["Id"] = split_df["Id"].astype(str)

    # Load model
    cnn_dim = int(np.load(cnn_path, mmap_mode="r").shape[1])
    text_dim = int(np.load(args.text_path, mmap_mode="r").shape[1])
    face_dim = int(np.load(args.face_path, mmap_mode="r").shape[1])

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        hidden1=512,
        hidden2=256,
        num_classes=5,
        dropout_p=0.4,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluate on each split
    print(f"\n{'='*55}")
    print(f"Cross-Split Generalization Evaluation")
    print(f"Checkpoint: {args.checkpoint_path.name}")
    print(f"Device: {device}")
    print(f"{'='*55}")

    results = []
    for split_name in args.splits:
        result = evaluate_split(
            model=model,
            dataset=dataset,
            split_df=split_df,
            split_dir=args.split_dir,
            split_name=split_name,
            batch_size=args.batch_size,
            device=device,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("No splits were successfully evaluated. Check that split CSVs exist.")
        return

    # Save table
    table = pd.DataFrame(results)
    table_path = args.output_dir / "cross_split_generalization.csv"
    table.to_csv(table_path, index=False)

    print(f"\n{'='*55}")
    print("Cross-Split Results Table:")
    print(table.to_string(index=False))
    print(f"\nSaved table -> {table_path}")

    # Plot
    plot_path = args.output_dir / "cross_split_auroc_f1.png"
    plot_cross_split(results, plot_path)

    # Save JSON summary
    json_path = args.output_dir / "cross_split_generalization.json"
    json_path.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint_path), "results": results}, indent=2
    ))
    print(f"Saved JSON summary -> {json_path}")

    # Print interpretation hint
    if len(results) >= 2:
        best = max(results, key=lambda r: r["AUROC"])
        worst = min(results, key=lambda r: r["AUROC"])
        drop = best["AUROC"] - worst["AUROC"]
        print(f"\nNote: AUROC drops {drop:.4f} from '{best['split']}' → '{worst['split']}'. "
              f"{'Large drop suggests limited generalization beyond seen channels/time windows.' if drop > 0.05 else 'Small drop suggests reasonable generalization.'}")


if __name__ == "__main__":
    main()