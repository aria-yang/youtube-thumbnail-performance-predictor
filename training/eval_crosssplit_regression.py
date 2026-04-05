"""
Prompt 11 — Cross-Split Generalization Evaluation (Regression Model)
Evaluates the trained regression FusionMLP on all three splits:
  - Random split
  - Channel-heldout split
  - Time-based split

Metrics: MAE, RMSE, R2, Spearman
Outputs:
  - Comparison table (CSV + printed)
  - Bar chart comparing Spearman across splits
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
from training.train_fusion_regression import (
    compute_regression_metrics,
    collect_predictions,
    load_saved_split_ids,
    resolve_cnn_path,
    restore_artifacts,
    set_seed,
)


DEFAULT_CSV_PATH = PROCESSED_DATA_DIR / "merged_labeled_data.csv"
DEFAULT_TEXT_PATH = PROCESSED_DATA_DIR / "merged_text_embeddings.npy"
DEFAULT_FACE_PATH = PROCESSED_DATA_DIR / "merged_face_embeddings.npy"
DEFAULT_CHECKPOINT = MODELS_DIR / "fusion_mlp_regression.pt"


def evaluate_split(
    model: FusionMLP,
    dataset: ThumbnailDataset,
    split_df: pd.DataFrame,
    split_dir: Path,
    split_name: str,
    batch_size: int,
    device: str,
) -> dict | None:
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

    preds, targets = collect_predictions(model, test_loader, device)
    metrics = compute_regression_metrics(preds, targets)

    print(
        f"  {split_name:12s} | n_test={len(test_indices):5d} | "
        f"MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
        f"R2={metrics['r2']:.4f} | Spearman={metrics['spearman']:.4f}"
    )
    return {"split": split_name, "n_test": len(test_indices), **metrics}


def plot_cross_split(results: list[dict], output_path: Path) -> None:
    splits = [r["split"] for r in results]
    metrics = ["mae", "rmse", "r2", "spearman"]
    colors = ["#F44336", "#FF9800", "#4CAF50", "#2196F3"]
    labels = ["MAE", "RMSE", "R²", "Spearman"]

    x = np.arange(len(splits))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.replace("_", "\n") for s in splits], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Cross-Split Generalization: Regression Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the regression FusionMLP on random, channel, and time-based splits."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--cnn_path", type=Path, default=None)
    parser.add_argument("--text_path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--face_path", type=Path, default=DEFAULT_FACE_PATH)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--splits", type=str, nargs="+", default=["random", "channel", "time"])
    parser.add_argument("--target_column", type=str, default="normalized_performance")
    parser.add_argument("--target_transform", type=str, default="log1p", choices=["log1p", "none"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("/content/drive/MyDrive/youtube-thumbnail-performance-predictor-artifacts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device

    set_seed(args.seed)
    cnn_path = resolve_cnn_path(args.cnn_path)
    target_transform = None if args.target_transform == "none" else args.target_transform

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

    dataset = ThumbnailDataset(
        csv_path=args.csv_path,
        cnn_path=cnn_path,
        text_path=args.text_path,
        face_path=args.face_path,
        target_column=args.target_column,
        target_transform=target_transform,
    )
    split_df = read_csv_with_fallback(args.csv_path)
    split_df["Id"] = split_df["Id"].astype(str)

    cnn_dim = int(np.load(cnn_path, mmap_mode="r").shape[1])
    text_dim = int(np.load(args.text_path, mmap_mode="r").shape[1])
    face_dim = int(np.load(args.face_path, mmap_mode="r").shape[1])

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        hidden1=checkpoint.get("hidden1", 512) if isinstance(checkpoint, dict) else 512,
        hidden2=checkpoint.get("hidden2", 256) if isinstance(checkpoint, dict) else 256,
        dropout_p=checkpoint.get("dropout_p", 0.4) if isinstance(checkpoint, dict) else 0.4,
        num_classes=1,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\n{'=' * 60}")
    print(f"Cross-Split Generalization Evaluation (Regression)")
    print(f"Checkpoint: {args.checkpoint_path.name}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

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

    table = pd.DataFrame(results)
    table_path = args.output_dir / "cross_split_regression.csv"
    table.to_csv(table_path, index=False)

    print(f"\n{'=' * 60}")
    print("Cross-Split Regression Results:")
    print(table.to_string(index=False))
    print(f"\nSaved table -> {table_path}")

    plot_path = args.output_dir / "cross_split_regression.png"
    plot_cross_split(results, plot_path)

    json_path = args.output_dir / "cross_split_regression.json"
    json_path.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint_path), "results": results}, indent=2
    ))
    print(f"Saved JSON -> {json_path}")

    if len(results) >= 2:
        best = max(results, key=lambda r: r["spearman"])
        worst = min(results, key=lambda r: r["spearman"])
        drop = best["spearman"] - worst["spearman"]
        print(
            f"\nNote: Spearman drops {drop:.4f} from '{best['split']}' → '{worst['split']}'. "
            f"{'Large drop suggests limited generalization.' if drop > 0.05 else 'Small drop suggests good generalization.'}"
        )


if __name__ == "__main__":
    main()
