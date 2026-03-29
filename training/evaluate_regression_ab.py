import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset, read_csv_with_fallback
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from training.train_fusion_regression import (
    DEFAULT_CSV_PATH,
    DEFAULT_FACE_PATH,
    DEFAULT_TEXT_PATH,
    compute_regression_metrics,
    load_saved_split_ids,
    resolve_cnn_path,
    restore_artifacts,
    set_seed,
)


def collect_predictions(
    model: FusionMLP,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for cnn_feat, text_feat, face_feat, targets in loader:
            cnn_feat = cnn_feat.to(device)
            text_feat = text_feat.to(device)
            face_feat = face_feat.to(device)
            preds = model(cnn_feat, text_feat, face_feat).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_pairwise_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
    tie_epsilon: float,
) -> dict[str, float | int]:
    n = len(predictions)
    all_pairs = n * (n - 1) // 2

    if n < 2:
        raise ValueError("Need at least 2 examples in the test split for pairwise evaluation.")

    if all_pairs <= max_pairs:
        left_idx, right_idx = np.triu_indices(n, k=1)
    else:
        left_idx = rng.integers(0, n, size=max_pairs)
        right_idx = rng.integers(0, n - 1, size=max_pairs)
        right_idx = np.where(right_idx >= left_idx, right_idx + 1, right_idx)

    pred_diff = predictions[left_idx] - predictions[right_idx]
    target_diff = targets[left_idx] - targets[right_idx]

    valid_mask = np.abs(target_diff) > tie_epsilon
    pred_sign = np.sign(pred_diff[valid_mask])
    target_sign = np.sign(target_diff[valid_mask])

    evaluated_pairs = int(valid_mask.sum())
    if evaluated_pairs == 0:
        raise ValueError(
            "All sampled target pairs were ties under the current tie_epsilon. "
            "Lower tie_epsilon or use a larger test set."
        )

    correct = int((pred_sign == target_sign).sum())
    pairwise_accuracy = correct / evaluated_pairs

    return {
        "pairwise_accuracy": float(pairwise_accuracy),
        "pairs_considered": int(len(left_idx)),
        "pairs_evaluated": evaluated_pairs,
        "pairs_skipped_as_ties": int(len(left_idx) - evaluated_pairs),
        "used_all_pairs": bool(all_pairs <= max_pairs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a regression model for offline A/B-style ranking on the test split."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--cnn_path", type=Path, default=None)
    parser.add_argument("--text_path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--face_path", type=Path, default=DEFAULT_FACE_PATH)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--split_name", type=str, default="random", choices=["random", "channel", "time"])
    parser.add_argument("--target_column", type=str, default="normalized_performance")
    parser.add_argument("--target_transform", type=str, default="log1p", choices=["log1p", "none"])
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=MODELS_DIR / "fusion_mlp_regression.pt",
        help="Path to a trained regression checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pairs", type=int, default=100000)
    parser.add_argument(
        "--tie_epsilon",
        type=float,
        default=1e-8,
        help="Pairs whose true target difference is within this threshold are skipped.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "fusion_mlp_regression_ab_test_metrics.json",
        help="Where to save the offline A/B-style evaluation results.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("/content/drive/MyDrive/ECE324/youtube-thumbnail-performance-predictor-artifacts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = (
        args.artifact_root
        if args.artifact_root.exists() or "drive" in str(args.artifact_root).lower()
        else None
    )
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cnn_path = resolve_cnn_path(args.cnn_path)
    target_transform = None if args.target_transform == "none" else args.target_transform

    restore_artifacts(
        [
            args.csv_path,
            cnn_path,
            args.text_path,
            args.face_path,
            args.checkpoint_path,
            args.split_dir / f"{args.split_name}_train.csv",
            args.split_dir / f"{args.split_name}_val.csv",
            args.split_dir / f"{args.split_name}_test.csv",
        ],
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
    _, _, test_ids = load_saved_split_ids(args.split_dir, args.split_name)
    id_to_idx = {video_id: idx for idx, video_id in enumerate(split_df["Id"])}
    test_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in test_ids]

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )

    cnn_dim = int(np.load(cnn_path, mmap_mode="r").shape[1])
    text_dim = int(np.load(args.text_path, mmap_mode="r").shape[1])
    face_dim = int(np.load(args.face_path, mmap_mode="r").shape[1])

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
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

    predictions, transformed_targets = collect_predictions(model, test_loader, device)
    test_metrics = compute_regression_metrics(predictions, transformed_targets)

    raw_targets = split_df.loc[test_indices, args.target_column].astype(float).to_numpy()
    pairwise_metrics = compute_pairwise_accuracy(
        predictions=predictions,
        targets=raw_targets,
        rng=rng,
        max_pairs=args.max_pairs,
        tie_epsilon=args.tie_epsilon,
    )

    output = {
        "split_name": args.split_name,
        "target_column": args.target_column,
        "target_transform": args.target_transform,
        "seed": args.seed,
        "device": device,
        "num_test_examples": len(test_indices),
        "standard_test_metrics_on_transformed_target": test_metrics,
        "offline_ab_pairwise_metrics_on_raw_target": pairwise_metrics,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2))

    print(f"Saved offline A/B-style evaluation to {args.output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
