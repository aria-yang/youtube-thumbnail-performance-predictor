from training.train_fusion_regression import (
    DEFAULT_CSV_PATH,
    DEFAULT_FACE_PATH,
    DEFAULT_TEXT_PATH,
    build_regression_loss,
    evaluate_regression,
    load_saved_split_ids,
    resolve_cnn_path,
    restore_artifacts,
    sync_artifacts_to_root,
    train_regression,
)
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from thumbnail_performance.dataset import read_csv_with_fallback
from thumbnail_performance.config import DATA_DIR, PROCESSED_DATA_DIR
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch
import pandas as pd
import numpy as np
import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_csv_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_hidden_dims(value: str) -> list[tuple[int, int]]:
    dims = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(
                f"Invalid hidden_dims entry '{item}'. Expected values like '512x256'."
            )
        hidden1, hidden2 = item.split("x", maxsplit=1)
        dims.append((int(hidden1), int(hidden2)))
    return dims


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_target_mode(
    values: np.ndarray,
    train_indices: list[int],
    target_mode: str,
) -> np.ndarray:
    arr = values.astype(np.float32).copy()
    normalized_mode = target_mode.lower()

    if normalized_mode in {"log1p", "log1p_zscore", "clip_log1p", "clip_log1p_zscore"}:
        if np.any(arr < 0):
            raise ValueError("Target contains negative values, so log1p-based modes are invalid.")

    if normalized_mode in {"clip_log1p", "clip_log1p_zscore"}:
        clip_value = np.quantile(arr[train_indices], 0.99)
        arr = np.clip(arr, a_min=0.0, a_max=clip_value)

    if normalized_mode in {"log1p", "log1p_zscore", "clip_log1p", "clip_log1p_zscore"}:
        arr = np.log1p(arr)
    elif normalized_mode == "none":
        pass
    else:
        raise ValueError(
            "Unsupported target_mode. Choose from: none, log1p, log1p_zscore, "
            "clip_log1p, clip_log1p_zscore."
        )

    if normalized_mode in {"log1p_zscore", "clip_log1p_zscore"}:
        train_mean = float(arr[train_indices].mean())
        train_std = float(arr[train_indices].std())
        if train_std < 1e-8:
            train_std = 1.0
        arr = (arr - train_mean) / train_std

    return arr.astype(np.float32)


def build_dataset(
    cnn_path: Path,
    text_path: Path,
    face_path: Path,
    targets: np.ndarray,
) -> tuple[TensorDataset, int, int, int]:
    cnn = torch.tensor(np.load(cnn_path), dtype=torch.float32)
    text = torch.tensor(np.load(text_path), dtype=torch.float32)
    face = torch.tensor(np.load(face_path), dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

    dataset = TensorDataset(cnn, text, face, target_tensor)
    return dataset, cnn.shape[1], text.shape[1], face.shape[1]


def rank_and_save_results(
    results_df: pd.DataFrame,
    all_results_path: Path,
    summary_path: Path,
    metric_to_rank: str,
    artifact_root: Path | None,
) -> None:
    all_results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(all_results_path, index=False)

    group_cols = [
        "split_name",
        "target_mode",
        "loss",
        "batch_size",
        "lr",
        "dropout",
        "hidden1",
        "hidden2",
    ]
    metric_cols = [
        "val_loss",
        "val_mae",
        "val_rmse",
        "val_r2",
        "val_spearman",
        "test_loss",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_spearman",
    ]

    summary_df = (
        results_df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary_df.columns
    ]

    rank_column = f"{metric_to_rank}_mean"
    ascending = metric_to_rank.endswith("loss") or metric_to_rank.endswith("mae") or metric_to_rank.endswith("rmse")
    summary_df = summary_df.sort_values(rank_column, ascending=ascending).reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)

    best_row = summary_df.iloc[0].to_dict()
    best_summary = {
        "metric_to_rank": metric_to_rank,
        "best_config": best_row,
        "num_trials": int(len(results_df)),
        "num_unique_configs": int(len(summary_df)),
    }

    json_path = summary_path.with_suffix(".json")
    json_path.write_text(json.dumps(best_summary, indent=2))
    sync_artifacts_to_root([all_results_path, summary_path, json_path], artifact_root)

    print(f"Saved all trial results to {all_results_path}")
    print(f"Saved aggregated summary to {summary_path}")
    print(f"Saved best-config summary to {json_path}")
    print("Best configuration:")
    print(json.dumps(best_summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter sweep for the FusionMLP regression model."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--cnn_path", type=Path, default=None)
    parser.add_argument("--text_path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--face_path", type=Path, default=DEFAULT_FACE_PATH)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument(
        "--split_names",
        type=str,
        default="random",
        help="Comma-separated split names, e.g. random,channel,time",
    )
    parser.add_argument(
        "--target_modes",
        type=str,
        default="log1p,log1p_zscore",
        help="Comma-separated target preprocessing modes.",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default="smoothl1,mse,l1",
        help="Comma-separated regression losses.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="64,128",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--lrs",
        type=str,
        default="0.0003,0.0005,0.001",
        help="Comma-separated learning rates.",
    )
    parser.add_argument(
        "--dropouts",
        type=str,
        default="0.3,0.4,0.5",
        help="Comma-separated dropout values.",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="256x128,512x256,1024x512",
        help="Comma-separated hidden layer pairs like 512x256,1024x512",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--metric_to_rank",
        type=str,
        default="val_spearman",
        choices=[
            "val_loss",
            "val_mae",
            "val_rmse",
            "val_r2",
            "val_spearman",
            "test_loss",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_spearman",
        ],
        help="Metric used to rank configurations in the summary.",
    )
    parser.add_argument(
        "--all_results_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "fusion_regression_tuning_all_results.csv",
        help="CSV file containing one row per trial.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "fusion_regression_tuning_summary.csv",
        help="CSV file containing aggregated config summaries.",
    )
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("/content/drive/MyDrive/ECE324/youtube-thumbnail-performance-predictor-artifacts"),
        help="Directory containing cached/generated artifacts to restore from and sync back to.",
    )
    args = parser.parse_args()

    artifact_root = (
        args.artifact_root
        if args.artifact_root.exists() or "drive" in str(args.artifact_root).lower()
        else None
    )
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    cnn_path = resolve_cnn_path(args.cnn_path)
    split_names = parse_csv_list(args.split_names, str)
    target_modes = parse_csv_list(args.target_modes, str)
    losses = parse_csv_list(args.losses, str)
    batch_sizes = parse_csv_list(args.batch_sizes, int)
    lrs = parse_csv_list(args.lrs, float)
    dropouts = parse_csv_list(args.dropouts, float)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    seeds = parse_csv_list(args.seeds, int)

    split_files = []
    for split_name in split_names:
        split_files.extend(
            [
                args.split_dir / f"{split_name}_train.csv",
                args.split_dir / f"{split_name}_val.csv",
                args.split_dir / f"{split_name}_test.csv",
            ]
        )

    restore_artifacts(
        [args.csv_path, cnn_path, args.text_path, args.face_path, *split_files],
        artifact_root,
        overwrite=False,
    )

    df = read_csv_with_fallback(args.csv_path)
    df["Id"] = df["Id"].astype(str)
    raw_targets = df["normalized_performance"].astype(float).to_numpy()

    results = []
    total_trials = (
        len(split_names)
        * len(target_modes)
        * len(losses)
        * len(batch_sizes)
        * len(lrs)
        * len(dropouts)
        * len(hidden_dims)
        * len(seeds)
    )
    trial_index = 0

    for split_name in split_names:
        train_ids, val_ids, test_ids = load_saved_split_ids(args.split_dir, split_name)
        id_to_idx = {video_id: idx for idx, video_id in enumerate(df["Id"])}
        train_indices = [id_to_idx[video_id] for video_id in df["Id"] if video_id in train_ids]
        val_indices = [id_to_idx[video_id] for video_id in df["Id"] if video_id in val_ids]
        test_indices = [id_to_idx[video_id] for video_id in df["Id"] if video_id in test_ids]

        if not train_indices or not val_indices or not test_indices:
            raise ValueError(
                f"Saved split '{split_name}' produced an empty train/val/test subset."
            )

        for target_mode in target_modes:
            processed_targets = apply_target_mode(raw_targets, train_indices, target_mode)
            dataset, cnn_dim, text_dim, face_dim = build_dataset(
                cnn_path=cnn_path,
                text_path=args.text_path,
                face_path=args.face_path,
                targets=processed_targets,
            )

            for loss, batch_size, lr, dropout, (hidden1, hidden2), seed in itertools.product(
                losses, batch_sizes, lrs, dropouts, hidden_dims, seeds
            ):
                trial_index += 1
                print(
                    f"[{trial_index}/{total_trials}] split={split_name} target={target_mode} "
                    f"loss={loss} batch={batch_size} lr={lr} dropout={dropout} "
                    f"hidden={hidden1}x{hidden2} seed={seed}"
                )

                set_seed(seed)
                train_loader = DataLoader(
                    Subset(dataset, train_indices),
                    batch_size=batch_size,
                    shuffle=True,
                )
                val_loader = DataLoader(
                    Subset(dataset, val_indices),
                    batch_size=batch_size,
                    shuffle=False,
                )
                test_loader = DataLoader(
                    Subset(dataset, test_indices),
                    batch_size=batch_size,
                    shuffle=False,
                )

                model = FusionMLP(
                    cnn_dim=cnn_dim,
                    text_dim=text_dim,
                    face_dim=face_dim,
                    hidden1=hidden1,
                    hidden2=hidden2,
                    dropout_p=dropout,
                    num_classes=1,
                )

                history = train_regression(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=args.num_epochs,
                    lr=lr,
                    loss_name=loss,
                    device=device,
                )

                criterion = build_regression_loss(loss)
                val_loss, val_metrics, _, _ = evaluate_regression(model, val_loader, criterion, device)
                test_loss, test_metrics, _, _ = evaluate_regression(
                    model, test_loader, criterion, device
                )

                results.append(
                    {
                        "split_name": split_name,
                        "target_mode": target_mode,
                        "loss": loss,
                        "batch_size": batch_size,
                        "lr": lr,
                        "dropout": dropout,
                        "hidden1": hidden1,
                        "hidden2": hidden2,
                        "seed": seed,
                        "epochs_ran": len(history["train_loss"]),
                        "best_val_loss_seen": min(history["val_loss"]) if history["val_loss"] else None,
                        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                        "val_loss": val_loss,
                        "val_mae": val_metrics["mae"],
                        "val_rmse": val_metrics["rmse"],
                        "val_r2": val_metrics["r2"],
                        "val_spearman": val_metrics["spearman"],
                        "test_loss": test_loss,
                        "test_mae": test_metrics["mae"],
                        "test_rmse": test_metrics["rmse"],
                        "test_r2": test_metrics["r2"],
                        "test_spearman": test_metrics["spearman"],
                    }
                )

    results_df = pd.DataFrame(results)
    rank_and_save_results(
        results_df=results_df,
        all_results_path=args.all_results_path,
        summary_path=args.summary_path,
        metric_to_rank=args.metric_to_rank,
        artifact_root=artifact_root,
    )


if __name__ == "__main__":
    main()
