import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Subset

from thumbnail_performance.config import DATA_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset, read_csv_with_fallback
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from training.train_fusion_regression import (
    DEFAULT_CSV_PATH,
    DEFAULT_FACE_PATH,
    DEFAULT_TEXT_PATH,
    build_regression_loss,
    evaluate_regression,
    load_saved_split_ids,
    resolve_cnn_path,
    restore_artifacts,
    set_seed,
    train_regression,
)


class AblationWrapper(nn.Module):
    def __init__(self, base_model: FusionMLP, use_text: bool = True, use_face: bool = True):
        super().__init__()
        self.base_model = base_model
        self.use_text = use_text
        self.use_face = use_face

    def forward(self, cnn_feat, text_feat, face_feat):
        if not self.use_text:
            text_feat = torch.zeros_like(text_feat)
        if not self.use_face:
            face_feat = torch.zeros_like(face_feat)
        return self.base_model(cnn_feat, text_feat, face_feat)


def run_ablation_experiment(args: argparse.Namespace) -> pd.DataFrame:
    artifact_root = (
        args.artifact_root
        if args.artifact_root.exists() or "drive" in str(args.artifact_root).lower()
        else None
    )
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    cnn_path = resolve_cnn_path(args.cnn_path)
    target_transform = None if args.target_transform == "none" else args.target_transform
    restore_artifacts(
        [
            args.csv_path,
            cnn_path,
            args.text_path,
            args.face_path,
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
    train_ids, val_ids, _ = load_saved_split_ids(args.split_dir, args.split_name)
    id_to_idx = {video_id: idx for idx, video_id in enumerate(split_df["Id"])}
    train_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in train_ids]
    val_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in val_ids]

    cnn_dim = int(np.load(cnn_path, mmap_mode="r").shape[1])
    text_dim = int(np.load(args.text_path, mmap_mode="r").shape[1])
    face_dim = int(np.load(args.face_path, mmap_mode="r").shape[1])

    configs = [
        {"name": "CNN-only", "use_text": False, "use_face": False},
        {"name": "CNN + Text", "use_text": True, "use_face": False},
        {"name": "CNN + Face", "use_text": False, "use_face": True},
        {"name": "CNN + Text + Face", "use_text": True, "use_face": True},
    ]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    results = []

    logger.info(
        f"Starting regression ablation study over {len(seeds)} seeds on {device.upper()} "
        f"using split '{args.split_name}'."
    )

    for config in configs:
        logger.info(f"Evaluating configuration: {config['name']}")
        for seed in seeds:
            set_seed(seed)
            train_generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=args.batch_size,
                shuffle=True,
                generator=train_generator,
            )
            val_loader = DataLoader(
                Subset(dataset, val_indices),
                batch_size=args.batch_size,
                shuffle=False,
            )

            base_model = FusionMLP(
                cnn_dim=cnn_dim,
                text_dim=text_dim,
                face_dim=face_dim,
                hidden1=args.hidden1,
                hidden2=args.hidden2,
                num_classes=1,
                dropout_p=args.dropout,
            )
            model = AblationWrapper(
                base_model=base_model,
                use_text=config["use_text"],
                use_face=config["use_face"],
            ).to(device)

            history = train_regression(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.num_epochs,
                lr=args.lr,
                loss_name=args.loss,
                device=device,
            )
            criterion = build_regression_loss(args.loss)
            val_loss, val_metrics, _, _ = evaluate_regression(model, val_loader, criterion, device)

            results.append(
                {
                    "model": config["name"],
                    "seed": seed,
                    "epochs_ran": len(history["train_loss"]),
                    "val_loss": val_loss,
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_spearman": val_metrics["spearman"],
                }
            )
            logger.info(
                f"--> {config['name']} | seed={seed} | "
                f"MAE={val_metrics['mae']:.4f} | RMSE={val_metrics['rmse']:.4f} | "
                f"R2={val_metrics['r2']:.4f} | Spearman={val_metrics['spearman']:.4f}"
            )

    return pd.DataFrame(results)


def generate_ablation_outputs(
    df_results: pd.DataFrame,
    output_dir: Path,
    ranking_metric: str,
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    summary = (
        df_results.groupby("model")[["val_mae", "val_rmse", "val_r2", "val_spearman"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    summary = summary.sort_values(
        f"{ranking_metric}_mean",
        ascending=ranking_metric in {"val_mae", "val_rmse"},
    ).reset_index(drop=True)

    summary_path = output_dir / "ablation_regression_summary.csv"
    raw_path = output_dir / "ablation_regression_all_runs.csv"
    plot_path = output_dir / f"ablation_regression_{ranking_metric}.png"
    df_results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_results,
        x="model",
        y=ranking_metric,
        order=summary["model"],
        palette="Greens",
        showfliers=False,
    )
    sns.stripplot(
        data=df_results,
        x="model",
        y=ranking_metric,
        order=summary["model"],
        color="black",
        alpha=0.6,
        jitter=True,
    )
    plt.title("Regression Multimodal Ablation Study", fontsize=14, pad=15)
    plt.ylabel(ranking_metric.replace("_", " ").title(), fontsize=12)
    plt.xlabel("Modality Configuration", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.success(f"Saved raw ablation runs to {raw_path}")
    logger.success(f"Saved ablation summary to {summary_path}")
    logger.success(f"Saved ablation plot to {plot_path}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multimodal regression ablation study."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--cnn_path", type=Path, default=None)
    parser.add_argument("--text_path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--face_path", type=Path, default=DEFAULT_FACE_PATH)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--split_name", type=str, default="random", choices=["random", "channel", "time"])
    parser.add_argument("--target_column", type=str, default="normalized_performance")
    parser.add_argument("--target_transform", type=str, default="log1p", choices=["log1p", "none"])
    parser.add_argument("--loss", type=str, default="smoothl1", choices=["smoothl1", "mse", "l1"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument(
        "--ranking_metric",
        type=str,
        default="val_spearman",
        choices=["val_mae", "val_rmse", "val_r2", "val_spearman"],
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("/content/drive/MyDrive/ECE324/youtube-thumbnail-performance-predictor-artifacts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_results = run_ablation_experiment(args)
    generate_ablation_outputs(df_results, output_dir=args.output_dir, ranking_metric=args.ranking_metric)


if __name__ == "__main__":
    main()
