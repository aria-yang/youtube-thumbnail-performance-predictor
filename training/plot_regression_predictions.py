import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset, read_csv_with_fallback
from thumbnail_performance.interpretability import (
    get_regression_inverse_stats,
    invert_regression_prediction,
    resolve_regression_target_mode,
)
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from training.train_fusion_regression import (
    DEFAULT_CSV_PATH,
    DEFAULT_FACE_PATH,
    DEFAULT_TEXT_PATH,
    compute_regression_metrics,
    load_saved_split_ids,
    restore_artifacts,
    set_seed,
)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUTS_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUTS_METRICS_DIR = OUTPUTS_DIR / "metrics"


def resolve_matching_cnn_path(explicit_path: Path | None, csv_path: Path) -> Path:
    expected_rows = len(read_csv_with_fallback(csv_path))

    def row_count(path: Path) -> int:
        return int(np.load(path, mmap_mode="r").shape[0])

    if explicit_path is not None:
        actual_rows = row_count(explicit_path)
        if actual_rows != expected_rows:
            raise ValueError(
                f"CNN embedding row count mismatch for {explicit_path}: "
                f"{actual_rows} rows vs {expected_rows} rows in {csv_path}."
            )
        return explicit_path

    candidates = [
        PROCESSED_DATA_DIR / "merged_cnn_embeddings_resnet50.npy",
        PROCESSED_DATA_DIR / "merged_cnn_embeddings.npy",
        PROCESSED_DATA_DIR / "cnn_embeddings.npy",
    ]

    checked = []
    for candidate in candidates:
        if candidate.exists():
            actual_rows = row_count(candidate)
            checked.append((candidate, actual_rows))
            if actual_rows == expected_rows:
                return candidate

    checked_text = ", ".join(f"{path.name}={rows}" for path, rows in checked) or "no candidate files found"
    raise ValueError(
        f"Could not find a CNN embedding file whose row count matches {csv_path.name} "
        f"({expected_rows} rows). Checked: {checked_text}. "
        "Pass --cnn_path explicitly with the matching embedding file."
    )


def collect_predictions(
    model: FusionMLP,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    all_preds = []

    with torch.no_grad():
        for cnn_feat, text_feat, face_feat, _targets in loader:
            cnn_feat = cnn_feat.to(device)
            text_feat = text_feat.to(device)
            face_feat = face_feat.to(device)
            preds = model(cnn_feat, text_feat, face_feat).squeeze(-1)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def invert_array(
    values: np.ndarray,
    target_mode: str,
    zscore_mean: float | None,
    zscore_std: float | None,
) -> np.ndarray:
    return np.array(
        [
            max(
                0.0,
                invert_regression_prediction(
                    raw_prediction=float(value),
                    target_mode=target_mode,
                    zscore_mean=zscore_mean,
                    zscore_std=zscore_std,
                ),
            )
            for value in values
        ],
        dtype=np.float64,
    )


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    split_name: str,
    metrics: dict[str, float],
    use_log1p_axes: bool,
    density_gridsize: int,
    metrics_label: str,
) -> None:
    x_plot = np.log1p(y_true) if use_log1p_axes else y_true
    y_plot = np.log1p(y_pred) if use_log1p_axes else y_pred

    min_val = float(min(x_plot.min(), y_plot.min()))
    max_val = float(max(x_plot.max(), y_plot.max()))
    padding = 0.04 * (max_val - min_val) if max_val > min_val else 0.05
    lo = max(0.0, min_val - padding)
    hi = max_val + padding

    fig, ax_main = plt.subplots(figsize=(7.3, 6.4))

    ax_main.scatter(
        x_plot,
        y_plot,
        s=18,
        alpha=0.32,
        color="#1f77b4",
        edgecolors="none",
    )

    ax_main.plot([lo, hi], [lo, hi], linestyle="--", color="#c62828", linewidth=2.0, label="Perfect prediction")
    if len(x_plot) >= 2:
        slope, intercept = np.polyfit(x_plot, y_plot, 1)
        fit_x = np.array([lo, hi], dtype=np.float64)
        fit_y = slope * fit_x + intercept
        ax_main.plot(fit_x, fit_y, color="#0d47a1", linewidth=2.0, label="Best fit")

    axis_label_suffix = " (log1p scale)" if use_log1p_axes else ""
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel(f"True normalized performance{axis_label_suffix}", fontsize=12)
    ax_main.set_ylabel(f"Predicted normalized performance{axis_label_suffix}", fontsize=12)
    ax_main.set_title("Regression Predictions vs Ground Truth", fontsize=13, fontweight="bold")

    ax_main.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)

    if use_log1p_axes:
        ax_main.text(
            0.98,
            0.03,
            metrics_label,
            transform=ax_main.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
            color="#555555",
        )

    ax_main.grid(alpha=0.18)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regression predictions vs. ground truth for the tuned regression model."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--cnn_path", type=Path, default=None)
    parser.add_argument("--text_path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--face_path", type=Path, default=DEFAULT_FACE_PATH)
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--split_name", type=str, default="random", choices=["random", "channel", "time"])
    parser.add_argument(
        "--split_part",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which subset of the saved split to plot.",
    )
    parser.add_argument("--target_column", type=str, default="normalized_performance")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=MODELS_DIR / "fusion_mlp_regression_final_seed42.pt",
        help="Path to the tuned/final regression checkpoint.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="log1p",
        help="Regression target transform to invert when plotting predictions.",
    )
    parser.add_argument(
        "--tuning_summary_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "fusion_regression_tuning_summary.json",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--use_log1p_axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot log1p-transformed axes to make heavy-tailed normalized performance easier to read.",
    )
    parser.add_argument(
        "--density_gridsize",
        type=int,
        default=35,
        help="Hexbin grid size for the density plot.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=OUTPUTS_FIGURES_DIR / "figure3_regression_predictions_vs_ground_truth.png",
    )
    parser.add_argument(
        "--predictions_csv_path",
        type=Path,
        default=OUTPUTS_TABLES_DIR / "figure3_regression_predictions_vs_ground_truth.csv",
    )
    parser.add_argument(
        "--metrics_json_path",
        type=Path,
        default=OUTPUTS_METRICS_DIR / "figure3_regression_predictions_vs_ground_truth_metrics.json",
    )
    parser.add_argument(
        "--display_metrics_from",
        type=str,
        default="transformed",
        choices=["raw", "transformed", "file"],
        help="Which metric values to display in the figure annotation box.",
    )
    parser.add_argument(
        "--display_metrics_file",
        type=Path,
        default=PROCESSED_DATA_DIR / "fusion_mlp_regression_final_seed42_metrics.json",
        help="Metrics JSON to read when --display_metrics_from=file.",
    )
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("/content/drive/MyDrive/ECE324/youtube-thumbnail-performance-predictor-artifacts"),
    )
    return parser.parse_args()


def resolve_display_metrics(
    display_source: str,
    metrics_raw: dict[str, float],
    metrics_transformed: dict[str, float],
    display_file: Path,
) -> tuple[dict[str, float], str]:
    if display_source == "raw":
        return metrics_raw, "Metrics on raw scale"
    if display_source == "transformed":
        return metrics_transformed, "Metrics on log1p scale"
    payload: dict[str, Any] = json.loads(display_file.read_text())
    file_metrics = payload.get("test_metrics") or payload.get("metrics_on_raw_normalized_performance")
    if not isinstance(file_metrics, dict):
        raise ValueError(f"Could not find a metrics dictionary in {display_file}")
    normalized_metrics = {
        "mae": float(file_metrics["mae"]),
        "rmse": float(file_metrics["rmse"]),
        "r2": float(file_metrics["r2"]),
        "spearman": float(file_metrics["spearman"]),
    }
    return normalized_metrics, f"Metrics from {display_file.name}"


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
    cnn_path = resolve_matching_cnn_path(args.cnn_path, args.csv_path)
    target_mode, _ = resolve_regression_target_mode(
        explicit_mode=args.target_mode,
        tuning_summary_path=args.tuning_summary_path,
    )
    zscore_mean, zscore_std = get_regression_inverse_stats(
        target_mode=target_mode,
        csv_path=args.csv_path,
        split_dir=args.split_dir,
        split_name=args.split_name,
        target_column=args.target_column,
    )

    restore_artifacts(
        [
            args.csv_path,
            cnn_path,
            args.text_path,
            args.face_path,
            args.checkpoint_path,
            args.tuning_summary_path,
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
        target_transform="log1p" if "log1p" in target_mode else "none",
    )
    split_df = read_csv_with_fallback(args.csv_path)
    split_df["Id"] = split_df["Id"].astype(str)

    train_ids, val_ids, test_ids = load_saved_split_ids(args.split_dir, args.split_name)
    ids_by_split = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    selected_ids = ids_by_split[args.split_part]
    id_to_idx = {video_id: idx for idx, video_id in enumerate(split_df["Id"])}
    selected_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in selected_ids]

    if not selected_indices:
        raise ValueError(
            f"Saved split '{args.split_name}' produced an empty '{args.split_part}' subset."
        )

    loader = DataLoader(
        Subset(dataset, selected_indices),
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

    pred_transformed = collect_predictions(model, loader, device)
    pred_raw = invert_array(pred_transformed, target_mode, zscore_mean, zscore_std)
    target_raw = (
        split_df.loc[selected_indices, args.target_column]
        .astype(float)
        .to_numpy(dtype=np.float64)
    )
    metrics_raw = compute_regression_metrics(pred_raw, target_raw)
    pred_transformed_for_metrics = np.log1p(pred_raw) if target_mode == "log1p" else pred_transformed
    target_transformed_for_metrics = np.log1p(target_raw) if target_mode == "log1p" else target_raw
    metrics_transformed = compute_regression_metrics(
        pred_transformed_for_metrics,
        target_transformed_for_metrics,
    )
    display_metrics, metrics_label = resolve_display_metrics(
        display_source=args.display_metrics_from,
        metrics_raw=metrics_raw,
        metrics_transformed=metrics_transformed,
        display_file=args.display_metrics_file,
    )

    args.predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {
            "true_normalized_performance": target_raw,
            "predicted_normalized_performance": pred_raw,
            "split_name": args.split_name,
            "split_part": args.split_part,
        }
    )
    predictions_df.to_csv(args.predictions_csv_path, index=False)

    summary = {
        "split_name": args.split_name,
        "split_part": args.split_part,
        "checkpoint_path": str(args.checkpoint_path),
        "target_mode": target_mode,
        "num_examples": int(len(predictions_df)),
        "metrics_on_raw_normalized_performance": metrics_raw,
        "metrics_on_target_transform_scale": metrics_transformed,
        "display_metrics_source": args.display_metrics_from,
        "display_metrics": display_metrics,
    }
    args.metrics_json_path.write_text(json.dumps(summary, indent=2))

    plot_predictions(
        y_true=target_raw,
        y_pred=pred_raw,
        output_path=args.output_path,
        split_name=f"{args.split_name}/{args.split_part}",
        metrics=display_metrics,
        use_log1p_axes=args.use_log1p_axes,
        density_gridsize=args.density_gridsize,
        metrics_label=metrics_label,
    )

    print(f"Saved Figure 3 plot to {args.output_path}")
    print(f"Saved prediction table to {args.predictions_csv_path}")
    print(f"Saved metric summary to {args.metrics_json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
