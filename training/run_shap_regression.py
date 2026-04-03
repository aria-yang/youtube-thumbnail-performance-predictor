from training.train_fusion_regression import (
    DEFAULT_CSV_PATH,
    DEFAULT_FACE_PATH,
    DEFAULT_TEXT_PATH,
    load_saved_split_ids,
    resolve_cnn_path,
    restore_artifacts,
    set_seed,
    train_regression,
)
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from thumbnail_performance.dataset import ThumbnailDataset, read_csv_with_fallback
from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


TEXT_FEATURE_NAMES = [
    "ocr_word_count",
    "ocr_capital_letter_pct",
    "ocr_has_numeric",
    "ocr_char_count",
]

FACE_FEATURE_NAMES = [
    "num_faces",
    "largest_face_area_ratio",
    "emotion_angry",
    "emotion_disgust",
    "emotion_fear",
    "emotion_happy",
    "emotion_sad",
    "emotion_surprise",
    "emotion_neutral",
    "emotion_unknown",
]


def ensure_shap_installed():
    script_dir = str(Path(__file__).resolve().parent)
    project_root = str(Path(__file__).resolve().parents[1])
    cwd = os.getcwd()
    removed_entries = []

    for entry in ("", script_dir, project_root, cwd):
        while entry in sys.path:
            idx = sys.path.index(entry)
            removed_entries.append((idx, entry))
            sys.path.pop(idx)

    cached_shap = sys.modules.get("shap")
    if cached_shap is not None:
        cached_file = getattr(cached_shap, "__file__", "")
        if cached_file:
            cached_path = Path(cached_file).resolve()
            if project_root in map(str, cached_path.parents) or cached_path == Path(__file__).resolve():
                del sys.modules["shap"]
        else:
            del sys.modules["shap"]

    try:
        shap = importlib.import_module("shap")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'shap' package is required to run this script. Install it in the active "
            "environment, then rerun 'python training/run_shap_regression.py'."
        ) from exc
    finally:
        for idx, entry in reversed(removed_entries):
            sys.path.insert(idx, entry)

    return shap


def build_feature_names(cnn_dim: int, text_dim: int, face_dim: int) -> list[str]:
    cnn_names = [f"cnn_embedding_{idx:03d}" for idx in range(cnn_dim)]

    text_names = TEXT_FEATURE_NAMES[:text_dim]
    if len(text_names) < text_dim:
        text_names.extend([f"text_feature_{idx}" for idx in range(len(text_names), text_dim)])

    face_names = FACE_FEATURE_NAMES[:face_dim]
    if len(face_names) < face_dim:
        face_names.extend([f"face_feature_{idx}" for idx in range(len(face_names), face_dim)])

    return cnn_names + text_names + face_names


class ConcatenatedFusionWrapper(nn.Module):
    def __init__(self, model: FusionMLP, cnn_dim: int, text_dim: int, face_dim: int):
        super().__init__()
        self.model = model
        self.cnn_dim = cnn_dim
        self.text_dim = text_dim
        self.face_dim = face_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = x[:, : self.cnn_dim]
        text_start = self.cnn_dim
        text_end = text_start + self.text_dim
        text_feat = x[:, text_start:text_end]
        face_feat = x[:, text_end: text_end + self.face_dim]
        return self.model(cnn_feat, text_feat, face_feat)


def normalise_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 2:
        return shap_values[:, :, None]
    if shap_values.ndim == 3:
        return shap_values
    raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")


def compute_shap_values(
    model: FusionMLP,
    background_x: np.ndarray,
    explain_x: np.ndarray,
    cnn_dim: int,
    text_dim: int,
    face_dim: int,
    device: str,
) -> np.ndarray:
    shap = ensure_shap_installed()

    wrapped_model = ConcatenatedFusionWrapper(model, cnn_dim, text_dim, face_dim).to(device)
    wrapped_model.eval()

    background_tensor = torch.tensor(background_x, dtype=torch.float32, device=device)
    explain_tensor = torch.tensor(explain_x, dtype=torch.float32, device=device)

    explainer = shap.DeepExplainer(wrapped_model, background_tensor)
    shap_values = explainer.shap_values(explain_tensor)
    return normalise_shap_values(shap_values)


def rank_global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    ranked = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked[["rank", "feature", "mean_abs_shap"]]


def save_global_importance_plot(
    ranked_importance: pd.DataFrame,
    output_path: Path,
    max_display: int,
) -> None:
    top_importance = ranked_importance.head(max_display).iloc[::-1]
    plt.figure(figsize=(10, max(6, 0.35 * len(top_importance))))
    plt.barh(top_importance["feature"], top_importance["mean_abs_shap"], color="#2ca02c")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance for Regression Fusion MLP")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_interpretation_notes(output_path: Path) -> None:
    notes = """# Regression SHAP Interpretation Notes

## How to interpret SHAP values
- A positive SHAP value means the feature pushed the predicted regression score upward on that example.
- A negative SHAP value means the feature pushed the predicted regression score downward on that example.
- Larger absolute SHAP values mean the feature had more influence on the prediction.
- Global importance is computed as mean absolute SHAP value, so it shows which features matter most overall.

## Patterns that would support our thesis
- OCR features ranking near the top would support the claim that thumbnail text design contributes to predicted performance.
- Face features such as `num_faces`, `largest_face_area_ratio`, or emotion indicators ranking highly would support the claim that human presence and expression matter.
- If both OCR and face features appear near the top rather than only CNN embedding dimensions, that supports the multimodal thesis more strongly than a vision-only explanation.
- If OCR and face features have very small SHAP importance compared with CNN embeddings, that weakens the claim that these multimodal cues add meaningful explanatory value.
"""
    output_path.write_text(notes)


def subset_to_arrays(dataset: ThumbnailDataset, subset: Subset) -> tuple[np.ndarray, np.ndarray]:
    indices = subset.indices
    features = np.concatenate(
        [
            dataset.cnn[indices].cpu().numpy(),
            dataset.text[indices].cpu().numpy(),
            dataset.face[indices].cpu().numpy(),
        ],
        axis=1,
    ).astype(np.float32)
    labels = dataset.labels[indices].cpu().numpy().astype(np.float32)
    return features, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SHAP feature importance for the regression Fusion MLP."
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
    parser.add_argument("--background_size", type=int, default=128)
    parser.add_argument("--explain_size", type=int, default=256)
    parser.add_argument("--plot_top_k", type=int, default=20)
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=MODELS_DIR / "fusion_mlp_regression.pt",
        help="Loads a saved regression model if present; otherwise trains and saves to this path.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            args.checkpoint_path,
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

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    cnn_dim = dataset.cnn.shape[1]
    text_dim = dataset.text.shape[1]
    face_dim = dataset.face.shape[1]
    feature_names = build_feature_names(cnn_dim, text_dim, face_dim)

    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        num_classes=1,
        dropout_p=args.dropout,
    )

    if args.checkpoint_path.exists():
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    else:
        print("Training regression FusionMLP because no checkpoint was provided.")
        history = train_regression(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            loss_name=args.loss,
            device=device,
        )
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cnn_dim": cnn_dim,
                "text_dim": text_dim,
                "face_dim": face_dim,
                "seed": args.seed,
                "history": history,
                "loss": args.loss,
                "target_transform": args.target_transform,
            },
            args.checkpoint_path,
        )
        print(f"Saved trained checkpoint to {args.checkpoint_path}")

    model = model.to(device)
    background_x, _ = subset_to_arrays(dataset, train_ds)
    explain_x, _ = subset_to_arrays(dataset, val_ds)
    background_x = background_x[: min(args.background_size, len(background_x))]
    explain_x = explain_x[: min(args.explain_size, len(explain_x))]

    shap_values = compute_shap_values(
        model=model,
        background_x=background_x,
        explain_x=explain_x,
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        device=device,
    )
    ranked_importance = rank_global_importance(shap_values, feature_names)

    all_features_path = args.output_dir / "shap_regression_feature_importance.csv"
    top10_path = args.output_dir / "shap_regression_top10_features.csv"
    plot_path = args.output_dir / "shap_regression_global_importance.png"
    notes_path = args.output_dir / "shap_regression_notes.md"
    meta_path = args.output_dir / "shap_regression_run_metadata.json"

    ranked_importance.to_csv(all_features_path, index=False)
    ranked_importance.head(10).to_csv(top10_path, index=False)
    save_global_importance_plot(ranked_importance, plot_path, args.plot_top_k)
    save_interpretation_notes(notes_path)
    meta_path.write_text(
        json.dumps(
            {
                "split_name": args.split_name,
                "target_column": args.target_column,
                "target_transform": args.target_transform,
                "loss": args.loss,
                "seed": args.seed,
                "background_size": len(background_x),
                "explain_size": len(explain_x),
            },
            indent=2,
        )
    )

    print(f"Saved global SHAP plot to {plot_path}")
    print(f"Saved full feature ranking to {all_features_path}")
    print(f"Saved top 10 features to {top10_path}")
    print(f"Saved interpretation notes to {notes_path}")
    print(f"Saved run metadata to {meta_path}")
    print("\nTop 10 features:")
    print(ranked_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
