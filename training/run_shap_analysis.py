import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from thumbnail_performance.dataset import parse_abbreviated_numeric, read_csv_with_fallback
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from training.train_fusion import load_saved_split_ids, train
from utils.class_weights import compute_class_weights


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


class InMemoryThumbnailDataset(Dataset):
    def __init__(
        self,
        cnn: np.ndarray,
        text: np.ndarray,
        face: np.ndarray,
        labels: np.ndarray,
    ):
        self.cnn = torch.tensor(cnn, dtype=torch.float32)
        self.text = torch.tensor(text, dtype=torch.float32)
        self.face = torch.tensor(face, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cnn[idx], self.text[idx], self.face[idx], self.labels[idx]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
            "The 'shap' package is required to run this script. "
            "Install it in your project environment, then rerun "
            "'python training/shap.py'."
        ) from exc
    finally:
        for idx, entry in reversed(removed_entries):
            sys.path.insert(idx, entry)

    shap_file = getattr(shap, "__file__", "")
    if shap_file:
        shap_path = Path(shap_file).resolve()
        if project_root in map(str, shap_path.parents):
            raise ModuleNotFoundError(
                "Python is resolving 'shap' to a local file inside this repository, but the "
                "external SHAP package is not installed in the active environment. Install "
                "the package in 'youtube-thumbnail-performance-predictor' and rerun the script."
            )

    return shap


def build_feature_names(cnn_dim: int, text_dim: int, face_dim: int) -> list[str]:
    cnn_names = [f"cnn_embedding_{idx:03d}" for idx in range(cnn_dim)]

    text_names = TEXT_FEATURE_NAMES[:text_dim]
    if len(text_names) < text_dim:
        text_names.extend(
            [f"text_feature_{idx}" for idx in range(len(text_names), text_dim)]
        )

    face_names = FACE_FEATURE_NAMES[:face_dim]
    if len(face_names) < face_dim:
        face_names.extend(
            [f"face_feature_{idx}" for idx in range(len(face_names), face_dim)]
        )

    return cnn_names + text_names + face_names


def add_engagement_labels(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    labeled_df = df.copy()
    labeled_df["views"] = labeled_df["Views"].apply(parse_abbreviated_numeric)
    labeled_df["subscriber_count"] = labeled_df["Subscribers"].apply(
        parse_abbreviated_numeric
    )
    labeled_df["normalized_performance"] = labeled_df["views"] / (
        labeled_df["subscriber_count"] + 1e-9
    )

    train_df, _ = train_test_split(
        labeled_df,
        test_size=test_size,
        random_state=random_state,
    )
    percentiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_edges = train_df["normalized_performance"].quantile(percentiles).values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    labeled_df["engagement_label"] = pd.cut(
        labeled_df["normalized_performance"],
        bins=bin_edges,
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(int)
    return labeled_df


def resolve_metadata_csv(preferred_csv_path: Path, target_rows: int) -> tuple[Path, pd.DataFrame]:
    candidates = [
        preferred_csv_path,
        PROCESSED_DATA_DIR / "merged_labeled_data.csv",
        PROCESSED_DATA_DIR / "new_labeled_data.csv",
        RAW_DATA_DIR / "data.csv",
        RAW_DATA_DIR / "merged_data.csv",
        RAW_DATA_DIR / "new_data.csv",
    ]

    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)

        df = read_csv_with_fallback(path)
        if len(df) != target_rows:
            continue

        if "engagement_label" not in df.columns:
            if {"Views", "Subscribers"}.issubset(df.columns):
                df = add_engagement_labels(df)
            else:
                continue

        return path, df

    raise ValueError(
        "Could not find a CSV with the same number of rows as the embedding arrays. "
        f"Expected {target_rows} rows based on the embeddings."
    )


def load_aligned_dataset(
    csv_path: Path,
    cnn_path: Path,
    text_path: Path,
    face_path: Path,
) -> tuple[InMemoryThumbnailDataset, Path]:
    cnn = np.load(cnn_path).astype(np.float32)
    text = np.load(text_path).astype(np.float32)
    face = np.load(face_path).astype(np.float32)

    row_counts = {cnn.shape[0], text.shape[0], face.shape[0]}
    if len(row_counts) != 1:
        raise ValueError(
            "Embedding arrays are not aligned. "
            f"Got cnn={cnn.shape[0]}, text={text.shape[0]}, face={face.shape[0]}."
        )

    target_rows = cnn.shape[0]
    resolved_csv_path, df = resolve_metadata_csv(csv_path, target_rows)
    labels = df["engagement_label"].astype(int).to_numpy()

    dataset = InMemoryThumbnailDataset(cnn=cnn, text=text, face=face, labels=labels)
    return dataset, resolved_csv_path


def split_dataset(
    dataset: Dataset,
    split_df: pd.DataFrame,
    split_dir: Path,
    split_name: str,
) -> tuple[Subset, Subset]:
    split_df = split_df.copy()
    split_df["Id"] = split_df["Id"].astype(str)
    train_ids, val_ids, _ = load_saved_split_ids(split_dir, split_name)
    id_to_idx = {video_id: idx for idx, video_id in enumerate(split_df["Id"])}
    train_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in train_ids]
    val_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in val_ids]

    if not train_indices or not val_indices:
        raise ValueError(
            f"Saved split '{split_name}' produced an empty train/val subset. "
            "Check that split CSVs match the aligned dataset IDs."
        )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    return train_ds, val_ds


def get_subset_labels(dataset: Dataset, subset: Subset) -> torch.Tensor:
    return dataset.labels[subset.indices]


def subset_to_arrays(dataset: Dataset, subset: Subset) -> tuple[np.ndarray, np.ndarray]:
    indices = subset.indices
    features = np.concatenate(
        [
            dataset.cnn[indices].cpu().numpy(),
            dataset.text[indices].cpu().numpy(),
            dataset.face[indices].cpu().numpy(),
        ],
        axis=1,
    ).astype(np.float32)
    labels = dataset.labels[indices].cpu().numpy().astype(np.int64)
    return features, labels


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
        face_feat = x[:, text_end : text_end + self.face_dim]
        return self.model(cnn_feat, text_feat, face_feat)


def train_or_load_model(
    args: argparse.Namespace,
    dataset: Dataset,
    train_ds: Subset,
    val_ds: Subset,
    cnn_dim: int,
    text_dim: int,
    face_dim: int,
) -> FusionMLP:
    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        num_classes=args.num_classes,
        dropout_p=args.dropout,
    )

    if args.checkpoint_path and args.checkpoint_path.exists():
        state_dict = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
        return model.to(args.device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weights(
        get_subset_labels(dataset, train_ds),
        num_classes=args.num_classes,
    )

    print("Training FusionMLP because no checkpoint was provided.")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        class_weights=class_weights,
    )

    if args.checkpoint_path:
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint_path)
        print(f"Saved trained checkpoint to {args.checkpoint_path}")

    return model.to(args.device)


def normalise_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=-1)

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


def rank_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    ranked = (
        pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        )
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
    plt.barh(top_importance["feature"], top_importance["mean_abs_shap"], color="#1f77b4")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance for Fusion MLP")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_interpretation_notes(output_path: Path) -> None:
    notes = """# SHAP Interpretation Notes

## How to interpret SHAP values
- A positive SHAP value means the feature pushed the model toward a higher logit for a class on that example.
- A negative SHAP value means the feature pushed the model away from that class on that example.
- Larger absolute SHAP values mean the feature had more influence on the prediction.
- Global importance is computed as mean absolute SHAP value, so it tells us which features mattered most overall, not whether they helped or hurt on average.

## Patterns that would support our thesis
- OCR features ranking near the top would support the claim that thumbnail text design contributes meaningfully to engagement prediction.
- Face features such as `num_faces`, `largest_face_area_ratio`, or emotion indicators ranking highly would support the claim that human presence and expressed emotion matter.
- If both OCR and face features appear in the global top features rather than only CNN embedding dimensions, that supports the multimodal thesis more strongly than a vision-only story.
- If the strongest non-CNN features are intuitive, such as more text density, numeric text, or expressive faces, that strengthens the interpretability argument for the project.
- If OCR and face features have negligible SHAP importance compared with CNN embeddings, that would weaken the thesis that these handcrafted multimodal cues add meaningful explanatory value.
"""
    output_path.write_text(notes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SHAP feature importance for the fusion MLP model."
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "labeled_data.csv",
    )
    parser.add_argument(
        "--cnn_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "cnn_embeddings.npy",
    )
    parser.add_argument(
        "--text_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "text_embeddings.npy",
    )
    parser.add_argument(
        "--face_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "face_embeddings.npy",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=MODELS_DIR / "fusion_mlp_shap.pt",
        help="Loads a saved model if present; otherwise trains and saves to this path.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=DATA_DIR / "splits",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="random",
        choices=["random", "channel", "time"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--background_size", type=int, default=128)
    parser.add_argument("--explain_size", type=int, default=256)
    parser.add_argument("--plot_top_k", type=int, default=20)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, resolved_csv_path = load_aligned_dataset(
        csv_path=args.csv_path,
        cnn_path=args.cnn_path,
        text_path=args.text_path,
        face_path=args.face_path,
    )
    print(f"Using metadata CSV: {resolved_csv_path}")

    cnn_dim = dataset.cnn.shape[1]
    text_dim = dataset.text.shape[1]
    face_dim = dataset.face.shape[1]
    feature_names = build_feature_names(cnn_dim, text_dim, face_dim)

    aligned_df = read_csv_with_fallback(resolved_csv_path)
    train_ds, val_ds = split_dataset(
        dataset,
        split_df=aligned_df,
        split_dir=args.split_dir,
        split_name=args.split_name,
    )
    model = train_or_load_model(
        args=args,
        dataset=dataset,
        train_ds=train_ds,
        val_ds=val_ds,
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
    )

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
        device=args.device,
    )

    ranked_importance = rank_global_importance(shap_values, feature_names)

    all_features_path = args.output_dir / "shap_feature_importance.csv"
    top10_path = args.output_dir / "shap_top10_features.csv"
    plot_path = args.output_dir / "shap_global_importance.png"
    notes_path = args.output_dir / "shap_notes.md"

    ranked_importance.to_csv(all_features_path, index=False)
    ranked_importance.head(10).to_csv(top10_path, index=False)
    save_global_importance_plot(
        ranked_importance=ranked_importance,
        output_path=plot_path,
        max_display=args.plot_top_k,
    )
    save_interpretation_notes(notes_path)

    print(f"Saved global SHAP plot to {plot_path}")
    print(f"Saved full feature ranking to {all_features_path}")
    print(f"Saved top 10 features to {top10_path}")
    print(f"Saved interpretation notes to {notes_path}")
    print("\nTop 10 features:")
    print(ranked_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
