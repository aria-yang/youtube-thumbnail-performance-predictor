import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset

from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset, read_csv_with_fallback
from thumbnail_performance.modeling.fusion_mlp import EarlyStopping, FusionMLP


DEFAULT_CSV_PATH = PROCESSED_DATA_DIR / "merged_labeled_data.csv"
DEFAULT_TEXT_PATH = PROCESSED_DATA_DIR / "merged_text_embeddings.npy"
DEFAULT_FACE_PATH = PROCESSED_DATA_DIR / "merged_face_embeddings.npy"


def resolve_cnn_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    candidates = [
        PROCESSED_DATA_DIR / "merged_cnn_embeddings_resnet50.npy",
        PROCESSED_DATA_DIR / "merged_cnn_embeddings.npy",
        PROCESSED_DATA_DIR / "cnn_embeddings.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_saved_split_ids(split_dir: Path, split_name: str) -> tuple[set[str], set[str], set[str]]:
    def read_ids(path: Path) -> set[str]:
        if not path.exists():
            raise FileNotFoundError(f"Saved split file not found: {path}")
        df = read_csv_with_fallback(path)
        return set(df["Id"].astype(str))

    train_ids = read_ids(split_dir / f"{split_name}_train.csv")
    val_ids = read_ids(split_dir / f"{split_name}_val.csv")
    test_ids = read_ids(split_dir / f"{split_name}_test.csv")
    return train_ids, val_ids, test_ids


def build_regression_loss(loss_name: str) -> nn.Module:
    normalized_name = loss_name.lower()
    if normalized_name == "smoothl1":
        return nn.SmoothL1Loss()
    if normalized_name == "mse":
        return nn.MSELoss()
    if normalized_name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss '{loss_name}'. Choose from: smoothl1, mse, l1.")


def collect_predictions(
    model: nn.Module,
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
            targets = targets.to(device)

            preds = model(cnn_feat, text_feat, face_feat).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    spearman = float(pd.Series(targets).corr(pd.Series(predictions), method="spearman"))

    if np.isnan(spearman):
        spearman = 0.0

    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": rmse,
        "r2": float(r2_score(targets, predictions)),
        "spearman": spearman,
    }


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for cnn_feat, text_feat, face_feat, targets in loader:
            cnn_feat = cnn_feat.to(device)
            text_feat = text_feat.to(device)
            face_feat = face_feat.to(device)
            targets = targets.to(device)

            preds = model(cnn_feat, text_feat, face_feat).squeeze(-1)
            total_loss += criterion(preds, targets).item() * targets.size(0)

    predictions, ground_truth = collect_predictions(model, loader, device)
    average_loss = total_loss / len(loader.dataset)
    metrics = compute_regression_metrics(predictions, ground_truth)
    return average_loss, metrics, predictions, ground_truth


def train_regression(
    model: FusionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    loss_name: str,
    device: str,
) -> dict[str, list[float]]:
    model.to(device)
    criterion = build_regression_loss(loss_name)
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=7, verbose=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
        "val_spearman": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for cnn_feat, text_feat, face_feat, targets in train_loader:
            cnn_feat = cnn_feat.to(device)
            text_feat = text_feat.to(device)
            face_feat = face_feat.to(device)
            targets = targets.to(device)

            optimiser.zero_grad()
            preds = model(cnn_feat, text_feat, face_feat).squeeze(-1)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss, val_metrics, _, _ = evaluate_regression(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_spearman"].append(val_metrics["spearman"])

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f} | "
            f"Val Spearman: {val_metrics['spearman']:.4f}"
        )

        if stopper.step(val_loss, model):
            break

    stopper.restore_best(model)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a regression variant of the multimodal FusionMLP."
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the labeled CSV containing normalized_performance.",
    )
    parser.add_argument(
        "--cnn_path",
        type=Path,
        default=None,
        help="Path to CNN embeddings. Defaults to the merged CNN artifact when present.",
    )
    parser.add_argument(
        "--text_path",
        type=Path,
        default=DEFAULT_TEXT_PATH,
        help="Path to OCR/text features.",
    )
    parser.add_argument(
        "--face_path",
        type=Path,
        default=DEFAULT_FACE_PATH,
        help="Path to face/emotion features.",
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=DATA_DIR / "splits",
        help="Directory containing saved split CSV files.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="random",
        choices=["random", "channel", "time"],
        help="Saved split prefix to use.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="normalized_performance",
        help="Continuous target column to predict.",
    )
    parser.add_argument(
        "--target_transform",
        type=str,
        default="log1p",
        choices=["log1p", "none"],
        help="Transform applied to the target before training.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="smoothl1",
        choices=["smoothl1", "mse", "l1"],
        help="Regression loss to optimize.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for training.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=MODELS_DIR / "fusion_mlp_regression.pt",
        help="Where to save the trained regression checkpoint.",
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "fusion_mlp_regression_metrics.json",
        help="Where to save final regression metrics.",
    )
    args = parser.parse_args()

    cnn_path = resolve_cnn_path(args.cnn_path)
    target_transform = None if args.target_transform == "none" else args.target_transform

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
    train_ids, val_ids, test_ids = load_saved_split_ids(args.split_dir, args.split_name)
    id_to_idx = {video_id: idx for idx, video_id in enumerate(split_df["Id"])}

    train_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in train_ids]
    val_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in val_ids]
    test_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in test_ids]

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            f"Saved split '{args.split_name}' produced an empty train/val/test subset. "
            "Check that split CSVs match the labeled dataset IDs."
        )

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size, shuffle=False)

    cnn_dim = int(np.load(cnn_path, mmap_mode="r").shape[1])
    text_dim = int(np.load(args.text_path, mmap_mode="r").shape[1])
    face_dim = int(np.load(args.face_path, mmap_mode="r").shape[1])

    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        num_classes=1,
    )

    history = train_regression(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        loss_name=args.loss,
        device=args.device,
    )

    criterion = build_regression_loss(args.loss)
    val_loss, val_metrics, _, _ = evaluate_regression(model, val_loader, criterion, args.device)
    test_loss, test_metrics, _, _ = evaluate_regression(model, test_loader, criterion, args.device)

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cnn_dim": cnn_dim,
            "text_dim": text_dim,
            "face_dim": face_dim,
            "target_column": args.target_column,
            "target_transform": args.target_transform,
            "loss": args.loss,
            "split_name": args.split_name,
            "history": history,
        },
        args.checkpoint_path,
    )

    output = {
        "target_column": args.target_column,
        "target_transform": args.target_transform,
        "loss": args.loss,
        "split_name": args.split_name,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(output, indent=2))

    print(f"Saved checkpoint to {args.checkpoint_path}")
    print(f"Saved metrics to {args.metrics_path}")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
