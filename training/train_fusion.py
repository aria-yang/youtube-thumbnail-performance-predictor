import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split

from thumbnail_performance.config import PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset
from thumbnail_performance.modeling.fusion_mlp import EarlyStopping, FusionMLP


def train(
    model: FusionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    class_weights: torch.Tensor = None,
) -> dict:
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=7, verbose=True)

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
            optimiser.zero_grad()
            logits = model(cnn_feat, text_feat, face_feat)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
                logits = model(cnn_feat, text_feat, face_feat)
                val_loss += criterion(logits, labels).item() * labels.size(0)
        val_loss /= len(val_loader.dataset)

        val_auroc = compute_auroc(model, val_loader, device=device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f}"
        )

        if stopper.step(val_loss, model):
            break

    stopper.restore_best(model)
    return history


def compute_auroc(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int = 5,
    device: str = "cpu",
) -> float:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
            logits = model(cnn_feat, text_feat, face_feat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    classes = list(range(num_classes))
    labels_bin = label_binarize(all_labels, classes=classes)
    auroc = roc_auc_score(labels_bin, all_probs, multi_class="ovr", average="macro")
    return float(auroc)


if __name__ == "__main__":
    from utils.class_weights import compute_class_weights

    parser = argparse.ArgumentParser(description="Train fusion model on aligned features.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "labeled_data.csv",
        help="Path to labeled CSV containing engagement_label.",
    )
    parser.add_argument(
        "--cnn_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "cnn_embeddings.npy",
        help="Path to CNN embedding array.",
    )
    parser.add_argument(
        "--text_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "text_embeddings.npy",
        help="Path to text feature array.",
    )
    parser.add_argument(
        "--face_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "face_embeddings.npy",
        help="Path to face feature array.",
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
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    args = parser.parse_args()

    dataset = ThumbnailDataset(args.csv_path, args.cnn_path, args.text_path, args.face_path)

    n_val = int(args.val_ratio * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    all_labels = torch.stack([dataset[i][3] for i in range(len(dataset))])
    class_weights = compute_class_weights(all_labels, num_classes=5)

    model = FusionMLP(cnn_dim=512, text_dim=4, face_dim=10)
    train(
        model,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        class_weights=class_weights,
        device=args.device,
    )
