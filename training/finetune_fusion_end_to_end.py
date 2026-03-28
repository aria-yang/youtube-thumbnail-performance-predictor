import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from thumbnail_performance.dataset import read_csv_with_fallback
from utils.class_weights import compute_class_weights


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

OCR_FEAT_COLS = ["word_count", "capital_letter_pct", "has_numeric", "char_count"]
FACE_FEAT_COLS = [
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    max_lr: float,
    warmup_epochs: int = 1,
    min_lr_ratio: float = 0.05,
):
    total_epochs = max(1, total_epochs)
    warmup_epochs = min(max(1, warmup_epochs), total_epochs)
    cosine_epochs = max(1, total_epochs - warmup_epochs)

    if total_epochs == 1:
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=max_lr * min_lr_ratio,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def build_thumbnail_index(thumbnail_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in thumbnail_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                index.setdefault(path.stem, path)
    return index


def load_ocr_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"OCR feature CSV not found: {path}")
    df = pd.read_csv(path, index_col="thumbnail_id")
    df.index = df.index.astype(str)
    return df


def load_face_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Face feature CSV not found: {path}")
    df = pd.read_csv(path, index_col="Id")
    df.index = df.index.astype(str)
    return df


def align_metadata(
    labeled_csv_path: Path,
    thumbnail_dirs: list[Path],
    ocr_csv_path: Path,
    face_csv_path: Path,
) -> pd.DataFrame:
    labeled = read_csv_with_fallback(labeled_csv_path)
    labeled["Id"] = labeled["Id"].astype(str)
    labeled["Channel"] = labeled["Channel"].astype(str)

    thumbnail_index = build_thumbnail_index(thumbnail_dirs)
    labeled["image_path"] = labeled["Id"].map(lambda vid: thumbnail_index.get(str(vid)))
    labeled["image_path"] = labeled["image_path"].apply(lambda p: str(p) if p is not None else "")

    ocr_df = load_ocr_features(ocr_csv_path)
    face_df = load_face_features(face_csv_path)

    labeled = labeled.join(ocr_df[OCR_FEAT_COLS], on="Id")
    labeled = labeled.join(face_df[FACE_FEAT_COLS], on="Id")

    for col in OCR_FEAT_COLS + FACE_FEAT_COLS:
        labeled[col] = labeled[col].fillna(0.0)

    labeled["has_image"] = labeled["image_path"].astype(str).str.len() > 0
    return labeled


def load_saved_split_ids(split_dir: Path, split_name: str) -> tuple[set[str], set[str], set[str]]:
    train_path = split_dir / f"{split_name}_train.csv"
    val_path = split_dir / f"{split_name}_val.csv"
    test_path = split_dir / f"{split_name}_test.csv"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Saved split file not found: {path}")

    def read_ids(path: Path) -> set[str]:
        df = read_csv_with_fallback(path)
        return set(df["Id"].astype(str))

    return read_ids(train_path), read_ids(val_path), read_ids(test_path)


class ThumbnailFineTuneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_transform: transforms.Compose):
        self.df = df.reset_index(drop=True).copy()
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        if image_path:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_transform(image)
            image_missing = torch.tensor(0.0, dtype=torch.float32)
        else:
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            image_missing = torch.tensor(1.0, dtype=torch.float32)

        text_tensor = torch.tensor(row[OCR_FEAT_COLS].values.astype(np.float32))
        face_tensor = torch.tensor(row[FACE_FEAT_COLS].values.astype(np.float32))
        label_tensor = torch.tensor(int(row["engagement_label"]), dtype=torch.long)

        return image_tensor, text_tensor, face_tensor, image_missing, label_tensor


class EndToEndFusionModel(nn.Module):
    def __init__(
        self,
        text_dim: int = 4,
        face_dim: int = 10,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout_p: float = 0.35,
        num_classes: int = 5,
        backbone_name: str = "resnet50",
    ):
        super().__init__()
        if backbone_name != "resnet50":
            raise ValueError("Only resnet50 is currently supported for end-to-end fine-tuning.")

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
        cnn_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.cnn_dim = cnn_dim
        cnn_proj_dim = min(hidden1, 512)
        aux_proj_dim = max(hidden2 // 2, 32)
        missing_proj_dim = max(aux_proj_dim // 2, 8)

        self.cnn_projector = nn.Sequential(
            nn.Linear(cnn_dim, cnn_proj_dim),
            nn.BatchNorm1d(cnn_proj_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, aux_proj_dim),
            nn.BatchNorm1d(aux_proj_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.face_projector = nn.Sequential(
            nn.Linear(face_dim, aux_proj_dim),
            nn.BatchNorm1d(aux_proj_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.missing_projector = nn.Sequential(
            nn.Linear(1, missing_proj_dim),
            nn.ReLU(),
        )

        self.input_dim = cnn_proj_dim + aux_proj_dim + aux_proj_dim + missing_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden2, num_classes),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_layer4(self) -> None:
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        text_feat: torch.Tensor,
        face_feat: torch.Tensor,
        image_missing: torch.Tensor,
    ) -> torch.Tensor:
        cnn_feat = self.backbone(images)
        cnn_feat = self.cnn_projector(cnn_feat)
        text_feat = self.text_projector(text_feat)
        face_feat = self.face_projector(face_feat)
        missing_feat = self.missing_projector(image_missing.unsqueeze(1))
        fused = torch.cat([cnn_feat, text_feat, face_feat, missing_feat], dim=1)
        return self.classifier(fused)


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int = 5,
) -> dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_items = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, text_feat, face_feat, image_missing, labels in loader:
            images = images.to(device)
            text_feat = text_feat.to(device)
            face_feat = face_feat.to(device)
            image_missing = image_missing.to(device)
            labels = labels.to(device)

            logits = model(images, text_feat, face_feat, image_missing)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_items += batch_size
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    return {
        "loss": total_loss / max(n_items, 1),
        "macro_auroc": float(
            roc_auc_score(labels_bin, all_probs, multi_class="ovr", average="macro")
        ),
        "macro_f1": float(
            f1_score(all_labels, np.argmax(all_probs, axis=1), average="macro")
        ),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for images, text_feat, face_feat, image_missing, labels in loader:
        images = images.to(device)
        text_feat = text_feat.to(device)
        face_feat = face_feat.to(device)
        image_missing = image_missing.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, text_feat, face_feat, image_missing)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def build_optimizer(
    model: EndToEndFusionModel,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    head_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("backbone.")
    ]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def build_stage_optimizer_and_scheduler(
    model: EndToEndFusionModel,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    remaining_epochs: int,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    optimizer = build_optimizer(
        model,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=remaining_epochs,
        max_lr=max(head_lr, backbone_lr),
    )
    return optimizer, scheduler


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end fine-tuning of ResNet-50 + OCR + face features."
    )
    parser.add_argument(
        "--labeled_csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_labeled_data.csv",
    )
    parser.add_argument(
        "--ocr_csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_ocr_features.csv",
    )
    parser.add_argument(
        "--face_csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_face_cache.csv",
    )
    parser.add_argument(
        "--thumbnail_dirs",
        type=Path,
        nargs="+",
        default=[
            RAW_DATA_DIR.parent / "thumbnails" / "images",
            RAW_DATA_DIR.parent / "thumbnails" / "new_images",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--freeze_epochs", type=int, default=3)
    parser.add_argument("--unfreeze_all_epoch", type=int, default=8)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--dropout_p", type=float, default=0.35)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=MODELS_DIR / "fusion_end_to_end_resnet50.pt",
    )
    parser.add_argument(
        "--history_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "fusion_end_to_end_history.csv",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    df = align_metadata(
        labeled_csv_path=args.labeled_csv_path,
        thumbnail_dirs=args.thumbnail_dirs,
        ocr_csv_path=args.ocr_csv_path,
        face_csv_path=args.face_csv_path,
    )

    print(f"Aligned dataset rows: {len(df)}")
    print(f"Rows with thumbnails: {int(df['has_image'].sum())}")
    print(f"Rows missing thumbnails: {int((~df['has_image']).sum())}")
    print(f"OCR dim: {len(OCR_FEAT_COLS)} | Face dim: {len(FACE_FEAT_COLS)}")

    train_ids, val_ids, test_ids = load_saved_split_ids(args.split_dir, args.split_name)
    train_df = df.loc[df["Id"].isin(train_ids)].copy()
    val_df = df.loc[df["Id"].isin(val_ids)].copy()
    test_df = df.loc[df["Id"].isin(test_ids)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Saved split '{args.split_name}' produced an empty subset. "
            "Check that split CSVs match the labeled dataset IDs."
        )

    print(
        f"Using saved split '{args.split_name}': "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    train_ds = ThumbnailFineTuneDataset(train_df, IMAGE_TRANSFORM)
    val_ds = ThumbnailFineTuneDataset(val_df, IMAGE_TRANSFORM)
    test_ds = ThumbnailFineTuneDataset(test_df, IMAGE_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = compute_class_weights(
        torch.tensor(train_df["engagement_label"].astype(int).values, dtype=torch.long),
        num_classes=5,
    ).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = EndToEndFusionModel(
        text_dim=len(OCR_FEAT_COLS),
        face_dim=len(FACE_FEAT_COLS),
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout_p=args.dropout_p,
    ).to(args.device)

    model.freeze_backbone()
    optimizer, scheduler = build_stage_optimizer_and_scheduler(
        model,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        remaining_epochs=args.num_epochs,
    )

    best_val_auroc = -float("inf")
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.num_epochs + 1):
        if epoch == args.freeze_epochs + 1:
            model.unfreeze_layer4()
            optimizer, scheduler = build_stage_optimizer_and_scheduler(
                model,
                head_lr=args.head_lr,
                backbone_lr=args.backbone_lr,
                weight_decay=args.weight_decay,
                remaining_epochs=args.num_epochs - epoch + 1,
            )
            print("Unfroze ResNet-50 layer4.")
        if epoch == args.unfreeze_all_epoch:
            model.unfreeze_all()
            optimizer, scheduler = build_stage_optimizer_and_scheduler(
                model,
                head_lr=args.head_lr,
                backbone_lr=args.backbone_lr,
                weight_decay=args.weight_decay,
                remaining_epochs=args.num_epochs - epoch + 1,
            )
            print("Unfroze all ResNet-50 layers.")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_metrics = compute_metrics(model, val_loader, args.device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_auroc={val_metrics['macro_auroc']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["macro_auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, args.checkpoint_path)
            print(f"Saved best checkpoint to {args.checkpoint_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = compute_metrics(model, test_loader, args.device)
    print(
        f"Test metrics | loss={test_metrics['loss']:.4f} | "
        f"macro_auroc={test_metrics['macro_auroc']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    save_history_csv(history, args.history_path)
    print(f"Saved training history to {args.history_path}")


if __name__ == "__main__":
    main()
