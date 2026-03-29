import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from thumbnail_performance.cnn_embeddings import (
    BACKBONE_NAME,
    EMBEDDING_DIM,
    _embed_image,
    build_embedding_model,
)
from thumbnail_performance.cnn_embeddings import resolve_thumbnail_path as resolve_cnn_path
from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from thumbnail_performance.dataset import (
    ThumbnailDataset,
    main as build_labeled_dataset,
    read_csv_with_fallback,
)
from thumbnail_performance.face_emotion_detection import EMOTIONS, extract_face_emotion_features
from thumbnail_performance.modeling.fusion_mlp import EarlyStopping, EarlyStoppingMax, FusionMLP
from thumbnail_performance.ocr_features import build_ocr_feature_dataframe
from utils.class_weights import compute_class_weights


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int = 2,
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
        eta_min=base_lr * min_lr_ratio,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def train(
    model: FusionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    class_weights: torch.Tensor = None,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_metric: str = "auroc",
) -> dict:
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimiser,
        total_epochs=num_epochs,
        base_lr=lr,
    )
    if early_stopping_metric == "loss":
        stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=True,
        )
    else:
        metric_name = "AUROC" if early_stopping_metric == "auroc" else "F1"
        stopper = EarlyStoppingMax(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=True,
            metric_name=metric_name,
        )

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_f1": []}

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
        val_f1 = compute_macro_f1(model, val_loader, device=device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if early_stopping_metric == "loss":
            monitored_value = val_loss
        elif early_stopping_metric == "f1":
            monitored_value = val_f1
        else:
            monitored_value = val_auroc
        if stopper.step(monitored_value, model):
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


def compute_macro_f1(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
            logits = model(cnn_feat, text_feat, face_feat)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return float(f1_score(all_labels, all_preds, average="macro"))


def compute_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
            logits = model(cnn_feat, text_feat, face_feat)
            total_loss += criterion(logits, labels).item() * labels.size(0)

    return float(total_loss / len(loader.dataset))


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
            logits = model(cnn_feat, text_feat, face_feat)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_preds, axis=0)


def print_classification_breakdown(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int = 5,
    device: str = "cpu",
) -> None:
    y_true, y_pred = collect_predictions(model, loader, device=device)
    labels = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[f"class_{idx}" for idx in labels],
        digits=4,
        zero_division=0,
    )

    print("Test confusion matrix:")
    print(cm)
    print("Test classification report:")
    print(report)


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


def restore_artifact(local_path: Path, artifact_root: Path | None, overwrite: bool = False) -> None:
    if artifact_root is None:
        return
    src = artifact_root / local_path.name
    if src.exists() and (overwrite or not local_path.exists()):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        print(f"Restored {local_path.name} from {src}")


def restore_artifacts(local_paths: list[Path], artifact_root: Path | None, overwrite: bool = False) -> None:
    for local_path in local_paths:
        restore_artifact(local_path, artifact_root, overwrite=overwrite)


def sync_artifact_to_root(local_path: Path, artifact_root: Path | None) -> None:
    if artifact_root is None or not local_path.exists():
        return
    artifact_root.mkdir(parents=True, exist_ok=True)
    dst = artifact_root / local_path.name
    shutil.copy2(local_path, dst)
    print(f"Synced {local_path.name} to {dst}")


def sync_artifacts_to_root(local_paths: list[Path], artifact_root: Path | None) -> None:
    for local_path in local_paths:
        sync_artifact_to_root(local_path, artifact_root)


def build_thumbnail_index(thumbnail_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in thumbnail_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                index.setdefault(path.stem, path)
    return index


def resolve_across_roots(
    thumbnail_dirs: list[Path],
    thumbnail_index: dict[str, Path],
    channel: str,
    video_id: str,
) -> Path | None:
    direct_match = thumbnail_index.get(video_id)
    if direct_match is not None:
        return direct_match

    for root in thumbnail_dirs:
        img_path = resolve_cnn_path(root, channel, video_id)
        if img_path is not None:
            return img_path
    return None


def run_cnn_stage(
    csv_path: Path,
    thumbnail_dirs: list[Path],
    output_path: Path,
    cache_path: Path,
    device: str,
) -> None:
    df = pd.read_csv(csv_path)
    df["Id"] = df["Id"].astype(str)
    thumbnail_index = build_thumbnail_index(thumbnail_dirs)
    print(f"CNN stage thumbnail index: {len(thumbnail_index)} files discovered")

    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="Id")
        cached.index = cached.index.astype(str)
        print(f"Loaded CNN cache with {len(cached)} rows from {cache_path}")
    else:
        cached = pd.DataFrame()

    missing_df = df.loc[~df["Id"].isin(cached.index)].copy()
    new_rows = []
    print(f"CNN stage: {len(cached)} cached, {len(missing_df)} to process")

    if len(missing_df) > 0:
        print(f"Loading {BACKBONE_NAME} weights for CNN embeddings...")
        model = build_embedding_model(device=device)
    else:
        model = None

    for _, row in tqdm(missing_df.iterrows(), total=len(missing_df), desc="CNN embeddings"):
        vid = str(row["Id"])
        channel = str(row["Channel"])
        img_path = resolve_across_roots(thumbnail_dirs, thumbnail_index, channel, vid)
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        missing = 1

        if img_path is not None:
            result = _embed_image(img_path, model, device=device)
            if result is not None:
                embedding = result
                missing = 0

        entry = {"Id": vid, "cnn_missing": missing}
        for i, val in enumerate(embedding):
            entry[f"cnn_{i}"] = float(val)
        new_rows.append(entry)

    if new_rows:
        new_df = pd.DataFrame(new_rows).set_index("Id")
        cached = pd.concat([cached, new_df]) if not cached.empty else new_df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_csv(cache_path)

    embed_cols = [f"cnn_{i}" for i in range(EMBEDDING_DIM)]
    all_embeddings = cached.reindex(df["Id"])[embed_cols].fillna(0).values.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_embeddings)
    print(f"Saved {output_path}")


def load_ocr_cache_sources(cache_paths: list[Path]) -> pd.DataFrame:
    frames = []
    seen = set()

    for cache_path in cache_paths:
        if cache_path in seen or not cache_path.exists():
            continue
        seen.add(cache_path)
        df = pd.read_csv(cache_path, index_col="thumbnail_id")
        df.index = df.index.astype(str)
        frames.append(df)
        print(f"Loaded OCR cache with {len(df)} rows from {cache_path}")

    if not frames:
        return pd.DataFrame()

    cached = pd.concat(frames)
    cached = cached[~cached.index.duplicated(keep="first")]
    return cached


def run_ocr_stage(
    csv_path: Path,
    thumbnail_dirs: list[Path],
    ocr_csv_path: Path,
    output_path: Path,
    backend: str,
    seed_ocr_cache_paths: list[Path],
    ocr_use_gpu: bool,
) -> None:
    labeled = pd.read_csv(csv_path)
    labeled["Id"] = labeled["Id"].astype(str)
    valid_ids = set(labeled["Id"])

    cache_sources = [ocr_csv_path] + seed_ocr_cache_paths
    cached = load_ocr_cache_sources(cache_sources)

    missing_ids = valid_ids - set(cached.index)
    print(
        f"OCR stage: {len(cached)} cached, {len(missing_ids)} to process "
        f"(backend={backend}, gpu={ocr_use_gpu})"
    )
    if missing_ids:
        new_frames = []
        remaining_ids = set(missing_ids)
        for thumbnail_dir in thumbnail_dirs:
            if not remaining_ids:
                break
            print(f"OCR scanning root: {thumbnail_dir}")
            try:
                new_df = build_ocr_feature_dataframe(
                    thumbnail_dir=str(thumbnail_dir),
                    valid_ids=remaining_ids,
                    backend=backend,
                    use_gpu=ocr_use_gpu,
                )
            except FileNotFoundError:
                continue
            if not new_df.empty:
                new_frames.append(new_df)
                remaining_ids -= set(new_df.index.astype(str))

        if new_frames:
            new_df = pd.concat(new_frames)
            new_df = new_df[~new_df.index.duplicated(keep="first")]
            cached = pd.concat([cached, new_df]) if not cached.empty else new_df

    ocr_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_csv(ocr_csv_path)

    feat_cols = ["word_count", "capital_letter_pct", "has_numeric", "char_count"]
    arr = cached.reindex(labeled["Id"])[feat_cols].fillna(0).values.astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    print(f"Saved {output_path}")


def run_face_stage(
    csv_path: Path,
    thumbnail_dirs: list[Path],
    output_path: Path,
    cache_path: Path,
    device: str,
) -> None:
    df = pd.read_csv(csv_path)
    df["Id"] = df["Id"].astype(str)
    frames = []
    remaining_ids = set(df["Id"])
    thumbnail_index = build_thumbnail_index(thumbnail_dirs)
    print(f"Face stage thumbnail index: {len(thumbnail_index)} files discovered")

    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="Id")
        cached.index = cached.index.astype(str)
        frames.append(cached)
        remaining_ids -= set(cached.index)
        print(f"Loaded face cache with {len(cached)} rows from {cache_path}")

    print(f"Face stage: {len(df) - len(remaining_ids)} cached, {len(remaining_ids)} to process")

    for thumbnail_dir in thumbnail_dirs:
        if not remaining_ids:
            break
        root_index = build_thumbnail_index([thumbnail_dir])
        root_ids = []
        for _, row in df.loc[df["Id"].isin(remaining_ids)].iterrows():
            img_path = resolve_across_roots(
                [thumbnail_dir],
                root_index,
                str(row["Channel"]),
                str(row["Id"]),
            )
            if img_path is not None:
                root_ids.append(str(row["Id"]))
        if not root_ids:
            continue
        subset_df = df.loc[df["Id"].isin(root_ids)].copy()
        feats = extract_face_emotion_features(
            subset_df,
            thumbnail_dir=thumbnail_dir,
            cache_path=None,
            device=device,
        )
        if not feats.empty:
            feats.index = feats.index.astype(str)
            found_ids = set(feats.index)
            remaining_ids -= found_ids
            frames.append(feats)

    feats = pd.concat(frames) if frames else pd.DataFrame()
    feats = feats[~feats.index.duplicated(keep="first")]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache_path)

    feat_cols = ["num_faces", "largest_face_area_ratio"] + [
        f"emotion_{emotion}" for emotion in EMOTIONS
    ] + ["emotion_unknown"]

    arr = feats.reindex(df["Id"].astype(str))[feat_cols].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    print(f"Saved {output_path}")


def run_training_stage(
    csv_path: Path,
    cnn_path: Path,
    text_path: Path,
    face_path: Path,
    batch_size: int,
    num_epochs: int,
    lr: float,
    hidden1: int,
    hidden2: int,
    dropout_p: float,
    split_dir: Path,
    split_name: str,
    early_stopping_metric: str,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    device: str,
    checkpoint_path: Path,
) -> None:
    dataset = ThumbnailDataset(csv_path, cnn_path, text_path, face_path)
    cnn_dim = int(np.load(cnn_path, mmap_mode="r").shape[1])
    text_dim = int(np.load(text_path, mmap_mode="r").shape[1])
    face_dim = int(np.load(face_path, mmap_mode="r").shape[1])
    print(f"Training stage dims: cnn={cnn_dim}, text={text_dim}, face={face_dim}")

    split_df = read_csv_with_fallback(csv_path)
    split_df["Id"] = split_df["Id"].astype(str)
    train_ids, val_ids, test_ids = load_saved_split_ids(split_dir, split_name)
    id_to_idx = {video_id: idx for idx, video_id in enumerate(split_df["Id"])}
    train_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in train_ids]
    val_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in val_ids]
    test_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in test_ids]

    if not train_indices or not val_indices:
        raise ValueError(
            f"Saved split '{split_name}' produced an empty train/val subset. "
            "Check that split CSVs match the labeled dataset IDs."
        )

    print(
        f"Using saved split '{split_name}': "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_labels = torch.stack([dataset[idx][3] for idx in train_indices])
    class_weights = compute_class_weights(all_labels, num_classes=5)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if device != "cpu" else class_weights
    )

    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout_p=dropout_p,
    )
    train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        class_weights=class_weights,
        device=device,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_metric=early_stopping_metric,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cnn_dim": cnn_dim,
            "text_dim": text_dim,
            "face_dim": face_dim,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "dropout_p": dropout_p,
            "split_name": split_name,
        },
        checkpoint_path,
    )
    print(f"Saved fusion checkpoint to {checkpoint_path}")

    test_loss = compute_loss(model, test_loader, criterion, device=device)
    test_auroc = compute_auroc(model, test_loader, device=device)
    test_f1 = compute_macro_f1(model, test_loader, device=device)
    print(
        f"Test metrics | "
        f"Loss: {test_loss:.4f} | "
        f"AUROC: {test_auroc:.4f} | "
        f"F1: {test_f1:.4f}"
    )
    print_classification_breakdown(model, test_loader, num_classes=5, device=device)
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build merged thumbnail features and train the fusion model."
    )
    parser.add_argument(
        "--raw_csv_path",
        type=Path,
        default=RAW_DATA_DIR / "merged_data.csv",
        help="Raw metadata CSV to label.",
    )
    parser.add_argument(
        "--labeled_csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_labeled_data.csv",
        help="Output path for labeled CSV.",
    )
    parser.add_argument(
        "--thumbnail_dirs",
        type=Path,
        nargs="+",
        default=[
            RAW_DATA_DIR.parent / "thumbnails" / "images",
            RAW_DATA_DIR.parent / "thumbnails" / "new_images",
        ],
        help="One or more root thumbnail directories with channel subfolders.",
    )
    parser.add_argument(
        "--cnn_output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / f"merged_cnn_embeddings_{BACKBONE_NAME}.npy",
        help="Output path for CNN embeddings.",
    )
    parser.add_argument(
        "--cnn_cache_path",
        type=Path,
        default=PROCESSED_DATA_DIR / f"merged_cnn_cache_{BACKBONE_NAME}.csv",
        help="Cache path for resumable CNN extraction.",
    )
    parser.add_argument(
        "--ocr_csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_ocr_features.csv",
        help="Output path for OCR feature CSV.",
    )
    parser.add_argument(
        "--seed_ocr_cache_paths",
        type=Path,
        nargs="*",
        default=[
            PROCESSED_DATA_DIR / "ocr_features.csv",
            PROCESSED_DATA_DIR / "new_ocr_features.csv",
        ],
        help="Existing OCR cache CSVs to reuse before recomputing missing IDs.",
    )
    parser.add_argument(
        "--text_output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_text_embeddings.npy",
        help="Output path for OCR/text feature array.",
    )
    parser.add_argument(
        "--face_output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_face_embeddings.npy",
        help="Output path for face feature array.",
    )
    parser.add_argument(
        "--face_cache_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_face_cache.csv",
        help="Cache path for resumable face extraction.",
    )
    parser.add_argument(
        "--ocr_backend",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract"],
        help="OCR backend to use.",
    )
    parser.add_argument(
        "--dataset_test_size",
        type=float,
        default=0.2,
        help="Split ratio used when creating engagement labels.",
    )
    parser.add_argument(
        "--dataset_random_state",
        type=int,
        default=42,
        help="Random seed used when creating engagement labels.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Fusion training batch size.",
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
        help="Fusion training learning rate.",
    )
    parser.add_argument(
        "--hidden1",
        type=int,
        default=512,
        help="Width of the first hidden layer in the fusion MLP.",
    )
    parser.add_argument(
        "--hidden2",
        type=int,
        default=256,
        help="Width of the second hidden layer in the fusion MLP.",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.4,
        help="Dropout probability in the fusion MLP.",
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
        help="Saved split prefix to use for fusion training.",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="auroc",
        choices=["auroc", "loss", "f1"],
        help="Validation metric to monitor for early stopping.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=12,
        help="Epochs to wait for improvement before stopping.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for embeddings and training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible fusion training.",
    )
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("/content/drive/MyDrive/ECE324/youtube-thumbnail-performance-predictor-artifacts"),
        help="Directory containing cached/generated artifacts to restore from and sync back to.",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Skip artifact restore/refresh/sync and only train from existing processed files.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=MODELS_DIR / "fusion_mlp.pt",
        help="Output path for the trained FusionMLP checkpoint.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    artifact_root = None
    if not args.train_only:
        artifact_root = (
            args.artifact_root
            if args.artifact_root.exists() or "drive" in str(args.artifact_root).lower()
            else None
        )
    ocr_use_gpu = (
        args.ocr_backend == "easyocr"
        and args.device.startswith("cuda")
        and torch.cuda.is_available()
    )

    if args.train_only:
        print("Train-only mode enabled: skipping artifact restore, feature refresh, and sync.")
    else:
        restore_artifacts([args.raw_csv_path], artifact_root, overwrite=True)

        restore_artifacts(
            [
                args.labeled_csv_path,
                args.cnn_cache_path,
                args.ocr_csv_path,
                args.face_cache_path,
            ],
            artifact_root,
            overwrite=True,
        )

        restore_artifacts(
            [
                args.cnn_output_path,
                args.text_output_path,
                args.face_output_path,
            ],
            artifact_root,
            overwrite=False,
        )

        print("Stage 1/5: Building labeled dataset")
        build_labeled_dataset(
            input_path=args.raw_csv_path,
            output_path=args.labeled_csv_path,
            test_size=args.dataset_test_size,
            random_state=args.dataset_random_state,
        )
        sync_artifacts_to_root([args.labeled_csv_path], artifact_root)

        print("Stage 2/5: Extracting CNN embeddings")
        run_cnn_stage(
            csv_path=args.labeled_csv_path,
            thumbnail_dirs=args.thumbnail_dirs,
            output_path=args.cnn_output_path,
            cache_path=args.cnn_cache_path,
            device=args.device,
        )
        sync_artifacts_to_root([args.cnn_output_path, args.cnn_cache_path], artifact_root)

        print("Stage 3/5: Refreshing OCR feature cache")
        run_ocr_stage(
            csv_path=args.labeled_csv_path,
            thumbnail_dirs=args.thumbnail_dirs,
            ocr_csv_path=args.ocr_csv_path,
            output_path=args.text_output_path,
            backend=args.ocr_backend,
            seed_ocr_cache_paths=args.seed_ocr_cache_paths,
            ocr_use_gpu=ocr_use_gpu,
        )
        sync_artifacts_to_root([args.ocr_csv_path, args.text_output_path], artifact_root)

        print("Stage 4/5: Extracting face/emotion features")
        run_face_stage(
            csv_path=args.labeled_csv_path,
            thumbnail_dirs=args.thumbnail_dirs,
            output_path=args.face_output_path,
            cache_path=args.face_cache_path,
            device=args.device,
        )
        sync_artifacts_to_root([args.face_output_path, args.face_cache_path], artifact_root)

    print("Training fusion model")
    run_training_stage(
        csv_path=args.labeled_csv_path,
        cnn_path=args.cnn_output_path,
        text_path=args.text_output_path,
        face_path=args.face_output_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout_p=args.dropout_p,
        split_dir=args.split_dir,
        split_name=args.split_name,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
    )

    if not args.train_only:
        sync_artifacts_to_root(
            [
                args.labeled_csv_path,
                args.cnn_output_path,
                args.cnn_cache_path,
                args.ocr_csv_path,
                args.text_output_path,
                args.face_output_path,
                args.face_cache_path,
                args.checkpoint_path,
            ],
            artifact_root,
        )


if __name__ == "__main__":
    main()
