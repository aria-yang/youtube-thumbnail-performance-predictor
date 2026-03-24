import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from thumbnail_performance.cnn_embeddings import EMBEDDING_DIM, _embed_image, build_embedding_model
from thumbnail_performance.cnn_embeddings import resolve_thumbnail_path as resolve_cnn_path
from thumbnail_performance.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset, main as build_labeled_dataset
from thumbnail_performance.face_emotion_detection import EMOTIONS, extract_face_emotion_features
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from thumbnail_performance.ocr_features import build_ocr_feature_dataframe
from training.train_fusion import train
from utils.class_weights import compute_class_weights


def run_cnn_stage(
    csv_path: Path,
    thumbnail_dir: Path,
    output_path: Path,
    cache_path: Path,
    device: str,
) -> None:
    df = pd.read_csv(csv_path)
    df["Id"] = df["Id"].astype(str)

    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="Id")
        cached.index = cached.index.astype(str)
        print(f"Loaded CNN cache with {len(cached)} rows from {cache_path}")
    else:
        cached = pd.DataFrame()

    missing_df = df.loc[~df["Id"].isin(cached.index)].copy()
    new_rows = []

    if len(missing_df) > 0:
        model = build_embedding_model(device=device)
    else:
        model = None

    for _, row in missing_df.iterrows():
        vid = str(row["Id"])
        channel = str(row["Channel"])
        img_path = resolve_cnn_path(thumbnail_dir, channel, vid)
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
    all_embeddings = (
        cached.reindex(df["Id"])[embed_cols].fillna(0).values.astype(np.float32)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, all_embeddings)
    print(f"Saved {output_path}")


def run_ocr_stage(
    csv_path: Path,
    thumbnail_dir: Path,
    ocr_csv_path: Path,
    output_path: Path,
    backend: str,
) -> None:
    labeled = pd.read_csv(csv_path)
    labeled["Id"] = labeled["Id"].astype(str)
    valid_ids = set(labeled["Id"])

    if ocr_csv_path.exists():
        cached = pd.read_csv(ocr_csv_path, index_col="thumbnail_id")
        cached.index = cached.index.astype(str)
        print(f"Loaded OCR cache with {len(cached)} rows from {ocr_csv_path}")
    else:
        cached = pd.DataFrame()

    missing_ids = valid_ids - set(cached.index)
    if missing_ids:
        new_df = build_ocr_feature_dataframe(
            thumbnail_dir=str(thumbnail_dir),
            valid_ids=missing_ids,
            backend=backend,
        )
        cached = pd.concat([cached, new_df]) if not cached.empty else new_df

    ocr_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_csv(ocr_csv_path)
    df = cached

    feat_cols = ["word_count", "capital_letter_pct", "has_numeric", "char_count"]
    arr = (
        df.reindex(labeled["Id"])[feat_cols]
        .fillna(0)
        .values.astype(np.float32)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    print(f"Saved {output_path}")


def run_face_stage(
    csv_path: Path,
    thumbnail_dir: Path,
    output_path: Path,
    cache_path: Path,
    device: str,
) -> None:
    df = pd.read_csv(csv_path)
    feats = extract_face_emotion_features(
        df,
        thumbnail_dir=thumbnail_dir,
        cache_path=cache_path,
        device=device,
    )

    feat_cols = ["num_faces", "largest_face_area_ratio"] + [
        f"emotion_{emotion}" for emotion in EMOTIONS
    ] + ["emotion_unknown"]

    arr = feats.reindex(df["Id"].astype(str))[feat_cols].values.astype(np.float32)
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
    val_ratio: float,
    device: str,
) -> None:
    dataset = ThumbnailDataset(csv_path, cnn_path, text_path, face_path)

    n_val = int(val_ratio * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_labels = torch.stack([dataset[i][3] for i in range(len(dataset))])
    class_weights = compute_class_weights(all_labels, num_classes=5)

    model = FusionMLP(cnn_dim=512, text_dim=4, face_dim=10)
    train(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=lr,
        class_weights=class_weights,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full multimodal pipeline for a new dataset."
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
        "--thumbnail_dir",
        type=Path,
        default=RAW_DATA_DIR.parent / "thumbnails" / "images",
        help="Root thumbnail directory with channel subfolders.",
    )
    parser.add_argument(
        "--cnn_output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_cnn_embeddings.npy",
        help="Output path for CNN embeddings.",
    )
    parser.add_argument(
        "--cnn_cache_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_cnn_cache.csv",
        help="Cache path for resumable CNN extraction.",
    )
    parser.add_argument(
        "--ocr_csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_ocr_features.csv",
        help="Output path for OCR feature CSV.",
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
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio for fusion training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for embeddings and training.",
    )
    args = parser.parse_args()

    print("Stage 1/5: Building labeled dataset")
    build_labeled_dataset(
        input_path=args.raw_csv_path,
        output_path=args.labeled_csv_path,
        test_size=args.dataset_test_size,
        random_state=args.dataset_random_state,
    )

    print("Stage 2/5: Extracting CNN embeddings")
    run_cnn_stage(
        csv_path=args.labeled_csv_path,
        thumbnail_dir=args.thumbnail_dir,
        output_path=args.cnn_output_path,
        cache_path=args.cnn_cache_path,
        device=args.device,
    )

    print("Stage 3/5: Extracting OCR features")
    run_ocr_stage(
        csv_path=args.labeled_csv_path,
        thumbnail_dir=args.thumbnail_dir,
        ocr_csv_path=args.ocr_csv_path,
        output_path=args.text_output_path,
        backend=args.ocr_backend,
    )

    print("Stage 4/5: Extracting face/emotion features")
    run_face_stage(
        csv_path=args.labeled_csv_path,
        thumbnail_dir=args.thumbnail_dir,
        output_path=args.face_output_path,
        cache_path=args.face_cache_path,
        device=args.device,
    )

    print("Stage 5/5: Training fusion model")
    run_training_stage(
        csv_path=args.labeled_csv_path,
        cnn_path=args.cnn_output_path,
        text_path=args.text_output_path,
        face_path=args.face_output_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
        device=args.device,
    )
