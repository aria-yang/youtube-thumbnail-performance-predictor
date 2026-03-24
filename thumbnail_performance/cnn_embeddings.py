from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


EMBEDDING_DIM = 2048

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def build_embedding_model(device: str = "cpu") -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model.to(device)


def resolve_thumbnail_path(
    thumbnail_root: Path, channel: str, video_id: str
) -> Optional[Path]:
    channel_dir = thumbnail_root / channel
    if not channel_dir.exists():
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = channel_dir / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def _embed_image(img_path: Path, model: nn.Module, device: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor)
        return embedding.squeeze(0).cpu().numpy()
    except Exception:
        return None


def extract_cnn_embeddings(
    df: pd.DataFrame,
    thumbnail_dir: Path,
    id_col: str = "Id",
    channel_col: str = "Channel",
    device: str = "cpu",
) -> pd.DataFrame:
    if id_col not in df.columns:
        raise KeyError(f"Expected column '{id_col}' in dataframe.")

    has_channel = channel_col in df.columns
    thumbnail_dir = Path(thumbnail_dir)
    model = build_embedding_model(device=device)

    rows: List[Dict] = []

    for _, row in df.iterrows():
        vid = str(row[id_col])
        channel = str(row[channel_col]) if has_channel else ""
        img_path = resolve_thumbnail_path(thumbnail_dir, channel, vid) if has_channel else None

        if img_path is None:
            embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            missing = 1
        else:
            result = _embed_image(img_path, model, device)
            if result is None:
                embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                missing = 1
            else:
                embedding = result
                missing = 0

        entry = {id_col: vid, "cnn_missing": missing}
        for i, val in enumerate(embedding):
            entry[f"cnn_{i}"] = float(val)
        rows.append(entry)

    return pd.DataFrame(rows).set_index(id_col)


if __name__ == "__main__":
    from tqdm import tqdm

    from thumbnail_performance.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

    parser = argparse.ArgumentParser(description="Extract CNN embeddings for thumbnails.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "labeled_data.csv",
        help="Path to labeled CSV used to define row order.",
    )
    parser.add_argument(
        "--thumbnail_dir",
        type=Path,
        default=RAW_DATA_DIR.parent / "thumbnails" / "images",
        help="Root directory containing channel subfolders with thumbnails.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "cnn_embeddings.npy",
        help="Path to save CNN embedding array.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    model = build_embedding_model(device=args.device)
    all_embeddings = np.zeros((len(df), EMBEDDING_DIM), dtype=np.float32)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="CNN embeddings"):
        vid = str(row["Id"])
        channel = str(row["Channel"])
        img_path = resolve_thumbnail_path(args.thumbnail_dir, channel, vid)

        if img_path is not None:
            result = _embed_image(img_path, model, device=args.device)
            if result is not None:
                all_embeddings[i] = result

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, all_embeddings)
    print(f"Saved {args.output_path.name} - shape {all_embeddings.shape}")
