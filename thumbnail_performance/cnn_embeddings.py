from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image



EMBEDDING_DIM = 512  # ResNet-18 penultimate layer output size

# ImageNet normalisation — required since ResNet-18 weights were trained on it
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_embedding_model(device: str = "cpu") -> nn.Module:
    """
    Load pretrained ResNet-18, remove the classification head,
    and freeze all weights.

    ResNet-18 architecture (simplified):
        conv1 -> bn1 -> relu -> maxpool
        -> layer1 -> layer2 -> layer3 -> layer4   (residual blocks)
        -> avgpool                                  (global average pool)
        -> fc                                       (1000-class classifier) <- REMOVED

    We replace fc with nn.Identity() so forward() returns the raw
    512-dim average-pooled feature vector instead of class logits.
    """
    # Load ImageNet-pretrained weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Remove the classification head — replace with a no-op
    model.fc = nn.Identity()

    # Freeze all parameters so no weights are updated during any training loop
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model.to(device)


def resolve_thumbnail_path(thumbnail_root: Path, channel: str, video_id: str) -> Optional[Path]:
    channel_dir = thumbnail_root / channel
    if not channel_dir.exists():
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = channel_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def _embed_image(img_path: Path, model: nn.Module, device: str) -> Optional[np.ndarray]:
    """Return a (512,) embedding for a single image, or None on failure."""
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        with torch.no_grad():
            embedding = model(tensor)                     # (1, 512)
        return embedding.squeeze(0).cpu().numpy()         # (512,)
    except Exception:
        return None



def extract_cnn_embeddings(
    df: pd.DataFrame,
    thumbnail_dir: Path,
    id_col: str = "Id",
    channel_col: str = "Channel",
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Extract 512-dim ResNet-18 embeddings for every thumbnail in df.

    Returns a DataFrame indexed by id_col with columns:
        cnn_0, cnn_1, ..., cnn_511   (embedding dimensions)
        cnn_missing                  (1 if thumbnail not found / failed)
    """
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

    df = pd.read_csv(PROCESSED_DATA_DIR / "labeled_data.csv")
    thumbnail_dir = RAW_DATA_DIR.parent / "thumbnails" / "images"

    model = build_embedding_model(device="cpu")
    all_embeddings = np.zeros((len(df), EMBEDDING_DIM), dtype=np.float32)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="CNN embeddings"):
        vid = str(row["Id"])
        channel = str(row["Channel"])
        img_path = resolve_thumbnail_path(thumbnail_dir, channel, vid)

        if img_path is not None:
            result = _embed_image(img_path, model, device="cpu")
            if result is not None:
                all_embeddings[i] = result

    np.save(PROCESSED_DATA_DIR / "cnn_embeddings.npy", all_embeddings)
    print(f"Saved cnn_embeddings.npy — shape {all_embeddings.shape}")
