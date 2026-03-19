from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from facenet_pytorch import MTCNN
from deepface import DeepFace

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def resolve_thumbnail_path(thumbnail_root: Path, channel: str, video_id: str) -> Optional[Path]:
    channel_dir = thumbnail_root / channel
    if not channel_dir.exists():
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = channel_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def _largest_face_area_ratio(boxes: Optional[np.ndarray], w: int, h: int) -> float:
    if boxes is None or len(boxes) == 0:
        return 0.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    img_area = float(w * h) if w > 0 and h > 0 else 0.0
    return float(np.max(areas)) / img_area if img_area > 0 else 0.0


def _dominant_emotion(img_np: np.ndarray) -> Optional[str]:
    try:
        result = DeepFace.analyze(img_np, actions=["emotion"], enforce_detection=False, silent=True)
        return result[0]["dominant_emotion"]
    except Exception:
        return None


def extract_face_emotion_features(
    df: pd.DataFrame,
    thumbnail_dir: Path,
    id_col: str = "Id",
    channel_col: str = "Channel",
    device: str = "cpu",
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    if id_col not in df.columns:
        raise KeyError(f"Expected column '{id_col}' in dataframe.")

    has_channel = channel_col in df.columns
    thumbnail_dir = Path(thumbnail_dir)
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load cache if it exists (allows resuming)
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=id_col)
        done_ids = set(cached.index.astype(str))
        print(f"Resuming — {len(done_ids)} already done")
    else:
        cached = None
        done_ids = set()

    rows: List[Dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Face features"):
        vid = str(row[id_col])

        # Skip already processed
        if vid in done_ids:
            continue

        channel = str(row[channel_col]) if has_channel else ""
        img_path = resolve_thumbnail_path(thumbnail_dir, channel, vid) if has_channel else None

        if img_path is None:
            rows.append({id_col: vid, "num_faces": 0,
                         "largest_face_area_ratio": 0.0, "dominant_emotion": None})
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            rows.append({id_col: vid, "num_faces": 0,
                         "largest_face_area_ratio": 0.0, "dominant_emotion": None})
            continue

        w, h = img.size
        try:
            boxes, _ = mtcnn.detect(img)
        except Exception:
            boxes = None

        num_faces = int(0 if boxes is None else len(boxes))
        ratio = _largest_face_area_ratio(boxes, w, h)
        emo = _dominant_emotion(np.array(img))

        rows.append({id_col: vid, "num_faces": num_faces,
                     "largest_face_area_ratio": float(ratio), "dominant_emotion": emo})

    # Merge new rows with cache
    new_df = pd.DataFrame(rows).set_index(id_col)
    feats = pd.concat([cached, new_df]) if cached is not None else new_df

    # Save cache
    if cache_path:
        feats.to_csv(cache_path)
        print(f"Cache saved to {cache_path}")

    for e in EMOTIONS:
        feats[f"emotion_{e}"] = (feats["dominant_emotion"] == e).astype(int)
    feats["emotion_unknown"] = feats["dominant_emotion"].isna().astype(int)

    return feats.drop(columns=["dominant_emotion"])


if __name__ == "__main__":
    from thumbnail_performance.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

    df = pd.read_csv(PROCESSED_DATA_DIR / "labeled_data.csv")

    feats = extract_face_emotion_features(
        df,
        thumbnail_dir=RAW_DATA_DIR.parent / "thumbnails" / "images",
        cache_path=PROCESSED_DATA_DIR / "face_cache.csv",  # resumes if interrupted
    )

    feat_cols = ["num_faces", "largest_face_area_ratio"] + \
                [f"emotion_{e}" for e in EMOTIONS] + ["emotion_unknown"]

    arr = feats.reindex(df["Id"].astype(str))[feat_cols].values.astype(np.float32)
    np.save(PROCESSED_DATA_DIR / "face_embeddings.npy", arr)
    print(f"Saved face_embeddings.npy — shape {arr.shape}")
