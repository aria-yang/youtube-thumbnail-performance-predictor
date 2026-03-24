from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from deepface import DeepFace
from facenet_pytorch import MTCNN


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


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


def _largest_face_area_ratio(boxes: Optional[np.ndarray], w: int, h: int) -> float:
    if boxes is None or len(boxes) == 0:
        return 0.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    img_area = float(w * h) if w > 0 and h > 0 else 0.0
    return float(np.max(areas)) / img_area if img_area > 0 else 0.0


def _dominant_emotion(img_np: np.ndarray) -> Optional[str]:
    try:
        result = DeepFace.analyze(
            img_np, actions=["emotion"], enforce_detection=False, silent=True
        )
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

    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=id_col)
        done_ids = set(cached.index.astype(str))
        print(f"Resuming - {len(done_ids)} already done")
    else:
        cached = None
        done_ids = set()

    rows: List[Dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Face features"):
        vid = str(row[id_col])

        if vid in done_ids:
            continue

        channel = str(row[channel_col]) if has_channel else ""
        img_path = resolve_thumbnail_path(thumbnail_dir, channel, vid) if has_channel else None

        if img_path is None:
            rows.append(
                {
                    id_col: vid,
                    "num_faces": 0,
                    "largest_face_area_ratio": 0.0,
                    "dominant_emotion": None,
                }
            )
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            rows.append(
                {
                    id_col: vid,
                    "num_faces": 0,
                    "largest_face_area_ratio": 0.0,
                    "dominant_emotion": None,
                }
            )
            continue

        w, h = img.size
        try:
            boxes, _ = mtcnn.detect(img)
        except Exception:
            boxes = None

        rows.append(
            {
                id_col: vid,
                "num_faces": int(0 if boxes is None else len(boxes)),
                "largest_face_area_ratio": float(_largest_face_area_ratio(boxes, w, h)),
                "dominant_emotion": _dominant_emotion(np.array(img)),
            }
        )

    new_df = pd.DataFrame(rows).set_index(id_col)
    feats = pd.concat([cached, new_df]) if cached is not None else new_df

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(cache_path)
        print(f"Cache saved to {cache_path}")

    for emotion in EMOTIONS:
        feats[f"emotion_{emotion}"] = (feats["dominant_emotion"] == emotion).astype(int)
    feats["emotion_unknown"] = feats["dominant_emotion"].isna().astype(int)

    return feats.drop(columns=["dominant_emotion"])


if __name__ == "__main__":
    from thumbnail_performance.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

    parser = argparse.ArgumentParser(description="Extract face and emotion features.")
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
        default=PROCESSED_DATA_DIR / "face_embeddings.npy",
        help="Path to save face feature array.",
    )
    parser.add_argument(
        "--cache_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "face_cache.csv",
        help="CSV cache path for resumable face feature extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    feats = extract_face_emotion_features(
        df,
        thumbnail_dir=args.thumbnail_dir,
        cache_path=args.cache_path,
        device=args.device,
    )

    feat_cols = ["num_faces", "largest_face_area_ratio"] + [
        f"emotion_{emotion}" for emotion in EMOTIONS
    ] + ["emotion_unknown"]

    arr = feats.reindex(df["Id"].astype(str))[feat_cols].values.astype(np.float32)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, arr)
    print(f"Saved {args.output_path.name} - shape {arr.shape}")
