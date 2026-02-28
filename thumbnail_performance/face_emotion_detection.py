from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from facenet_pytorch import MTCNN
from fer import FER

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def resolve_thumbnail_path(thumbnail_dir: Path, video_id: str) -> Optional[Path]:
    """Find <Id>.<ext> in thumbnail_dir for common image extensions."""
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = thumbnail_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def _largest_face_area_ratio(boxes: Optional[np.ndarray], w: int, h: int) -> float:
    if boxes is None or len(boxes) == 0:
        return 0.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    img_area = float(w * h) if w > 0 and h > 0 else 0.0
    return float(np.max(areas)) / img_area if img_area > 0 else 0.0


def _dominant_emotion(img_np: np.ndarray, fer_model: FER) -> Optional[str]:
    """Pick the single highest-confidence emotion across all detected faces."""
    try:
        dets = fer_model.detect_emotions(img_np)
    except Exception:
        return None
    if not dets:
        return None

    best_e, best_s = None, -1.0
    for det in dets:
        emos: Dict[str, float] = det.get("emotions", {}) or {}
        if not emos:
            continue
        e = max(emos, key=emos.get)
        s = float(emos[e])
        if s > best_s:
            best_e, best_s = e, s
    return best_e


def extract_face_emotion_features(
    df: pd.DataFrame,
    thumbnail_dir: Path,
    id_col: str = "Id",
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Extract face + emotion features indexed by Id.

    Outputs:
      - num_faces
      - largest_face_area_ratio
      - emotion_<7> one-hots
      - emotion_unknown
    """
    if id_col not in df.columns:
        raise KeyError(f"Expected column '{id_col}' in dataframe.")

    thumbnail_dir = Path(thumbnail_dir)
    mtcnn = MTCNN(keep_all=True, device=device)
    fer_model = FER(mtcnn=True)

    rows: List[Dict] = []

    for vid in df[id_col].astype(str).tolist():
        img_path = resolve_thumbnail_path(thumbnail_dir, vid)
        if img_path is None:
            rows.append(
                {id_col: vid, "num_faces": 0, "largest_face_area_ratio": 0.0, "dominant_emotion": None}
            )
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            rows.append(
                {id_col: vid, "num_faces": 0, "largest_face_area_ratio": 0.0, "dominant_emotion": None}
            )
            continue

        w, h = img.size
        try:
            boxes, _ = mtcnn.detect(img)
        except Exception:
            boxes = None

        num_faces = int(0 if boxes is None else len(boxes))
        ratio = _largest_face_area_ratio(boxes, w, h)

        emo = _dominant_emotion(np.array(img), fer_model)

        rows.append(
            {id_col: vid, "num_faces": num_faces, "largest_face_area_ratio": float(ratio), "dominant_emotion": emo}
        )

    feats = pd.DataFrame(rows).set_index(id_col)

    for e in EMOTIONS:
        feats[f"emotion_{e}"] = (feats["dominant_emotion"] == e).astype(int)
    feats["emotion_unknown"] = feats["dominant_emotion"].isna().astype(int)

    return feats.drop(columns=["dominant_emotion"])