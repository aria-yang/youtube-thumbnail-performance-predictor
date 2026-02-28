import pandas as pd
from pathlib import Path

from thumbnail_performance.face_emotion_detection import extract_face_emotion_features

def test_contract_no_images(tmp_path: Path):
    df = pd.DataFrame({"Id": ["a", "b", "c"]})
    feats = extract_face_emotion_features(df, thumbnail_dir=tmp_path, id_col="Id", device="cpu")

    assert feats.shape[0] == 3
    assert "num_faces" in feats.columns
    assert "largest_face_area_ratio" in feats.columns
    assert "emotion_unknown" in feats.columns
    assert feats["num_faces"].min() >= 0
    assert (feats["largest_face_area_ratio"] >= 0).all()