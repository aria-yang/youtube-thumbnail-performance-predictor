import pandas as pd
from pathlib import Path

from thumbnail_performance.face_emotion_detection import extract_face_emotion_features


def test_contract_no_images(tmp_path: Path):
    """
    Contract test: when no thumbnails exist on disk, the function must still
    return a well-formed DataFrame with one row per input video.

    The Channel column is intentionally omitted here to verify that the
    function degrades gracefully when channel info is unavailable.
    """
    df = pd.DataFrame({"Id": ["a", "b", "c"]})
    feats = extract_face_emotion_features(df, thumbnail_dir=tmp_path, id_col="Id", device="cpu")

    # Shape
    assert feats.shape[0] == 3, f"Expected 3 rows, got {feats.shape[0]}"

    # Required columns present
    assert "num_faces" in feats.columns
    assert "largest_face_area_ratio" in feats.columns
    assert "emotion_unknown" in feats.columns

    # One-hot emotion columns present for all 7 emotions
    for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
        assert f"emotion_{emotion}" in feats.columns, f"Missing column: emotion_{emotion}"

    # Value sanity checks
    assert feats["num_faces"].min() >= 0
    assert (feats["largest_face_area_ratio"] >= 0).all()

    # With no images, everything should default to 0 faces / unknown emotion
    assert (feats["num_faces"] == 0).all()
    assert (feats["emotion_unknown"] == 1).all()


def test_contract_with_channel_no_images(tmp_path: Path):
    """
    Same contract but with a Channel column present and no matching files on disk.
    Should still return safe zero-valued defaults.
    """
    df = pd.DataFrame({"Id": ["x", "y"], "Channel": ["ChannelA", "ChannelB"]})
    feats = extract_face_emotion_features(
        df, thumbnail_dir=tmp_path, id_col="Id", channel_col="Channel", device="cpu"
    )

    assert feats.shape[0] == 2
    assert (feats["num_faces"] == 0).all()
    assert (feats["largest_face_area_ratio"] == 0.0).all()
    assert (feats["emotion_unknown"] == 1).all()
