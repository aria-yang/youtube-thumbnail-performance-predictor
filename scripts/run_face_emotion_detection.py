from pathlib import Path
import pandas as pd

from thumbnail_performance.config import DATA_DIR
from thumbnail_performance.face_emotion_detection import extract_face_emotion_features

def main():
    split_csv = DATA_DIR / "splits" / "random_train.csv"
    thumbnail_dir = DATA_DIR / "thumbnails" / "images"
    out_csv = DATA_DIR / "processed" / "face_emotion_features.csv"

    df = pd.read_csv(split_csv)
    feats = extract_face_emotion_features(df, thumbnail_dir=thumbnail_dir, id_col="Id", device="cpu")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_csv)
    print(f"Saved features: {feats.shape} -> {out_csv}")

if __name__ == "__main__":
    main()