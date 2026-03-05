from pathlib import Path
import pandas as pd

from thumbnail_performance.config import DATA_DIR
from thumbnail_performance.cnn_embeddings import extract_cnn_embeddings


def main():
    split_csv = DATA_DIR / "splits" / "random_train.csv"
    thumbnail_dir = DATA_DIR / "thumbnails" / "images"
    out_csv = DATA_DIR / "processed" / "cnn_embeddings.csv"

    df = pd.read_csv(split_csv)
    feats = extract_cnn_embeddings(df, thumbnail_dir=thumbnail_dir, id_col="Id", device="cpu")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_csv)
    print(f"Saved embeddings: {feats.shape} -> {out_csv}")


if __name__ == "__main__":
   main()