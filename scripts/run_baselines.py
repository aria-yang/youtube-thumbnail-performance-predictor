from pathlib import Path
import pandas as pd

from thumbnail_performance.config import DATA_DIR, FIGURES_DIR
from thumbnail_performance.baselines import run_baselines


def main():
    table = run_baselines(
        train_labels_csv=DATA_DIR / "splits" / "random_train.csv",
        test_labels_csv=DATA_DIR / "splits" / "random_test.csv",
        train_cnn_csv=DATA_DIR / "processed" / "cnn_embeddings_train.csv",
        test_cnn_csv=DATA_DIR / "processed" / "cnn_embeddings_test.csv",
        figures_dir=FIGURES_DIR,
    )
    table.to_csv(DATA_DIR / "processed" / "baseline_results.csv")
    print("Done.")


if __name__ == "__main__":
    # Quick smoke test using same split for train/test (just to verify code runs)
    table = run_baselines(
        train_labels_csv=DATA_DIR / "splits" / "random_train.csv",
        test_labels_csv=DATA_DIR / "splits" / "random_train.csv",
        train_cnn_csv=DATA_DIR / "processed" / "cnn_embeddings.csv",
        test_cnn_csv=DATA_DIR / "processed" / "cnn_embeddings.csv",
        figures_dir=FIGURES_DIR,
    )