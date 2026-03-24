from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger
from tqdm import tqdm
import typer
import torch
from torch.utils.data import Dataset

from thumbnail_performance.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def read_csv_with_fallback(path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Read CSV files that may have been saved with a legacy encoding.

    The dataset includes channel names such as "Bon Appétit", and some CSVs in this
    project were written in a non-UTF-8 encoding. We try UTF-8 first, then fall back
    to common single-byte encodings used by spreadsheet exports.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        f"Unable to decode CSV at {path} using tried encodings: {encodings}",
    )

def parse_abbreviated_numeric(val):
    """
    Parses strings like '10M views' or '3.35M subscribers' into float values.
    """
    if pd.isna(val) or not isinstance(val, str): 
        return np.nan
    
    # Isolate the number/abbreviation before the first space and remove commas
    val = val.split(' ')[0].upper().replace(',', '')
    
    # Handle standard multipliers
    if 'M' in val: 
        return float(val.replace('M', '')) * 1e6
    if 'K' in val: 
        return float(val.replace('K', '')) * 1e3
    if 'B' in val: 
        return float(val.replace('B', '')) * 1e9
    
    return float(val)

@app.command()
def main(
    # Set default paths based on the project's config
    input_path: Path = RAW_DATA_DIR / "data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "labeled_data.csv",
    test_size: float = 0.2,
    random_state: int = 42
):
    logger.info(f"Loading dataset from {input_path}...")
    df = read_csv_with_fallback(input_path)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Data Cleaning: Drop unnecessary text columns
    cols_to_drop = [col for col in ['CC', 'Transcript', 'transcript'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped text columns: {cols_to_drop}")
    
    # Extract and format numerical features using tqdm for progress tracking
    logger.info("Parsing numeric strings for Views and Subscribers...")
    tqdm.pandas(desc="Parsing Views")
    df['views'] = df['Views'].progress_apply(parse_abbreviated_numeric)
    
    tqdm.pandas(desc="Parsing Subscribers")
    df['subscriber_count'] = df['Subscribers'].progress_apply(parse_abbreviated_numeric)
    
    # Compute normalized performance
    logger.info("Computing normalized performance (Views / Subscribers)...")
    df['normalized_performance'] = df['views'] / (df['subscriber_count'] + 1e-9)
    
    # Split data to calculate leakage safe bin boundaries
    logger.info("Splitting data to calculate leakage-safe percentile bins...")
    train_df, _ = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Calculate percentiles strictly on the training set
    percentiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_edges = train_df['normalized_performance'].quantile(percentiles).values
    
    # Extend the outer bounds to capture unseen extremes in the test set
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    logger.info(f"Computed bin edges from training set: {bin_edges}")
    
    # Apply labels mapping 0 to 4
    logger.info("Assigning engagement labels...")
    label_mapping = [0, 1, 2, 3, 4] 
    df['engagement_label'] = pd.cut(
        df['normalized_performance'], 
        bins=bin_edges, 
        labels=label_mapping, 
        include_lowest=True
    )
    
    # Ensure output directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    
    logger.success(f"Processing complete. Saved {len(df)} records.")
    logger.info(f"Class Distribution:\n{df['engagement_label'].value_counts().sort_index()}")

class ThumbnailDataset(Dataset):
    """
    Loads pre-extracted embeddings aligned to labeled_data.csv row order.

    Expected files in data/processed/:
        cnn_embeddings.npy   shape (N, 512)
        text_embeddings.npy  shape (N, 768)
        face_embeddings.npy  shape (N, 128)
    """
    def __init__(self, csv_path, cnn_path, text_path, face_path):
        import numpy as np
        df = read_csv_with_fallback(csv_path)
        self.labels = torch.tensor(
            df["engagement_label"].astype(int).values, dtype=torch.long
        )
        self.cnn  = torch.tensor(np.load(cnn_path),  dtype=torch.float32)
        self.text = torch.tensor(np.load(text_path), dtype=torch.float32)
        self.face = torch.tensor(np.load(face_path), dtype=torch.float32)
        assert len(self.cnn) == len(self.labels), \
            f"Embedding/label mismatch: {len(self.cnn)} vs {len(self.labels)}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cnn[idx], self.text[idx], self.face[idx], self.labels[idx]

if __name__ == "__main__":
    app()
