from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from thumbnail_performance.config import PROCESSED_DATA_DIR


def load_title_embedding_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    cached = pd.read_csv(cache_path, index_col="Id")
    cached.index = cached.index.astype(str)
    return cached


def build_title_embedding_cache(
    df: pd.DataFrame,
    cache_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    text_col: str = "Title",
    id_col: str = "Id",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()
    df[id_col] = df[id_col].astype(str)

    cached = load_title_embedding_cache(cache_path)
    missing_df = df.loc[~df[id_col].isin(cached.index)].copy()
    print(f"Title stage: {len(cached)} cached, {len(missing_df)} to process")

    if len(missing_df) == 0:
        return cached

    model = SentenceTransformer(model_name, device=device)
    texts = missing_df[text_col].fillna("").astype(str).tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    records = []
    for idx, row in enumerate(tqdm(missing_df.itertuples(index=False), total=len(missing_df), desc="Title cache")):
        entry = {id_col: str(getattr(row, id_col))}
        for dim_idx, value in enumerate(embeddings[idx]):
            entry[f"text_{dim_idx}"] = float(value)
        records.append(entry)

    new_df = pd.DataFrame(records).set_index(id_col)
    cached = pd.concat([cached, new_df]) if not cached.empty else new_df
    cached = cached[~cached.index.duplicated(keep="first")]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_csv(cache_path)
    print(f"Saved title embedding cache to {cache_path}")
    return cached


def build_title_embedding_array(
    csv_path: Path,
    output_path: Path,
    cache_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    text_col: str = "Title",
    id_col: str = "Id",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df[id_col] = df[id_col].astype(str)

    cached = build_title_embedding_cache(
        df=df,
        cache_path=cache_path,
        model_name=model_name,
        text_col=text_col,
        id_col=id_col,
        batch_size=batch_size,
        device=device,
    )

    feat_cols = [col for col in cached.columns if col.startswith("text_")]
    feat_cols.sort(key=lambda col: int(col.split("_")[1]))
    arr = cached.reindex(df[id_col])[feat_cols].fillna(0).values.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    print(f"Saved {output_path.name} - shape {arr.shape}")
    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build sentence-transformer title embeddings.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_labeled_data.csv",
        help="Path to labeled CSV used to define row order.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_text_embeddings.npy",
        help="Path to save title embedding array.",
    )
    parser.add_argument(
        "--cache_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "merged_title_embedding_cache.csv",
        help="CSV cache path for resumable title embeddings.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Model device override, e.g. cpu or cuda.",
    )
    args = parser.parse_args()

    build_title_embedding_array(
        csv_path=args.csv_path,
        output_path=args.output_path,
        cache_path=args.cache_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
