import numpy as np
import pandas as pd
from pathlib import Path

from thumbnail_performance.cnn_embeddings import extract_cnn_embeddings, EMBEDDING_DIM


def test_contract_no_images(tmp_path: Path):
    """
    With no thumbnails on disk, all rows should return zero embeddings
    and cnn_missing = 1.
    """
    df = pd.DataFrame({"Id": ["a", "b", "c"]})
    feats = extract_cnn_embeddings(df, thumbnail_dir=tmp_path, id_col="Id", device="cpu")

    # Shape: 3 rows, 512 cnn dims + 1 missing flag
    assert feats.shape == (3, EMBEDDING_DIM + 1), f"Unexpected shape: {feats.shape}"

    # All marked missing
    assert (feats["cnn_missing"] == 1).all()

    # All embeddings are zero vectors
    cnn_cols = [f"cnn_{i}" for i in range(EMBEDDING_DIM)]
    assert (feats[cnn_cols].values == 0).all()


def test_contract_with_channel_no_images(tmp_path: Path):
    """Channel column present but no files — same zero-default behaviour."""
    df = pd.DataFrame({"Id": ["x", "y"], "Channel": ["ChanA", "ChanB"]})
    feats = extract_cnn_embeddings(
        df, thumbnail_dir=tmp_path, id_col="Id", channel_col="Channel", device="cpu"
    )

    assert feats.shape[0] == 2
    assert (feats["cnn_missing"] == 1).all()


def test_embedding_dim():
    """EMBEDDING_DIM constant should be 512 for ResNet-18."""
    assert EMBEDDING_DIM == 512