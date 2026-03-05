import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from thumbnail_performance.baselines import (
    run_random_baseline,
    run_cnn_baseline,
    plot_auroc_comparison,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    """Small synthetic dataset with 5 classes matching project labels (0-4)."""
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 50
    n_classes = 5

    X_train = rng.standard_normal((n_train, 512))
    X_test = rng.standard_normal((n_test, 512))
    y_train = rng.integers(0, n_classes, n_train)
    y_test = rng.integers(0, n_classes, n_test)

    return X_train, y_train, X_test, y_test

def test_random_baseline_metrics(dummy_data):
    X_train, y_train, X_test, y_test = dummy_data
    X_dummy_train = np.zeros((len(y_train), 1))
    X_dummy_test = np.zeros((len(y_test), 1))

    results = run_random_baseline(X_dummy_train, y_train, X_dummy_test, y_test)

    assert set(results.keys()) == {"AUROC", "AUPRC", "Accuracy", "Precision", "Recall"}
    assert 0.0 <= results["AUROC"] <= 1.0
    assert 0.0 <= results["Accuracy"] <= 1.0


def test_cnn_baseline_metrics(dummy_data):
    X_train, y_train, X_test, y_test = dummy_data
    results = run_cnn_baseline(X_train, y_train, X_test, y_test)

    assert set(results.keys()) == {"AUROC", "AUPRC", "Accuracy", "Precision", "Recall"}
    assert 0.0 <= results["AUROC"] <= 1.0


def test_cnn_beats_random(dummy_data):
    """CNN embeddings should learn signal; AUROC should exceed random on synthetic data."""
    X_train, y_train, X_test, y_test = dummy_data
    X_dummy_train = np.zeros((len(y_train), 1))
    X_dummy_test = np.zeros((len(y_test), 1))

    random_results = run_random_baseline(X_dummy_train, y_train, X_dummy_test, y_test)
    cnn_results = run_cnn_baseline(X_train, y_train, X_test, y_test)

    assert cnn_results["AUROC"] >= random_results["AUROC"]


def test_plot_saves_file(tmp_path):
    results = {
        "Random": {"AUROC": 0.50, "AUPRC": 0.20, "Accuracy": 0.20, "Precision": 0.20, "Recall": 0.20},
        "CNN-only": {"AUROC": 0.72, "AUPRC": 0.45, "Accuracy": 0.41, "Precision": 0.39, "Recall": 0.41},
    }
    out = tmp_path / "figure2_baseline_auroc.png"
    plot_auroc_comparison(results, out)
    assert out.exists()
    assert out.stat().st_size > 0