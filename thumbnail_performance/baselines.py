from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def _load_cnn_features(cnn_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cnn_csv, index_col=0)
    cnn_cols = [c for c in df.columns if c.startswith("cnn_") and c != "cnn_missing"]
    return df[cnn_cols]


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute all required metrics. Handles multiclass via macro averaging."""
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2

    auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    auprc = average_precision_score(y_true, y_prob, average="macro")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {"AUROC": round(auroc, 4), "AUPRC": round(auprc, 4),
            "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4)}


def run_random_baseline(X_train, y_train, X_test, y_test) -> Dict:
    clf = DummyClassifier(strategy="stratified", random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    return _compute_metrics(y_test, y_prob, y_pred)


def run_cnn_baseline(X_train, y_train, X_test, y_test) -> Dict:
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    return _compute_metrics(y_test, y_prob, y_pred)


def plot_auroc_comparison(results: Dict[str, Dict], out_path: Path) -> None:
    """
    Figure 2: Bar chart of AUROC for Random vs CNN-only.
    Clean, paper-ready style.
    """
    models = list(results.keys())
    aurocs = [results[m]["AUROC"] for m in models]
    colors = ["#b0b0b0", "#2196F3"][:len(models)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, aurocs, color=colors, width=0.4, edgecolor="black", linewidth=0.8)

    # Annotate bars with values
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Reference line at 0.5 (random chance)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random chance (0.5)")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (macro-average)", fontsize=12)
    ax.set_title("Figure 2: Baseline AUROC Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out_path}")


def run_baselines(
    train_labels_csv: Path,
    test_labels_csv: Path,
    train_cnn_csv: Path,
    test_cnn_csv: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    # Load labels
    train_meta = pd.read_csv(train_labels_csv, index_col=0)
    test_meta = pd.read_csv(test_labels_csv, index_col=0)

    y_train = train_meta["engagement_label"].values
    y_test = test_meta["engagement_label"].values

    # Load CNN embeddings and align indices
    train_cnn = _load_cnn_features(train_cnn_csv)
    test_cnn = _load_cnn_features(test_cnn_csv)

    common_train = train_meta.index.intersection(train_cnn.index)
    common_test = test_meta.index.intersection(test_cnn.index)

    X_train_cnn = train_cnn.loc[common_train].values
    y_train_cnn = train_meta.loc[common_train, "engagement_label"].values
    X_test_cnn = test_cnn.loc[common_test].values
    y_test_cnn = test_meta.loc[common_test, "engagement_label"].values

    # Dummy features for random baseline (just zeros)
    X_dummy = np.zeros((len(y_train), 1))
    X_dummy_test = np.zeros((len(y_test), 1))

    print("Running random baseline...")
    random_results = run_random_baseline(X_dummy, y_train, X_dummy_test, y_test)

    print("Running CNN-only baseline...")
    cnn_results = run_cnn_baseline(X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn)

    results = {"Random": random_results, "CNN-only": cnn_results}

    # Print comparison table
    table = pd.DataFrame(results).T
    print("\n── Baseline Results ──────────────────────────────")
    print(table.to_string())
    print("──────────────────────────────────────────────────\n")

    # Save figure
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_auroc_comparison(results, figures_dir / "figure2_baseline_auroc.png")

    return table