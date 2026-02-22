# src/04_diagnose.py
"""
Diagnostic script to check Clinical-T5 / sentence-transformer embedding quality.
Run as:  python -m src.04_diagnose   OR   python src/04_diagnose.py
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Import fix: works both as package module and direct script ─────────────────
try:
    from .config import EMBEDDINGS_CSV, FEATURES_CSV, FIGURES_DIR, RESULTS_DIR
    from .embedding_utils import validate_embeddings
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import EMBEDDINGS_CSV, FEATURES_CSV, FIGURES_DIR, RESULTS_DIR
    from embedding_utils import validate_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_diagnostics():
    logger.info("Starting Clinical Embedding Diagnostics ...")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if not os.path.exists(EMBEDDINGS_CSV):
        logger.error("Embeddings not found at %s. Run 02_embed.py first.", EMBEDDINGS_CSV)
        return
    if not os.path.exists(FEATURES_CSV):
        logger.error("Features not found at %s. Run 01_extract.py first.", FEATURES_CSV)
        return

    emb_df  = pd.read_csv(EMBEDDINGS_CSV)
    feat_df = pd.read_csv(FEATURES_CSV, low_memory=False)

    # ── 2. Align on hadm_id ───────────────────────────────────────────────────
    merged = feat_df[["hadm_id", "readmit_30"]].merge(emb_df, on="hadm_id", how="inner")
    logger.info("Matched samples: %d", len(merged))

    text_cols = [c for c in emb_df.columns if c.startswith("ct5_")]
    if not text_cols:
        logger.error("No ct5_* columns found in %s.", EMBEDDINGS_CSV)
        return

    X = merged[text_cols].values
    y = merged["readmit_30"].values

    # ── 3. Quality checks ─────────────────────────────────────────────────────
    is_valid, issues, metrics = validate_embeddings(X, y)
    logger.info("Valid: %s | Metrics: %s", is_valid, metrics)
    if issues:
        for issue in issues:
            logger.warning("  %s", issue)

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA explained variance
    n_components = min(10, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    axes[0].set_title("PCA Explained Variance (Embeddings)")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance Ratio")

    # Mean embedding per class
    axes[1].scatter(range(len(text_cols)), X[y == 0].mean(axis=0),
                    alpha=0.5, s=8, label="No readmit (0)")
    axes[1].scatter(range(len(text_cols)), X[y == 1].mean(axis=0),
                    alpha=0.5, s=8, label="Readmit (1)")
    axes[1].legend()
    axes[1].set_title("Mean Embedding Values by Class")
    axes[1].set_xlabel("Embedding Dimension")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "embedding_diagnostics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Plot saved -> %s", out_path)

    # ── 5. Save report ────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_samples":  int(len(merged)),
        "n_dims":     int(len(text_cols)),
        "is_valid":   bool(is_valid),
        "issues":     issues,
        "metrics":    {k: float(v) for k, v in metrics.items()},
    }
    report_path = os.path.join(RESULTS_DIR, "embedding_diagnostics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved -> %s", report_path)
    logger.info("Diagnostics complete. Valid: %s", is_valid)


if __name__ == "__main__":
    run_diagnostics()