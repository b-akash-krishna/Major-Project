# src/05_analyze.py
"""
SHAP Interpretability Analysis for ACAGN (base ensemble).
Run as:  python -m src.05_analyze   OR   python src/05_analyze.py
"""

import os
import sys
import json
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── Import fix: works both as package module and direct script ─────────────────
try:
    from .config import (
        MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY, FEATURES_CSV, EMBEDDINGS_CSV,
        FIGURES_DIR, RESULTS_DIR, RANDOM_STATE, SHAP_N_SAMPLES,
    )
    from .embedding_utils import _extract_primary_model
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY, FEATURES_CSV, EMBEDDINGS_CSV,
        FIGURES_DIR, RESULTS_DIR, RANDOM_STATE, SHAP_N_SAMPLES,
    )
    from embedding_utils import _extract_primary_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SHAPAnalyzer:

    def __init__(self):
        self.model_data  = None
        self.primary_model = None   # actual LightGBM estimator (not a list)
        self.df          = None
        self.explainer   = None

    def load_resources(self):
        """Loads the trained model and fused feature DataFrame."""
        model_path = MAIN_MODEL_PKL
        if not os.path.exists(model_path) and os.path.exists(MAIN_MODEL_PKL_LEGACY):
            logger.warning(
                "Model not found at %s; falling back to legacy path %s",
                model_path,
                MAIN_MODEL_PKL_LEGACY,
            )
            model_path = MAIN_MODEL_PKL_LEGACY
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run 03_train.py first.")

        self.model_data = joblib.load(model_path)
        # ── FIX: pkl stores list of (name, model) tuples under 'models', not 'model'
        self.primary_model = _extract_primary_model(self.model_data)

        # Load data — prefer pruned CSV to avoid OOM on the 400MB full file
        features = self.model_data["features"]
        pruned_path = FEATURES_CSV.replace(".csv", "_pruned.csv")
        csv_path = pruned_path if os.path.exists(pruned_path) else FEATURES_CSV

        # Only load columns the model actually needs (+ hadm_id)
        tab_features = [f for f in features if not f.startswith("ct5_")]
        use_cols = list(set(["hadm_id", "readmit_30"] + tab_features))

        try:
            tabular = pd.read_csv(csv_path, usecols=lambda c: c in use_cols,
                                  low_memory=False)
        except Exception:
            # Last resort: read in chunks and concat
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=50_000, low_memory=False):
                keep = [c for c in use_cols if c in chunk.columns]
                chunks.append(chunk[keep])
            tabular = pd.concat(chunks, ignore_index=True)

        if os.path.exists(EMBEDDINGS_CSV):
            embs    = pd.read_csv(EMBEDDINGS_CSV)
            self.df = tabular.merge(embs, on="hadm_id", how="left").fillna(0)
        else:
            self.df = tabular.fillna(0)
            logger.warning("Embeddings not found — using tabular features only.")

        logger.info("Resources loaded. Data shape: %s", self.df.shape)
        return self.df

    def analyze(self, n_samples: int = SHAP_N_SAMPLES):
        """Runs SHAP TreeExplainer on a random subsample."""
        features = self.model_data["features"]

        # Keep only features that exist in the loaded DataFrame
        available = [f for f in features if f in self.df.columns]
        missing   = set(features) - set(available)
        if missing:
            logger.warning("%d features missing from data: %s ...",
                           len(missing), list(missing)[:5])

        X        = self.df[available]
        X_sample = X.sample(n=min(n_samples, len(X)), random_state=RANDOM_STATE)

        logger.info("Computing SHAP values for %d samples ...", len(X_sample))
        self.explainer = shap.TreeExplainer(self.primary_model)
        shap_values    = self.explainer.shap_values(X_sample)

        # LightGBM binary returns list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values, X_sample

    def generate_plots(self, shap_values: np.ndarray, X_sample: pd.DataFrame):
        """Saves SHAP summary plot and feature importance CSV."""
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Summary beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, max_display=30, show=False)
        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, "shap_summary.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved -> %s", out_path)

        # Feature importance CSV
        importance = np.abs(shap_values).mean(axis=0)
        feat_imp = (
            pd.DataFrame({"feature": X_sample.columns, "importance": importance})
            .sort_values("importance", ascending=False)
        )
        csv_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
        feat_imp.to_csv(csv_path, index=False)
        logger.info("Feature importance CSV saved -> %s", csv_path)

        # Top-20 bar chart
        top20 = feat_imp.head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top20["feature"][::-1], top20["importance"][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 20 Features by SHAP Importance")
        plt.tight_layout()
        bar_path = os.path.join(FIGURES_DIR, "shap_bar.png")
        plt.savefig(bar_path, dpi=150)
        plt.close()
        logger.info("SHAP bar chart saved -> %s", bar_path)

    def run(self, n_samples: int = SHAP_N_SAMPLES):
        self.load_resources()
        shap_vals, X_sample = self.analyze(n_samples=n_samples)
        self.generate_plots(shap_vals, X_sample)
        logger.info("SHAP analysis complete.")


if __name__ == "__main__":
    analyzer = SHAPAnalyzer()
    analyzer.run()
