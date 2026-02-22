# src/06_visualize.py
"""
Journal-Grade Visualization Suite.
Run as:  python -m src.06_visualize   OR   python src/06_visualize.py
"""

import os
import sys
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ── Import fix: works both as package module and direct script ─────────────────
try:
    from .config import FIGURES_DIR, RESULTS_DIR, MAIN_MODEL_PKL, FEATURES_CSV
    from .embedding_utils import _extract_primary_model
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import FIGURES_DIR, RESULTS_DIR, MAIN_MODEL_PKL, FEATURES_CSV
    from embedding_utils import _extract_primary_model

plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class JournalVisualizer:

    def __init__(self):
        self.journal_dir   = os.path.join(FIGURES_DIR, "journal")
        os.makedirs(self.journal_dir, exist_ok=True)
        self.model_data    = None
        self.primary_model = None   # actual LightGBM estimator

    def load_data(self):
        if not os.path.exists(MAIN_MODEL_PKL):
            logger.error("Model not found at %s. Run 03_train.py first.", MAIN_MODEL_PKL)
            return False
        self.model_data    = joblib.load(MAIN_MODEL_PKL)
        # ── FIX: pkl stores list of (name, model) under 'models', not 'model'
        self.primary_model = _extract_primary_model(self.model_data)
        logger.info("Model loaded.")
        return True

    # ── Plot 1: Feature Importance ────────────────────────────────────────────

    def plot_top_features(self, top_n: int = 15):
        if self.primary_model is None:
            return
        logger.info("Generating feature importance plot ...")

        feats = self.model_data["features"]
        imp   = self.primary_model.feature_importances_

        df_imp = (
            pd.DataFrame({"feature": feats, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        if HAS_SEABORN:
            import seaborn as sns
            sns.barplot(data=df_imp, x="importance", y="feature",
                        palette="viridis", ax=ax)
        else:
            ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])

        ax.set_title(f"Top {top_n} Predictive Clinical Features (LightGBM gain)")
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        out = os.path.join(self.journal_dir, "fig_feature_importance.png")
        plt.savefig(out, dpi=300)
        plt.close()
        logger.info("Feature importance plot saved -> %s", out)

    # ── Plot 2: Training results from JSON ────────────────────────────────────

    def plot_results_summary(self):
        report_path = os.path.join(RESULTS_DIR, "training_report.json")
        if not os.path.exists(report_path):
            logger.warning("training_report.json not found — skipping results plot.")
            return

        import json
        with open(report_path) as f:
            report = json.load(f)

        metrics = {
            "AUROC (cal)": report.get("auroc_calibrated", 0),
            "AUPRC":       report.get("auprc", 0),
            "Recall@thr":  report.get("recall_readmit1", 0),
            "Precision":   report.get("precision_readmit1", 0),
            "F1":          report.get("f1_readmit1", 0),
        }

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                      color=["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"])
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Summary")
        ax.set_ylabel("Score")
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        out = os.path.join(self.journal_dir, "fig_results_summary.png")
        plt.savefig(out, dpi=300)
        plt.close()
        logger.info("Results summary plot saved -> %s", out)

    # ── Plot 3: Ablation comparison ───────────────────────────────────────────

    def plot_ablation(self):
        report_path = os.path.join(RESULTS_DIR, "training_report.json")
        if not os.path.exists(report_path):
            return

        import json
        with open(report_path) as f:
            report = json.load(f)

        ablation = report.get("ablation", {})
        if not ablation:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(ablation.keys()), list(ablation.values()), color="#607D8B")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Ablation Study — AUROC by Feature Set")
        ax.set_ylabel("AUROC")
        for i, (k, v) in enumerate(ablation.items()):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
        plt.tight_layout()
        out = os.path.join(self.journal_dir, "fig_ablation.png")
        plt.savefig(out, dpi=300)
        plt.close()
        logger.info("Ablation plot saved -> %s", out)

    def run(self):
        if not self.load_data():
            return
        self.plot_top_features()
        self.plot_results_summary()
        self.plot_ablation()
        logger.info("All journal visualizations saved -> %s", self.journal_dir)


if __name__ == "__main__":
    viz = JournalVisualizer()
    viz.run()