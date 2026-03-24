"""
ACAGN: Metrics Tables + Diagrams (Paper-Ready)
=============================================

Generates standardized metric tables and comparison diagrams for:
  - Base Ensemble
  - ACAGN-Gate
  - ACAGN-Hybrid

Inputs (expected):
  - results/hybrid_predictions.csv        (hadm_id, p_base, p_gate, p_hybrid, y_base)
  - results/training_report.json          (base report; includes CV AUROC, operating points, etc.)
  - results/gate_training_report.json     (gate report)
  - results/gate_interpretability.csv     (Mann–Whitney U p-values for gate-weight analysis)

Outputs:
  - results/metrics/*.csv, *.json
  - figures/metrics/*.png
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)


@dataclass(frozen=True)
class Paths:
    results_dir: str = "results"
    figures_dir: str = "figures"

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.results_dir, "metrics")

    @property
    def metrics_fig_dir(self) -> str:
        return os.path.join(self.figures_dir, "metrics")

    @property
    def hybrid_predictions_csv(self) -> str:
        return os.path.join(self.results_dir, "hybrid_predictions.csv")

    @property
    def base_report_json(self) -> str:
        return os.path.join(self.results_dir, "training_report.json")

    @property
    def gate_report_json(self) -> str:
        return os.path.join(self.results_dir, "gate_training_report.json")

    @property
    def gate_interpretability_csv(self) -> str:
        return os.path.join(self.results_dir, "gate_interpretability.csv")


def ensure_dirs(paths: Paths) -> None:
    os.makedirs(paths.metrics_dir, exist_ok=True)
    os.makedirs(paths.metrics_fig_dir, exist_ok=True)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    c = confusion_counts(y_true, y_pred)
    return float(c["tn"] / max(c["tn"] + c["fp"], 1))


def youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = specificity_score(y_true, y_pred)
    return float(rec + spec - 1.0)


def operating_points(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Iterable[float] = np.arange(0.05, 0.70, 0.01),
) -> pd.DataFrame:
    rows: List[Dict] = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        c = confusion_counts(y_true, pred)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": float((pred == y_true).mean()),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "specificity": float(specificity_score(y_true, pred)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, pred)),
                "youden_j": float(youden_j(y_true, pred)),
                **c,
            }
        )
    return pd.DataFrame(rows)


def select_thresholds(op: pd.DataFrame) -> Dict[str, float]:
    """
    Returns a dict with strategy -> threshold.
    These strategies are computed on the same labels/probabilities passed into
    `operating_points` and are therefore *test-optimized* when called on test.

    Strategies:
      - test_opt_mcc: max MCC
      - test_opt_f1: max F1
      - test_opt_recall80: smallest threshold achieving recall >= 0.80
      - test_opt_j: max Youden-J
    """
    out: Dict[str, float] = {}

    out["test_opt_mcc"] = float(op.loc[op["mcc"].idxmax(), "threshold"])
    out["test_opt_f1"] = float(op.loc[op["f1"].idxmax(), "threshold"])
    out["test_opt_j"] = float(op.loc[op["youden_j"].idxmax(), "threshold"])

    recall80 = op[op["recall"] >= 0.80]
    if len(recall80) > 0:
        out["test_opt_recall80"] = float(recall80.sort_values("threshold").iloc[0]["threshold"])
    else:
        out["test_opt_recall80"] = float(op.sort_values("recall", ascending=False).iloc[0]["threshold"])

    return out


# ---- DeLong test for correlated AUROC ---------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks for DeLong."""
    x = np.asarray(x)
    idx = np.argsort(x)
    sorted_x = x[idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1  # 1-based
        midranks[i:j] = mid
        i = j
    out = np.empty(n, dtype=float)
    out[idx] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong implementation.
    predictions_sorted_transposed shape: (n_classifiers, n_examples) with examples sorted by label (pos first).
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    tx = np.vstack([_compute_midrank(p) for p in pos])
    ty = np.vstack([_compute_midrank(p) for p in neg])
    tz = np.vstack([_compute_midrank(p) for p in predictions_sorted_transposed])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_pvalue(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Two-sided DeLong p-value for AUROC difference between two correlated curves.
    """
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    order = np.argsort(-y_true)  # positives first
    y_sorted = y_true[order]
    preds = np.vstack([p1, p2])[:, order]
    m = int(y_sorted.sum())

    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        return float("nan")

    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(p)


# -----------------------------------------------------------------------------

def load_aligned_predictions(paths: Paths) -> Tuple[pd.DataFrame, Dict]:
    if not os.path.exists(paths.hybrid_predictions_csv):
        raise FileNotFoundError(f"Missing {paths.hybrid_predictions_csv}")

    df = pd.read_csv(paths.hybrid_predictions_csv)
    required = {"hadm_id", "p_base", "y_base", "p_gate", "y_gate", "p_hybrid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in hybrid_predictions.csv: {sorted(missing)}")

    # Deduplicate hadm_id to avoid merge amplification if any model has duplicate IDs.
    dupe_counts = df["hadm_id"].value_counts()
    n_dupe_ids = int((dupe_counts > 1).sum())
    max_dupe = int(dupe_counts.max()) if len(dupe_counts) else 1
    df_dedup = df.drop_duplicates("hadm_id", keep="first").reset_index(drop=True)

    # Ensure label agreement after alignment.
    label_mismatch = int((df_dedup["y_base"].astype(int) != df_dedup["y_gate"].astype(int)).sum())

    meta = {
        "n_rows_raw": int(len(df)),
        "n_rows_dedup": int(len(df_dedup)),
        "n_unique_hadm": int(df["hadm_id"].nunique()),
        "n_dupe_hadm_ids": n_dupe_ids,
        "max_dupe_count": max_dupe,
        "label_mismatch_rows_after_dedup": label_mismatch,
    }
    return df_dedup, meta


def compute_threshold_free_metrics(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(average_precision_score(y, probs)),
        "ece": float(compute_ece(probs, y)),
        "brier": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
    }


def plot_roc_pr(y: np.ndarray, model_probs: Dict[str, np.ndarray], out_dir: str) -> None:
    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in model_probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in model_probs.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, lw=2, label=f"{name} (AUPRC={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (Test Set)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=150)
    plt.close(fig)


def plot_metric_bars(summary: pd.DataFrame, out_dir: str) -> None:
    # Discrimination bars (threshold-free)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width / 2, summary["auroc"], width, label="AUROC")
    ax.bar(x + width / 2, summary["auprc"], width, label="AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"].tolist(), rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_title("Discrimination Metrics (Test Set)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "discrimination_metrics.png"), dpi=150)
    plt.close(fig)

    # Calibration bars
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["ece", "brier", "log_loss"]
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * 0.25, summary[m], 0.25, label=m.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"].tolist(), rotation=0)
    ax.set_title("Calibration & Probability Metrics (Test Set)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_metrics.png"), dpi=150)
    plt.close(fig)


def plot_confusion_panels(rows: pd.DataFrame, out_dir: str, strategy: str = "mcc") -> None:
    """
    Plot confusion matrices for a given strategy across models.
    """
    df = rows[rows["strategy"] == strategy].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, len(df), figsize=(5 * len(df), 4))
    if len(df) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, df.iterrows()):
        cm = np.array([[int(r["tn"]), int(r["fp"])], [int(r["fn"]), int(r["tp"])]])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{r['model']} @ {strategy} (t={r['threshold']:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrices_{strategy}.png"), dpi=150)
    plt.close(fig)


def summarize_gate_mwu(paths: Paths) -> Tuple[pd.DataFrame, Dict]:
    if not os.path.exists(paths.gate_interpretability_csv):
        return pd.DataFrame(), {"available": False}
    df = pd.read_csv(paths.gate_interpretability_csv)
    if "p_value" not in df.columns:
        return pd.DataFrame(), {"available": False}

    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["p_value"]).copy()
    df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))

    meta = {
        "available": True,
        "n_rows": int(len(df)),
        "n_significant_p05": int((df["p_value"] < 0.05).sum()),
        "n_significant_bonferroni": int(df.get("significant_bonferroni", False).astype(bool).sum())
        if "significant_bonferroni" in df.columns
        else None,
    }
    return df.sort_values("p_value", ascending=True), meta


def plot_gate_mwu_top(df_sorted: pd.DataFrame, out_dir: str, top_n: int = 20) -> None:
    if df_sorted.empty:
        return
    top = df_sorted.head(top_n).copy()
    top["label"] = top["condition"].astype(str) + " | " + top["feature"].astype(str)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["label"], top["neg_log10_p"].astype(float))
    ax.set_title(f"ACAGN-Gate Mann–Whitney U Significance (Top {top_n})")
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gate_mannwhitney_top.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    paths = Paths()
    ensure_dirs(paths)

    df, dq = load_aligned_predictions(paths)
    with open(os.path.join(paths.metrics_dir, "data_quality.json"), "w") as f:
        json.dump(dq, f, indent=2)

    y = df["y_base"].astype(int).values
    probs = {
        "Base Ensemble": df["p_base"].astype(float).values,
        "ACAGN-Gate": df["p_gate"].astype(float).values,
        "ACAGN-Hybrid": df["p_hybrid"].astype(float).values,
    }

    # Threshold-free metrics
    metric_rows = []
    for name, p in probs.items():
        m = compute_threshold_free_metrics(y, p)
        metric_rows.append({"model": name, **m})
    summary = pd.DataFrame(metric_rows).sort_values("model")
    summary.to_csv(os.path.join(paths.metrics_dir, "threshold_free_metrics.csv"), index=False)

    # ROC + PR diagrams
    plot_roc_pr(y, probs, paths.metrics_fig_dir)
    plot_metric_bars(summary, paths.metrics_fig_dir)

    # Operating points across strategies (per model)
    op_rows: List[Dict] = []
    for name, p in probs.items():
        op = operating_points(y, p)
        th = select_thresholds(op)
        for strat, t in th.items():
            r = op.loc[np.isclose(op["threshold"], t)].iloc[0].to_dict()
            op_rows.append({"model": name, "strategy": strat, **r})
    op_df = pd.DataFrame(op_rows)
    op_df.to_csv(os.path.join(paths.metrics_dir, "operating_points.csv"), index=False)
    plot_confusion_panels(op_df, paths.metrics_fig_dir, strategy="test_opt_mcc")
    plot_confusion_panels(op_df, paths.metrics_fig_dir, strategy="test_opt_recall80")

    # DeLong p-values (AUROC comparisons)
    pairs = [
        ("Base Ensemble", "ACAGN-Gate"),
        ("Base Ensemble", "ACAGN-Hybrid"),
        ("ACAGN-Gate", "ACAGN-Hybrid"),
    ]
    delong_rows = []
    for a, b in pairs:
        pval = delong_pvalue(y, probs[a], probs[b])
        delong_rows.append({"model_a": a, "model_b": b, "delong_pvalue": pval})
    pd.DataFrame(delong_rows).to_csv(os.path.join(paths.metrics_dir, "delong_auroc_pvalues.csv"), index=False)

    # CV AUROC (only available for base report)
    cv_out = {"cv_available": False}
    base_threshold = None
    if os.path.exists(paths.base_report_json):
        base_report = json.loads(open(paths.base_report_json, "r").read())
        base_threshold = base_report.get("best_threshold")
        cv_out = {
            "cv_available": True,
            "cv_auroc_mean": base_report.get("cv_auroc_mean"),
            "cv_auroc_std": base_report.get("cv_auroc_std"),
            "n_folds": base_report.get("n_folds"),
        }
    with open(os.path.join(paths.metrics_dir, "cross_validation_summary.json"), "w") as f:
        json.dump(cv_out, f, indent=2)

    # Shared-threshold clinical utility metrics (use the Base threshold, applied to all models)
    # This avoids test-optimizing thresholds and matches a realistic deployment workflow.
    if base_threshold is not None:
        shared_rows = []
        for name, p in probs.items():
            pred = (p >= float(base_threshold)).astype(int)
            c = confusion_counts(y, pred)
            shared_rows.append(
                {
                    "model": name,
                    "threshold_source": "base_best_threshold",
                    "threshold": float(base_threshold),
                    "accuracy": float((pred == y).mean()),
                    "precision": float(precision_score(y, pred, zero_division=0)),
                    "recall": float(recall_score(y, pred, zero_division=0)),
                    "specificity": float(specificity_score(y, pred)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                    "mcc": float(matthews_corrcoef(y, pred)),
                    "youden_j": float(youden_j(y, pred)),
                    **c,
                }
            )
        shared_df = pd.DataFrame(shared_rows)
        shared_df.to_csv(os.path.join(paths.metrics_dir, "shared_threshold_operating_metrics.csv"), index=False)

        # Diagram: threshold-based metrics at shared threshold (excluding TP/TN/FP/FN)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_cols = ["accuracy", "precision", "recall", "specificity", "f1", "mcc", "youden_j"]
        x = np.arange(len(shared_df))
        bar_w = 0.11
        for i, col in enumerate(plot_cols):
            ax.bar(x + (i - (len(plot_cols) - 1) / 2) * bar_w, shared_df[col], bar_w, label=col)
        ax.set_xticks(x)
        ax.set_xticklabels(shared_df["model"].tolist(), rotation=0)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Threshold-Based Operating Metrics @ Shared Threshold (t={float(base_threshold):.3f})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(paths.metrics_fig_dir, "shared_threshold_operating_metrics.png"), dpi=150)
        plt.close(fig)

    # Gate Mann–Whitney U summary diagram + CSV
    mwu_df, mwu_meta = summarize_gate_mwu(paths)
    with open(os.path.join(paths.metrics_dir, "gate_mannwhitney_summary.json"), "w") as f:
        json.dump(mwu_meta, f, indent=2)
    if not mwu_df.empty:
        mwu_df.to_csv(os.path.join(paths.metrics_dir, "gate_mannwhitney_all.csv"), index=False)
        plot_gate_mwu_top(mwu_df, paths.metrics_fig_dir, top_n=20)

    # A compact, categorized index of outputs for easy access.
    index = {
        "discrimination_metrics": {
            "tables": [
                "results/metrics/threshold_free_metrics.csv",
                "results/metrics/operating_points.csv",
                "results/metrics/shared_threshold_operating_metrics.csv",
            ],
            "figures": [
                "figures/metrics/roc_curves.png",
                "figures/metrics/pr_curves.png",
                "figures/metrics/discrimination_metrics.png",
            ],
        },
        "calibration_metrics": {
            "tables": ["results/metrics/threshold_free_metrics.csv"],
            "figures": ["figures/metrics/calibration_metrics.png"],
        },
        "clinical_utility_operating_points": {
            "tables": ["results/metrics/operating_points.csv"],
            "figures": [
                "figures/metrics/confusion_matrices_test_opt_mcc.png",
                "figures/metrics/confusion_matrices_test_opt_recall80.png",
                "figures/metrics/shared_threshold_operating_metrics.png",
            ],
        },
        "stability_and_significance": {
            "tables": ["results/metrics/delong_auroc_pvalues.csv"],
            "json": [
                "results/metrics/cross_validation_summary.json",
                "results/metrics/gate_mannwhitney_summary.json",
            ],
            "figures": ["figures/metrics/gate_mannwhitney_top.png"],
        },
        "data_quality": {"json": ["results/metrics/data_quality.json"]},
    }
    with open(os.path.join(paths.metrics_dir, "INDEX.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
