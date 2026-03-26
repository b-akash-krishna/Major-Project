"""  
Temporal Drift Analysis  
========================  
Evaluates model performance across anchor_year_group cohorts  
in MIMIC-IV (2008-2022). No retraining — same trained model  
evaluated on each temporal slice of the test set.

Produces:  
  - results/temporal_drift_results.csv  
  - figures/temporal_drift.png  
"""

import os  
import sys  
import logging  
import numpy as np  
import pandas as pd  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_auc_score  
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
try:  
    from config import (  
        FEATURES_CSV, EMBEDDINGS_CSV, MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY,  
        RESULTS_DIR, FIGURES_DIR, TEMPORAL_DRIFT_CSV,  
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,  
    )  
except ImportError:  
    from .config import (  
        FEATURES_CSV, EMBEDDINGS_CSV, MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY,  
        RESULTS_DIR, FIGURES_DIR, TEMPORAL_DRIFT_CSV,  
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,  
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  
logger = logging.getLogger(__name__)

YEAR_GROUP_LABELS = {  
    0: "2008-2010", 1: "2011-2013", 2: "2014-2016",  
    3: "2017-2019", 4: "2020-2022",  
}

def get_test_mask(groups):  
    rng = np.random.RandomState(RANDOM_STATE)  
    unique = np.unique(groups)  
    rng.shuffle(unique)  
    n = len(unique)  
    n_test = int(n * TRAIN_TEST_FRAC)  
    n_val  = int(n * TRAIN_VAL_FRAC)  
    test_pats = set(unique[-n_test:])  
    return np.array([g in test_pats for g in groups])

def run_temporal_drift():  
    os.makedirs(RESULTS_DIR, exist_ok=True)  
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pruned = FEATURES_CSV.replace(".csv", "_pruned.csv")  
    feat_path = pruned if os.path.exists(pruned) else FEATURES_CSV  
    df = pd.read_csv(feat_path, low_memory=False).fillna(0)

    if "anchor_year_group" not in df.columns:  
        logger.error("anchor_year_group not found in features. Run 01_extract.py first.")  
        return

    groups    = df["subject_id"].values  
    test_mask = get_test_mask(groups)  
    df_test   = df[test_mask].reset_index(drop=True)  
    y_test    = df_test["readmit_30"].values

    rows = []

    # ── Base ensemble predictions ───────────────────────────────────────────  
    base_model_path = MAIN_MODEL_PKL
    if not os.path.exists(base_model_path) and os.path.exists(MAIN_MODEL_PKL_LEGACY):
        logger.warning(
            "Base model not found at %s; falling back to legacy path %s",
            base_model_path,
            MAIN_MODEL_PKL_LEGACY,
        )
        base_model_path = MAIN_MODEL_PKL_LEGACY
    if os.path.exists(base_model_path):  
        bundle     = joblib.load(base_model_path)  
        lgbm_probs = bundle.get("test_probs_cal")  
        if lgbm_probs is not None and len(lgbm_probs) == len(y_test):  
            for yg_code, yg_label in YEAR_GROUP_LABELS.items():  
                mask = df_test["anchor_year_group"].values == yg_code  
                if mask.sum() < 50:  
                    continue  
                auroc = roc_auc_score(y_test[mask], lgbm_probs[mask])  
                rows.append({  
                    "model": "LightGBM-ensemble",  
                    "year_group": yg_label,  
                    "year_group_code": yg_code,  
                    "n_admissions": int(mask.sum()),  
                    "readmit_rate": round(float(y_test[mask].mean()), 4),  
                    "auroc": round(float(auroc), 4),  
                })  
                logger.info("LightGBM | %s | AUROC: %.4f (n=%d)",  
                            yg_label, auroc, mask.sum())

    # ── ACAGN-Gate predictions ─────────────────────────────────────────────  
    gate_model_path = GATE_MODEL_PKL
    if not os.path.exists(gate_model_path) and os.path.exists(GATE_MODEL_PKL_LEGACY):
        logger.warning(
            "Gate model not found at %s; falling back to legacy path %s",
            gate_model_path,
            GATE_MODEL_PKL_LEGACY,
        )
        gate_model_path = GATE_MODEL_PKL_LEGACY
    if os.path.exists(gate_model_path):  
        bundle     = joblib.load(gate_model_path)  
        gate_probs = bundle.get("test_probs_cal")  
        if gate_probs is not None and len(gate_probs) == len(y_test):  
            for yg_code, yg_label in YEAR_GROUP_LABELS.items():  
                mask = df_test["anchor_year_group"].values == yg_code  
                if mask.sum() < 50:  
                    continue  
                auroc = roc_auc_score(y_test[mask], gate_probs[mask])  
                rows.append({  
                    "model": "ACAGN-Gate",  
                    "year_group": yg_label,  
                    "year_group_code": yg_code,  
                    "n_admissions": int(mask.sum()),  
                    "readmit_rate": round(float(y_test[mask].mean()), 4),  
                    "auroc": round(float(auroc), 4),  
                })  
                logger.info("ACAGN-Gate | %s | AUROC: %.4f (n=%d)",  
                            yg_label, auroc, mask.sum())

    if not rows:  
        logger.error("No results generated. Ensure models are trained.")  
        return

    df_results = pd.DataFrame(rows)  
    df_results.to_csv(TEMPORAL_DRIFT_CSV, index=False)  
    logger.info("Temporal drift results saved -> %s", TEMPORAL_DRIFT_CSV)

    # Plot  
    fig, ax = plt.subplots(figsize=(9, 5))  
    for model_name, grp in df_results.groupby("model"):  
        grp = grp.sort_values("year_group_code")  
        ax.plot(grp["year_group"], grp["auroc"], "o-",  
                linewidth=2, markersize=7, label=model_name)

    ax.set_xlabel("Year group")  
    ax.set_ylabel("AUROC")  
    ax.set_title("Model performance across time periods (temporal drift)")  
    ax.legend()  
    ax.set_ylim(0.5, 1.0)  
    ax.grid(True, alpha=0.3)  
    plt.xticks(rotation=20)  
    plt.tight_layout()  
    path = os.path.join(FIGURES_DIR, "temporal_drift.png")  
    plt.savefig(path, dpi=200, bbox_inches="tight")  
    plt.close()  
    logger.info("Temporal drift plot saved -> %s", path)

    print(df_results.to_string(index=False))  
    return df_results

if __name__ == "__main__":  
    run_temporal_drift()
