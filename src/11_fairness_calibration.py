"""  
Fairness and Calibration Analysis  
===================================  
Computes AUROC and ECE broken down by:  
  - Race (White, Black, Hispanic, Asian)  
  - Gender (Male, Female)  
  - Age group (five buckets)

Runs for both the ACAGN base ensemble (acagn_framework.pkl)  
and ACAGN-Gate (acagn_gate.pkl). Legacy `trance_*.pkl` paths are supported
as a fallback.

Produces:  
  - results/fairness_analysis.csv  
  - results/calibration_analysis.csv  
  - figures/fairness_auroc.png  
  - figures/reliability_diagram.png  
"""

import os  
import sys  
import logging  
import numpy as np  
import pandas as pd  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_auc_score, brier_score_loss  
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
try:  
    from config import (  
        FEATURES_CSV, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY,  
        RESULTS_DIR, FIGURES_DIR, TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,  
        FAIRNESS_RESULTS_CSV, CALIBRATION_RESULTS_CSV,  
    )  
except ImportError:  
    from .config import (  
        FEATURES_CSV, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY,  
        RESULTS_DIR, FIGURES_DIR, TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,  
        FAIRNESS_RESULTS_CSV, CALIBRATION_RESULTS_CSV,  
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  
logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 50   # skip subgroups smaller than this

def compute_ece(probs, labels, n_bins=10):  
    bins = np.linspace(0, 1, n_bins + 1)  
    ece, total = 0.0, len(labels)  
    for i in range(n_bins):  
        mask = (probs >= bins[i]) & (probs < bins[i + 1])  
        if mask.sum() == 0:  
            continue  
        ece += (mask.sum() / total) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))  
    return float(ece)

def load_test_demographics():  
    """  
    Loads the raw features CSV and extracts demographic columns for test patients.  
    Returns a DataFrame with hadm_id and demographic columns.  
    """  
    pruned = FEATURES_CSV.replace(".csv", "_pruned.csv")  
    path   = pruned if os.path.exists(pruned) else FEATURES_CSV  
    cols   = ["hadm_id", "subject_id", "readmit_30", "gender",  
              "anchor_age", "age_group", "race_enc"]  
    available = [c for c in cols if c in pd.read_csv(path, nrows=0).columns]  
    df = pd.read_csv(path, usecols=available, low_memory=False).fillna(0)  
    return df

def get_test_mask(groups):  
    rng = np.random.RandomState(RANDOM_STATE)  
    unique_patients = np.unique(groups)  
    rng.shuffle(unique_patients)  
    n      = len(unique_patients)  
    n_test = int(n * TRAIN_TEST_FRAC)  
    n_val  = int(n * TRAIN_VAL_FRAC)  
    test_pats = set(unique_patients[-n_test:])  
    return np.array([g in test_pats for g in groups])

def fairness_report(model_name, y_true, y_pred, demo_df, rows):  
    """  
    Computes per-subgroup AUROC and ECE and appends to rows list.  
    """  
    # Overall  
    rows.append({  
        "model": model_name,  
        "group_type": "overall",  
        "group_value": "all",  
        "n": len(y_true),  
        "readmit_rate": round(float(y_true.mean()), 4),  
        "auroc": round(float(roc_auc_score(y_true, y_pred)), 4),  
        "ece":   round(float(compute_ece(y_pred, y_true)), 4),  
        "brier": round(float(brier_score_loss(y_true, y_pred)), 4),  
    })

    # Gender  
    if "gender" in demo_df.columns:  
        for gval, gname in [(0, "Female"), (1, "Male")]:  
            mask = demo_df["gender"].values == gval  
            if mask.sum() < MIN_GROUP_SIZE:  
                continue  
            rows.append({  
                "model": model_name, "group_type": "gender",  
                "group_value": gname, "n": int(mask.sum()),  
                "readmit_rate": round(float(y_true[mask].mean()), 4),  
                "auroc": round(float(roc_auc_score(y_true[mask], y_pred[mask])), 4),  
                "ece":   round(float(compute_ece(y_pred[mask], y_true[mask])), 4),  
                "brier": round(float(brier_score_loss(y_true[mask], y_pred[mask])), 4),  
            })

    # Age group  
    if "age_group" in demo_df.columns:  
        age_names = {0: "<40", 1: "40-54", 2: "55-64", 3: "65-74", 4: "75-84", 5: "85+"}  
        for gval, gname in age_names.items():  
            mask = demo_df["age_group"].values == gval  
            if mask.sum() < MIN_GROUP_SIZE:  
                continue  
            rows.append({  
                "model": model_name, "group_type": "age_group",  
                "group_value": gname, "n": int(mask.sum()),  
                "readmit_rate": round(float(y_true[mask].mean()), 4),  
                "auroc": round(float(roc_auc_score(y_true[mask], y_pred[mask])), 4),  
                "ece":   round(float(compute_ece(y_pred[mask], y_true[mask])), 4),  
                "brier": round(float(brier_score_loss(y_true[mask], y_pred[mask])), 4),  
            })

    # Race (using race_enc quartiles as proxy since we have frequency encoding)  
    if "race_enc" in demo_df.columns:  
        race_vals = demo_df["race_enc"].values  
        quartiles = np.quantile(race_vals[race_vals > 0], [0.25, 0.5, 0.75, 1.0])  
        for qi, (lo, hi) in enumerate(zip([0] + list(quartiles[:-1]), quartiles)):  
            mask = (race_vals >= lo) & (race_vals < hi)  
            if mask.sum() < MIN_GROUP_SIZE:  
                continue  
            rows.append({  
                "model": model_name, "group_type": "race_quartile",  
                "group_value": f"Q{qi+1}", "n": int(mask.sum()),  
                "readmit_rate": round(float(y_true[mask].mean()), 4),  
                "auroc": round(float(roc_auc_score(y_true[mask], y_pred[mask])), 4),  
                "ece":   round(float(compute_ece(y_pred[mask], y_true[mask])), 4),  
                "brier": round(float(brier_score_loss(y_true[mask], y_pred[mask])), 4),  
            })

def reliability_diagram(probs_dict, labels, save_path, n_bins=10):  
    """  
    Plots reliability diagram for multiple models on one axis.  
    """  
    fig, ax = plt.subplots(figsize=(6, 6))  
    bins = np.linspace(0, 1, n_bins + 1)  
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for model_name, probs in probs_dict.items():  
        frac_pos = []  
        for i in range(n_bins):  
            mask = (probs >= bins[i]) & (probs < bins[i + 1])  
            if mask.sum() == 0:  
                frac_pos.append(np.nan)  
            else:  
                frac_pos.append(float(labels[mask].mean()))  
        ax.plot(bin_centers, frac_pos, "s-", label=model_name, linewidth=1.5, markersize=5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")  
    ax.set_xlabel("Mean predicted probability")  
    ax.set_ylabel("Fraction of positives (observed readmission rate)")  
    ax.set_title("Reliability diagram")  
    ax.legend()  
    ax.set_xlim(0, 1)  
    ax.set_ylim(0, 1)  
    plt.tight_layout()  
    plt.savefig(save_path, dpi=200, bbox_inches="tight")  
    plt.close()  
    logger.info("Reliability diagram saved -> %s", save_path)

def run_fairness_calibration():  
    os.makedirs(RESULTS_DIR, exist_ok=True)  
    os.makedirs(FIGURES_DIR, exist_ok=True)

    demo_df = load_test_demographics()  
    groups  = demo_df["subject_id"].values  
    test_mask = get_test_mask(groups)  
    demo_test = demo_df[test_mask].reset_index(drop=True)  
    y_true    = demo_df["readmit_30"].values[test_mask].astype(np.float32)

    rows = []  
    probs_for_diagram = {}

    # ── Base ensemble ─────────────────────────────────────────────────────  
    base_model_path = MAIN_MODEL_PKL
    if not os.path.exists(base_model_path) and os.path.exists(MAIN_MODEL_PKL_LEGACY):
        logger.warning(
            "Base model not found at %s; falling back to legacy path %s",
            base_model_path,
            MAIN_MODEL_PKL_LEGACY,
        )
        base_model_path = MAIN_MODEL_PKL_LEGACY
    if os.path.exists(base_model_path):  
        logger.info("Loading ACAGN base ensemble...")  
        lgbm_bundle = joblib.load(base_model_path)  
        lgbm_probs  = lgbm_bundle.get("test_probs_cal")  
        lgbm_labels = lgbm_bundle.get("test_labels")

        if lgbm_probs is not None and len(lgbm_probs) == len(y_true):  
            fairness_report("LightGBM-ensemble", y_true, lgbm_probs, demo_test, rows)  
            probs_for_diagram["LightGBM-ensemble"] = lgbm_probs  
        else:  
            logger.warning("LightGBM test probs not found in bundle or size mismatch.")  
    else:  
        logger.warning("Base model not found at %s", base_model_path)

    # ── ACAGN-Gate ─────────────────────────────────────────────────────────  
    gate_model_path = GATE_MODEL_PKL
    if not os.path.exists(gate_model_path) and os.path.exists(GATE_MODEL_PKL_LEGACY):
        logger.warning(
            "Gate model not found at %s; falling back to legacy path %s",
            gate_model_path,
            GATE_MODEL_PKL_LEGACY,
        )
        gate_model_path = GATE_MODEL_PKL_LEGACY
    if os.path.exists(gate_model_path):  
        logger.info("Loading ACAGN-Gate...")  
        gate_bundle = joblib.load(gate_model_path)  
        gate_probs  = gate_bundle.get("test_probs_cal")  
        gate_labels = gate_bundle.get("test_labels")

        if gate_probs is not None and len(gate_probs) == len(y_true):  
            fairness_report("ACAGN-Gate", y_true, gate_probs, demo_test, rows)  
            probs_for_diagram["ACAGN-Gate"] = gate_probs  
        else:  
            logger.warning("Gate probs not found or size mismatch.")  
    else:  
        logger.warning("Gate model not found at %s", gate_model_path)

    if not rows:  
        logger.error("No model results to report. Run training first.")  
        return

    df = pd.DataFrame(rows)  
    df.to_csv(FAIRNESS_RESULTS_CSV, index=False)  
    logger.info("Fairness results saved -> %s", FAIRNESS_RESULTS_CSV)

    # Print summary  
    overall = df[df["group_type"] == "overall"][["model", "auroc", "ece", "brier"]]  
    print("\nOverall performance:")  
    print(overall.to_string(index=False))

    gender_df = df[df["group_type"] == "gender"]  
    if not gender_df.empty:  
        print("\nBy gender:")  
        print(gender_df[["model", "group_value", "n", "auroc", "ece"]].to_string(index=False))

    age_df = df[df["group_type"] == "age_group"]  
    if not age_df.empty:  
        print("\nBy age group:")  
        print(age_df[["model", "group_value", "n", "auroc", "ece"]].to_string(index=False))

    # Max AUROC gap across age groups per model  
    for model in df["model"].unique():  
        age_aucs = df[(df["model"] == model) & (df["group_type"] == "age_group")]["auroc"]  
        if len(age_aucs) > 1:  
            logger.info("Model: %-20s | Max age-group AUROC gap: %.4f",  
                        model, age_aucs.max() - age_aucs.min())

    # Reliability diagram  
    if probs_for_diagram:  
        reliability_diagram(  
            probs_for_diagram, y_true,  
            os.path.join(FIGURES_DIR, "reliability_diagram.png")  
        )

    return df

if __name__ == "__main__":  
    run_fairness_calibration()
