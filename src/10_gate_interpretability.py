"""  
Gate Interpretability Analysis  
===============================  
Groups test patients by clinical keyword presence in their notes,  
then compares average gate weights on related EHR features between  
the keyword-present and keyword-absent groups.

Produces:  
  - results/gate_interpretability.csv  (per-condition per-feature stats)  
  - figures/gate_heatmap.png           (condition x feature group heatmap)  
"""

import os  
import sys  
import json  
import logging  
import numpy as np  
import pandas as pd  
from scipy import stats  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
try:  
    from config import (  
        GATE_WEIGHTS_NPY, GATE_PATIENT_IDS_NPY, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY,  
        MIMIC_NOTE_DIR, MIMIC_BHC_DIR, RESULTS_DIR, FIGURES_DIR,  
    )  
except ImportError:  
    from .config import (  
        GATE_WEIGHTS_NPY, GATE_PATIENT_IDS_NPY, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY,  
        MIMIC_NOTE_DIR, MIMIC_BHC_DIR, RESULTS_DIR, FIGURES_DIR,  
    )  
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  
logger = logging.getLogger(__name__)

# ── Clinical condition definitions ────────────────────────────────────────────  
# Each entry: condition name -> {keywords in notes, related tabular features}  
# Keywords are matched case-insensitively as substrings in the discharge text.  
# Feature names must match your tab_cols list exactly.

CONDITION_FEATURE_MAP = {  
    "chronic_anemia": {  
        "keywords": ["chronic anemia", "known anemia", "baseline anemia", "anemia of chronic"],  
        "features": ["lab_hemoglobin_min", "lab_hemoglobin_range", "anemia", "lab_hematocrit_min"],  
    },  
    "chronic_kidney_disease": {  
        "keywords": ["chronic kidney disease", "ckd", "chronic renal", "end stage renal", "esrd"],  
        "features": ["lab_creatinine_mean", "lab_creatinine_max", "lab_bun_mean", "lab_bun_max",  
                     "lab_creatinine_range", "lab_bun_range", "cm_renal_fail"],  
    },  
    "heart_failure": {  
        "keywords": ["heart failure", "chf", "congestive heart", "systolic dysfunction",  
                     "diastolic dysfunction", "reduced ejection fraction"],  
        "features": ["cm_chf", "icu_los_sum", "icu_count", "lab_sodium_range",  
                     "lab_bicarb_range", "had_icu"],  
    },  
    "copd": {  
        "keywords": ["copd", "chronic obstructive", "emphysema", "chronic bronchitis"],  
        "features": ["lab_pao2_range", "lab_paco2_range", "lab_pao2_mean",  
                     "lab_ph_range", "cm_copd"],  
    },  
    "diabetes": {  
        "keywords": ["diabetes mellitus", "diabetic", "type 2 diabetes", "type 1 diabetes",  
                     "insulin dependent"],  
        "features": ["lab_glucose_mean", "lab_glucose_max", "lab_glucose_range",  
                     "lab_glucose_last", "cm_diabetes"],  
    },  
    "hypertension": {  
        "keywords": ["hypertension", "hypertensive", "high blood pressure"],  
        "features": ["v_sbp_mean", "v_sbp_std", "cm_hypertension"],  
    },  
    "liver_disease": {  
        "keywords": ["cirrhosis", "hepatic", "liver disease", "liver failure",  
                     "portal hypertension"],  
        "features": ["cm_liver", "lab_bicarb_max", "lab_platelets_min",  
                     "lab_platelets_range", "thrombocytopenia"],  
    },  
    "cancer": {  
        "keywords": ["malignancy", "carcinoma", "cancer", "metastatic", "oncology",  
                     "chemotherapy", "radiation therapy"],  
        "features": ["cm_cancer", "lab_wbc_range", "lab_platelets_range",  
                     "lab_hemoglobin_min", "high_risk_org"],  
    },  
}

def load_discharge_notes(hadm_ids: set) -> dict:  
    """  
    Loads discharge note text for given hadm_ids.  
    Returns dict mapping hadm_id -> lowercased note text.  
    """  
    note_text = {}  
    for base in [MIMIC_NOTE_DIR, MIMIC_BHC_DIR]:  
        if not os.path.isdir(base):  
            continue  
        for fn in ["discharge.csv.gz", "discharge.csv"]:  
            path = os.path.join(base, fn)  
            if not os.path.exists(path):  
                # walk subdirectories  
                for root, _, files in os.walk(base):  
                    for f in files:  
                        if f == fn:  
                            path = os.path.join(root, f)  
                            break  
            if not os.path.exists(path):  
                continue  
            logger.info("Loading notes from %s", path)  
            try:  
                df = pd.read_csv(path, usecols=["hadm_id", "text"],  
                                 low_memory=True, nrows=2_000_000)  
                df = df[df["hadm_id"].isin(hadm_ids)].dropna(subset=["text"])  
                for row in df.itertuples():  
                    hadm = int(row.hadm_id)  
                    if hadm not in note_text:  
                        note_text[hadm] = str(row.text).lower()  
                    else:  
                        note_text[hadm] += " " + str(row.text).lower()  
                logger.info("  Loaded notes for %d admissions", len(note_text))  
            except Exception as e:  
                logger.warning("  Failed: %s", e)  
            break  
    return note_text

def keyword_present(text: str, keywords: list) -> bool:  
    return any(kw in text for kw in keywords)

def run_interpretability_analysis():  
    os.makedirs(RESULTS_DIR, exist_ok=True)  
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load gate weights and patient ids  
    gate_weights  = np.load(GATE_WEIGHTS_NPY)    # (n_test, n_features)  
    test_hadm_ids = np.load(GATE_PATIENT_IDS_NPY) # (n_test,)  
    model_path = GATE_MODEL_PKL
    if not os.path.exists(model_path) and os.path.exists(GATE_MODEL_PKL_LEGACY):
        logger.warning(
            "Gate model not found at %s; falling back to legacy path %s",
            model_path,
            GATE_MODEL_PKL_LEGACY,
        )
        model_path = GATE_MODEL_PKL_LEGACY
    bundle        = joblib.load(model_path)  
    tab_cols      = bundle["tab_cols"]            # list of feature names

    logger.info("Gate weights shape: %s", gate_weights.shape)  
    logger.info("Features: %d", len(tab_cols))

    # Build feature index lookup  
    feat_idx = {name: i for i, name in enumerate(tab_cols)}

    # Load discharge notes for test patients  
    note_text = load_discharge_notes(set(test_hadm_ids.tolist()))  
    logger.info("Notes loaded for %d / %d test patients",  
                len(note_text), len(test_hadm_ids))

    # Run analysis for each condition  
    rows = []  
    for condition, spec in CONDITION_FEATURE_MAP.items():  
        keywords = spec["keywords"]  
        features = [f for f in spec["features"] if f in feat_idx]

        if not features:  
            logger.warning("No matching features for condition: %s", condition)  
            continue

        # Boolean mask: which test patients have this condition mentioned in notes  
        has_condition = np.array([  
            keyword_present(note_text.get(int(hid), ""), keywords)  
            for hid in test_hadm_ids  
        ])

        n_mentioned     = has_condition.sum()  
        n_not_mentioned = (~has_condition).sum()

        if n_mentioned < 30 or n_not_mentioned < 30:  
            logger.warning("Too few patients for %s (mentioned=%d, not=%d)",  
                           condition, n_mentioned, n_not_mentioned)  
            continue

        logger.info("Condition: %-30s | mentioned: %d | not: %d",  
                    condition, n_mentioned, n_not_mentioned)

        for feat_name in features:  
            fi = feat_idx[feat_name]  
            weights_mentioned     = gate_weights[has_condition,  fi]  
            weights_not_mentioned = gate_weights[~has_condition, fi]

            mean_mentioned     = float(np.mean(weights_mentioned))  
            mean_not_mentioned = float(np.mean(weights_not_mentioned))  
            mean_diff          = mean_mentioned - mean_not_mentioned

            # Mann-Whitney U test (non-parametric, appropriate for gate weights)  
            stat, pval = stats.mannwhitneyu(  
                weights_mentioned, weights_not_mentioned, alternative="two-sided"  
            )

            rows.append({  
                "condition":        condition,  
                "feature":          feat_name,  
                "mean_mentioned":   round(mean_mentioned,     4),  
                "mean_not_mentioned": round(mean_not_mentioned, 4),  
                "mean_difference":  round(mean_diff,          4),  
                "n_mentioned":      int(n_mentioned),  
                "n_not_mentioned":  int(n_not_mentioned),  
                "mann_whitney_u":   round(float(stat),        2),  
                "p_value":          float(pval),  
                "significant_p05":  bool(pval < 0.05),  
            })

    df = pd.DataFrame(rows)

    # Bonferroni correction  
    n_tests = len(df)  
    df["p_bonferroni"]   = (df["p_value"] * n_tests).clip(upper=1.0)  
    df["significant_bonferroni"] = df["p_bonferroni"] < 0.05

    results_path = os.path.join(RESULTS_DIR, "gate_interpretability.csv")  
    df.to_csv(results_path, index=False)  
    logger.info("Interpretability results saved -> %s", results_path)

    # ── Heatmap ───────────────────────────────────────────────────────────────  
    # Pivot: conditions as rows, features as columns, mean_difference as values  
    pivot = df.pivot_table(  
        index="condition", columns="feature",  
        values="mean_difference", aggfunc="mean"  
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2),  
                                    max(6, len(pivot.index) * 0.8)))  
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",  
                   vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(len(pivot.columns)))  
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)  
    ax.set_yticks(range(len(pivot.index)))  
    ax.set_yticklabels([c.replace("_", " ") for c in pivot.index], fontsize=9)

    plt.colorbar(im, ax=ax, label="Mean gate weight difference\n(mentioned − not mentioned)")  
    ax.set_title("Gate weight suppression/amplification by clinical condition\n"  
                 "Blue = suppressed when condition mentioned | Red = amplified")

    # Add significance markers  
    for i, cond in enumerate(pivot.index):  
        for j, feat in enumerate(pivot.columns):  
            subset = df[(df["condition"] == cond) & (df["feature"] == feat)]  
            if len(subset) > 0 and subset["significant_bonferroni"].values[0]:  
                ax.text(j, i, "*", ha="center", va="center", fontsize=12, color="black")

    plt.tight_layout()  
    heatmap_path = os.path.join(FIGURES_DIR, "gate_heatmap.png")  
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")  
    plt.close()  
    logger.info("Heatmap saved -> %s", heatmap_path)

    # Summary to console  
    logger.info("\nTop suppressed feature-condition pairs (negative = suppressed):")  
    top_suppressed = df.nsmallest(10, "mean_difference")[  
        ["condition", "feature", "mean_difference", "p_bonferroni", "significant_bonferroni"]  
    ]  
    print(top_suppressed.to_string(index=False))

    return df

if __name__ == "__main__":  
    run_interpretability_analysis()
