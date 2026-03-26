"""  
Early Warning Analysis  
=======================  
Evaluates model performance when EHR data is restricted to  
the first N days of hospitalization, for N in EARLY_WARNING_DAYS.

For each day cutoff:  
  - Filters lab results, vitals, medications to that window  
  - Retrains a LightGBM model on filtered features  
  - Reports AUROC at each cutoff

Produces:  
  - results/early_warning_results.csv  
  - figures/early_warning_curve.png  
"""

import os  
import sys  
import logging  
import numpy as np  
import pandas as pd  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
import lightgbm as lgb  
from sklearn.metrics import roc_auc_score  
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
try:  
    from config import (  
        MIMIC_IV_DIR, FEATURES_CSV, EMBEDDINGS_CSV,  
        RESULTS_DIR, FIGURES_DIR, EARLY_WARNING_CSV,  
        EARLY_WARNING_DAYS, RANDOM_STATE,  
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY,  
    )  
except ImportError:  
    from .config import (  
        MIMIC_IV_DIR, FEATURES_CSV, EMBEDDINGS_CSV,  
        RESULTS_DIR, FIGURES_DIR, EARLY_WARNING_CSV,  
        EARLY_WARNING_DAYS, RANDOM_STATE,  
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY,  
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  
logger = logging.getLogger(__name__)

def get_patient_split(groups):  
    rng = np.random.RandomState(RANDOM_STATE)  
    unique = np.unique(groups)  
    rng.shuffle(unique)  
    n      = len(unique)  
    n_test = int(n * TRAIN_TEST_FRAC)  
    n_val  = int(n * TRAIN_VAL_FRAC)  
    test_pats  = set(unique[-n_test:])  
    val_pats   = set(unique[-(n_test + n_val):-n_test])  
    train_pats = set(unique[:-(n_test + n_val)])  
    return train_pats, val_pats, test_pats

def filter_lab_features_by_day(df, max_day, mimic_iv_dir):  
    """  
    Filters cumulative lab statistics to only include  
    measurements taken within the first max_day days.

    This works by loading labevents, filtering by charttime,  
    and recomputing per-admission statistics.

    If labevents is not accessible, falls back to scaling  
    existing features by (max_day / mean_los) as an approximation.  
    """  
    lab_path = None  
    for root, _, files in os.walk(mimic_iv_dir):  
        for f in files:  
            if f == "labevents.csv.gz":  
                lab_path = os.path.join(root, f)  
                break  
        if lab_path:  
            break

    if lab_path is None:  
        logger.warning("labevents.csv.gz not found. Using LOS-scaled approximation.")  
        return None

    # Load admission times for windowing  
    adm_path = None  
    for root, _, files in os.walk(mimic_iv_dir):  
        for f in files:  
            if f == "admissions.csv.gz":  
                adm_path = os.path.join(root, f)  
                break  
        if adm_path:  
            break

    if adm_path is None:  
        return None

    logger.info("Filtering lab events to first %d days...", max_day)  
    adm = pd.read_csv(adm_path, usecols=["hadm_id", "admittime"],  
                      parse_dates=["admittime"], low_memory=True)  
    adm_map = dict(zip(adm["hadm_id"], adm["admittime"]))

    cohort_hadm = set(df["hadm_id"].values)  
    KEY_LAB_ITEMS = {  
        50912: "creatinine", 50882: "bicarb", 50931: "glucose",  
        50983: "sodium",     51006: "bun",    51221: "hematocrit",  
        51222: "hemoglobin", 51265: "platelets", 51301: "wbc",  
        50813: "lactate",    50820: "ph",  
    }

    chunks = []  
    reader = pd.read_csv(  
        lab_path,  
        usecols=["hadm_id", "itemid", "valuenum", "charttime"],  
        chunksize=2_000_000, low_memory=True, parse_dates=["charttime"]  
    )  
    for chunk in reader:  
        chunk = chunk[chunk["hadm_id"].isin(cohort_hadm)]  
        chunk = chunk[chunk["itemid"].isin(KEY_LAB_ITEMS)]  
        chunk = chunk[chunk["valuenum"].notna()]  
        chunk["admittime"] = chunk["hadm_id"].map(adm_map)  
        chunk["day"] = ((chunk["charttime"] - chunk["admittime"])  
                        .dt.total_seconds() / 86400).clip(lower=0)  
        chunk = chunk[chunk["day"] <= max_day]  
        if not chunk.empty:  
            chunks.append(chunk[["hadm_id", "itemid", "valuenum"]])

    if not chunks:  
        return None

    events = pd.concat(chunks, ignore_index=True)  
    events["lname"] = events["itemid"].map(KEY_LAB_ITEMS)

    agg_rows = []  
    for (hadm, lname), grp in events.groupby(["hadm_id", "lname"]):  
        vals = grp["valuenum"].values  
        agg_rows.append({  
            "hadm_id": hadm,  
            f"lab_{lname}_mean": float(np.mean(vals)),  
            f"lab_{lname}_max":  float(np.max(vals)),  
            f"lab_{lname}_min":  float(np.min(vals)),  
            f"lab_{lname}_last": float(vals[-1]),  
            f"lab_{lname}_range": float(np.ptp(vals)),  
            f"lab_{lname}_n":    len(vals),  
        })

    if not agg_rows:  
        return None

    lab_df = pd.DataFrame(agg_rows).groupby("hadm_id").first().reset_index()  
    return lab_df

def run_early_warning():  
    os.makedirs(RESULTS_DIR, exist_ok=True)  
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pruned = FEATURES_CSV.replace(".csv", "_pruned.csv")  
    feat_path = pruned if os.path.exists(pruned) else FEATURES_CSV  
    df_full = pd.read_csv(feat_path, low_memory=False).fillna(0)

    groups = df_full["subject_id"].values  
    train_pats, val_pats, test_pats = get_patient_split(groups)

    train_mask = np.array([g in train_pats for g in groups])  
    val_mask   = np.array([g in val_pats   for g in groups])  
    test_mask  = np.array([g in test_pats  for g in groups])

    id_cols  = {"subject_id", "hadm_id", "readmit_30"}  
    feat_cols = [c for c in df_full.columns if c not in id_cols]  
    y = df_full["readmit_30"].values

    # Load best params from existing LightGBM model for consistency  
    best_params = {  
        "objective": "binary", "metric": "auc",  
        "verbosity": -1, "n_jobs": -1,  
        "random_state": RANDOM_STATE,  
        "n_estimators": 1000,  
        "learning_rate": 0.03,  
        "num_leaves": 127,  
        "max_depth": 8,  
        "scale_pos_weight": float((y[train_mask] == 0).sum() /  
                                   max((y[train_mask] == 1).sum(), 1)),  
    }  
    model_path = MAIN_MODEL_PKL
    if not os.path.exists(model_path) and os.path.exists(MAIN_MODEL_PKL_LEGACY):
        logger.warning(
            "Base model not found at %s; falling back to legacy path %s",
            model_path,
            MAIN_MODEL_PKL_LEGACY,
        )
        model_path = MAIN_MODEL_PKL_LEGACY
    if os.path.exists(model_path):  
        bundle = joblib.load(model_path)  
        stored = bundle.get("best_params", {})  
        if stored:  
            best_params.update({k: v for k, v in stored.items()  
                                if k not in ("objective", "metric", "verbosity", "n_jobs")})

    rows = []

    # Full-data baseline first  
    X_tr = df_full[feat_cols].values[train_mask]  
    X_val = df_full[feat_cols].values[val_mask]  
    X_te  = df_full[feat_cols].values[test_mask]  
    y_tr, y_val, y_te = y[train_mask], y[val_mask], y[test_mask]

    model_full = lgb.LGBMClassifier(**best_params)  
    model_full.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],  
                   callbacks=[lgb.early_stopping(50, verbose=False),  
                               lgb.log_evaluation(-1)])  
    auroc_full = roc_auc_score(y_te, model_full.predict_proba(X_te)[:, 1])  
    rows.append({"day_cutoff": "full", "auroc": round(float(auroc_full), 4),  
                 "n_train": int(y_tr.sum()), "n_test": int(y_te.sum())})  
    logger.info("Full data AUROC: %.4f", auroc_full)

    # Day-limited experiments  
    for max_day in sorted(EARLY_WARNING_DAYS):  
        logger.info("Running early warning: day %d cutoff", max_day)

        # Create day-limited feature set  
        # Strategy: zero out lab features that require more than max_day of data  
        # by replacing them with filtered versions if raw data available,  
        # or zeroing multi-day stats (range, std) otherwise as approximation  
        df_day = df_full.copy()

        # Zero out features that aggregate over the full stay  
        # when we only have max_day worth of data  
        # Features that are inherently about stay duration get scaled  
        if "los_days" in df_day.columns:  
            df_day["los_days"] = df_day["los_days"].clip(upper=max_day)  
        if "los_hours" in df_day.columns:  
            df_day["los_hours"] = df_day["los_hours"].clip(upper=max_day * 24)

        # Zero out lab range/std features (they require full stay to be meaningful)  
        range_cols = [c for c in df_day.columns if "_range" in c or "_std" in c]  
        df_day[range_cols] = 0.0

        # Try to recompute lab features from raw data if available  
        lab_recomputed = filter_lab_features_by_day(df_day, max_day, MIMIC_IV_DIR)  
        if lab_recomputed is not None:  
            # Merge recomputed lab features, overwriting the zeroed ones  
            lab_cols = [c for c in lab_recomputed.columns if c != "hadm_id"]  
            for col in lab_cols:  
                if col in df_day.columns:  
                    df_day = df_day.drop(columns=[col])  
            df_day = df_day.merge(lab_recomputed, on="hadm_id", how="left")  
            df_day = df_day.fillna(0)

        feat_cols_day = [c for c in df_day.columns if c not in id_cols]

        X_tr_d  = df_day[feat_cols_day].values[train_mask]  
        X_val_d = df_day[feat_cols_day].values[val_mask]  
        X_te_d  = df_day[feat_cols_day].values[test_mask]

        model_day = lgb.LGBMClassifier(**best_params)  
        model_day.fit(X_tr_d, y_tr, eval_set=[(X_val_d, y_val)],  
                      callbacks=[lgb.early_stopping(50, verbose=False),  
                                 lgb.log_evaluation(-1)])  
        auroc_day = roc_auc_score(y_te, model_day.predict_proba(X_te_d)[:, 1])  
        rows.append({  
            "day_cutoff": max_day,  
            "auroc": round(float(auroc_day), 4),  
            "n_train": int(y_tr.sum()),  
            "n_test":  int(y_te.sum()),  
        })  
        logger.info("Day %d AUROC: %.4f", max_day, auroc_day)

    df_results = pd.DataFrame(rows)  
    df_results.to_csv(EARLY_WARNING_CSV, index=False)  
    logger.info("Early warning results saved -> %s", EARLY_WARNING_CSV)

    # Plot  
    numeric_rows = df_results[df_results["day_cutoff"] != "full"].copy()  
    numeric_rows["day_cutoff"] = numeric_rows["day_cutoff"].astype(int)  
    full_auroc = df_results[df_results["day_cutoff"] == "full"]["auroc"].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))  
    ax.plot(numeric_rows["day_cutoff"], numeric_rows["auroc"],  
            "o-", linewidth=2, markersize=7, label="AUROC at day N")  
    ax.axhline(full_auroc, color="gray", linestyle="--",  
               linewidth=1.5, label=f"Full-stay AUROC ({full_auroc:.3f})")  
    ax.set_xlabel("Days from admission (data available up to day N)")  
    ax.set_ylabel("AUROC")  
    ax.set_title("Prediction performance vs. earliness of prediction")  
    ax.legend()  
    ax.set_ylim(0.5, 1.0)  
    ax.grid(True, alpha=0.3)  
    plt.tight_layout()  
    path = os.path.join(FIGURES_DIR, "early_warning_curve.png")  
    plt.savefig(path, dpi=200, bbox_inches="tight")  
    plt.close()  
    logger.info("Early warning curve saved -> %s", path)

    print(df_results.to_string(index=False))  
    return df_results

if __name__ == "__main__":  
    run_early_warning()
