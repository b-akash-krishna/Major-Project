# colab_train.py  ──  TRANCE Baseline: XGBoost GPU + LightGBM CPU Ensemble
# =========================================================================
# Designed for Colab Free Tier T4 GPU (15.6 GB VRAM, ~12 GB RAM usable)
#
# Architecture:
#   Primary  : XGBoost  (tree_method="hist", device="cuda")
#              → ships CUDA-ready on every Colab instance, no recompile
#   Diversity: LightGBM (CPU, always works on Colab)
#   Blend    : 70% XGB + 30% LGB per fold
#   Post     : Isotonic calibration on OOF + MCC threshold search
#
# What this script produces for the paper:
#   • AUROC / AUPRC / MCC / F1 / Brier / LogLoss on held-out test set
#   • OOF predictions → used for final ensembling with colab_gated_train.py
#   • Calibrated probabilities saved to results/trance_baseline_preds.csv
#   • Feature importance CSV + SHAP summary PNG
#   • ROC and PR curve PNGs
# =========================================================================

import gc
import json
import logging
import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
DRIVE_PROJECT_PATH = "WeCare Model/data"

DATA_DIR = "/content"
if os.path.exists("/content/drive/MyDrive"):
    DATA_DIR = f"/content/drive/MyDrive/{DRIVE_PROJECT_PATH}"
    print(f"✓ Google Drive detected → {DATA_DIR}")

MODELS_DIR  = "/content/models"
RESULTS_DIR = "/content/results"
FIGURES_DIR = "/content/figures"

FEATURES_CSV   = "/content/ultimate_features_pruned.csv"
EMBEDDINGS_CSV = "/content/embeddings.csv"

# Fallback to Drive if local copy wasn't made
if not os.path.exists(FEATURES_CSV):
    FEATURES_CSV = os.path.join(DATA_DIR, "ultimate_features_pruned.csv")
if not os.path.exists(EMBEDDINGS_CSV):
    EMBEDDINGS_CSV = os.path.join(DATA_DIR, "embeddings.csv")

RANDOM_STATE             = 42
TRAIN_OPTUNA_TRIALS      = 60       # GPU trials converge fast; 60 is sufficient
TRAIN_N_FOLDS            = 5
TRAIN_SEEDS              = [42, 2024, 777]
TRAIN_CT5_KEEP_DIMS      = 768      # full 768-dim fine-tuned signal
TRAIN_THRESHOLD_STRATEGY = "mcc"
TRAIN_ENABLE_STACK       = True
XGB_BLEND_WEIGHT         = 0.7     # XGB share in blend

# ── GPU ────────────────────────────────────────────────────────
import torch
HAS_GPU    = torch.cuda.is_available()
XGB_DEVICE = "cuda" if HAS_GPU else "cpu"
LGB_DEVICE = "cpu"                  # Colab LGB wheel has no CUDA

if HAS_GPU:
    _p = torch.cuda.get_device_properties(0)
    print(f"✓ GPU  : {_p.name}  |  VRAM: {_p.total_memory/1e9:.1f} GB")
    print(f"✓ XGBoost → cuda  |  LightGBM → cpu")
else:
    print("✓ No GPU — XGBoost CPU mode")

# ──────────────────────────────────────────────────────────────
# DIRECTORIES & LOGGING
# ──────────────────────────────────────────────────────────────
for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_DIR, "train_baseline.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def _set_seed(seed: int) -> None:
    np.random.seed(int(seed)); random.seed(int(seed))

def _flush_gpu() -> None:
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()

def _optuna_cb(study, trial) -> None:
    if trial.value is not None:
        logger.info("  Trial %3d | AUC %.4f | best %.4f | depth %s | lr %.5f",
                    trial.number, trial.value, study.best_value,
                    trial.params.get("max_depth", "?"),
                    trial.params.get("learning_rate", 0.0))

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    total = len(labels)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.sum() == 0: continue
        ece += (m.sum() / total) * abs(float(labels[m].mean()) - float(probs[m].mean()))
    return float(ece)

# ──────────────────────────────────────────────────────────────
# DATA LOADING  (chunked, float32, memory-safe)
# ──────────────────────────────────────────────────────────────
def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Loading tabular features ...")
    id_dtypes = {"subject_id": "int32", "hadm_id": "int32", "readmit_30": "int8"}
    tab = pd.read_csv(FEATURES_CSV, low_memory=False, dtype=id_dtypes).fillna(0)
    tab[tab.select_dtypes("float64").columns] = tab.select_dtypes("float64").astype("float32")

    if os.path.exists(EMBEDDINGS_CSV):
        logger.info("Loading embeddings (chunked) ...")
        emb_header = pd.read_csv(EMBEDDINGS_CSV, nrows=0)
        emb_dtypes = {c: "float32" for c in emb_header.columns if c.startswith("ct5_")}
        emb_dtypes["hadm_id"] = "int32"
        target_ids = set(tab["hadm_id"])
        chunks = []
        for chunk in pd.read_csv(EMBEDDINGS_CSV, chunksize=25_000,
                                  low_memory=False, dtype=emb_dtypes):
            chunk = chunk[chunk["hadm_id"].isin(target_ids)]
            if not chunk.empty: chunks.append(chunk)
            del chunk; gc.collect()
        emb = pd.concat(chunks, axis=0, ignore_index=True)
        df  = tab.merge(emb, on="hadm_id", how="left").fillna(0)
        del tab, emb, chunks; gc.collect()
    else:
        df = tab; del tab; gc.collect()

    groups = df["subject_id"].astype(int)
    y      = df["readmit_30"].astype("int8")

    # High-variance ct5 dim selection (full 768 if available)
    ct5_cols = [c for c in df.columns if c.startswith("ct5_") and c[4:].isdigit()]
    if len(ct5_cols) > TRAIN_CT5_KEEP_DIMS:
        keep = (df[ct5_cols].var().sort_values(ascending=False)
                .head(TRAIN_CT5_KEEP_DIMS).index.tolist())
        fixed = [c for c in df.columns if not (c.startswith("ct5_") and c[4:].isdigit())]
        df = df[fixed + keep]
        logger.info("Kept %d ct5 dims.", len(keep))

    X = df.drop(columns={"subject_id", "hadm_id", "readmit_30"})
    logger.info("Feature matrix: %s", X.shape)
    return X, y, groups

# ──────────────────────────────────────────────────────────────
# HPO — XGBoost on GPU with pruning
# ──────────────────────────────────────────────────────────────
def optimize_xgb(X_tr, y_tr, X_val, y_val, pos_weight, n_trials=60) -> Dict:
    logger.info("Optuna HPO | %d trials | XGBoost on %s | pos_weight=%.2f",
                n_trials, XGB_DEVICE, pos_weight)
    dtrain = xgb.DMatrix(X_tr,  label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    def objective(trial):
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "tree_method":      "hist",
            "device":           XGB_DEVICE,
            "verbosity":        0,
            "scale_pos_weight": pos_weight,
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 4.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 6.0),
            "seed":             RANDOM_STATE,
        }
        n_rounds = trial.suggest_int("n_estimators", 300, 2000)
        evals_result: Dict = {}
        bst = xgb.train(params, dtrain, num_boost_round=n_rounds,
                        evals=[(dval, "val")], early_stopping_rounds=30,
                        evals_result=evals_result, verbose_eval=False)
        val_aucs = evals_result["val"]["auc"]
        for step, v in enumerate(val_aucs[::20]):
            trial.report(v, step)
            if trial.should_prune():
                del bst; _flush_gpu(); raise optuna.exceptions.TrialPruned()
        auc = max(val_aucs)
        del bst; _flush_gpu()
        return float(auc)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE, multivariate=True),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_optuna_cb])
    best = study.best_params.copy()
    best.update({"objective": "binary:logistic", "eval_metric": "auc",
                 "tree_method": "hist", "device": XGB_DEVICE,
                 "verbosity": 0, "scale_pos_weight": pos_weight, "seed": RANDOM_STATE})
    logger.info("Best HPO AUC: %.4f", study.best_value)
    return best

# ──────────────────────────────────────────────────────────────
# THRESHOLD SEARCH
# ──────────────────────────────────────────────────────────────
def find_best_threshold(probs, y_true, strategy="mcc") -> float:
    best_score, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.70, 200):
        preds = (probs >= t).astype(int)
        score = (matthews_corrcoef(y_true, preds) if strategy == "mcc"
                 else f1_score(y_true, preds, zero_division=0))
        if score > best_score:
            best_score, best_t = score, t
    logger.info("Best threshold (%s): %.3f  score=%.4f", strategy, best_t, best_score)
    return best_t

# ──────────────────────────────────────────────────────────────
# OOF CROSS-VALIDATION  (XGB GPU + LGB CPU blend, seed ensemble)
# ──────────────────────────────────────────────────────────────
def cross_val_oof(X, y, groups, xgb_params, n_folds, seeds
                  ) -> Tuple[np.ndarray, List, List]:
    gkf        = GroupKFold(n_splits=n_folds)
    oof_probs  = np.zeros(len(y), dtype=np.float64)
    xgb_models, lgb_models = [], []

    # Mirror key LGB params from XGB HPO result for coherent diversity
    lgb_base = {
        "objective":         "binary",
        "metric":            "auc",
        "verbosity":         -1,
        "boosting_type":     "gbdt",
        "device_type":       LGB_DEVICE,
        "n_jobs":            -1,
        "num_leaves":        63,
        "learning_rate":     xgb_params.get("learning_rate", 0.05),
        "colsample_bytree":  xgb_params.get("colsample_bytree", 0.8),
        "subsample":         xgb_params.get("subsample", 0.8),
        "subsample_freq":    1,
        "reg_alpha":         xgb_params.get("reg_alpha", 0.1),
        "reg_lambda":        xgb_params.get("reg_lambda", 1.0),
        "scale_pos_weight":  xgb_params.get("scale_pos_weight", 1.0),
        "min_child_samples": max(5, int(xgb_params.get("min_child_weight", 20))),
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info("── Fold %d/%d ──", fold_idx + 1, n_folds)
        X_tr, y_tr   = X[tr_idx],  y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        xgb_fold = np.zeros(len(val_idx), dtype=np.float64)
        lgb_fold = np.zeros(len(val_idx), dtype=np.float64)

        for seed in seeds:
            # XGBoost GPU
            xp = xgb_params.copy(); xp["seed"] = seed
            n_rounds = xp.pop("n_estimators", 1000)
            dtrain  = xgb.DMatrix(X_tr,  label=y_tr)
            dval_dm = xgb.DMatrix(X_val, label=y_val)
            bst_x = xgb.train(xp, dtrain, num_boost_round=n_rounds,
                               evals=[(dval_dm, "val")], early_stopping_rounds=50,
                               verbose_eval=False)
            xgb_fold += bst_x.predict(dval_dm)
            xgb_models.append(bst_x)
            _flush_gpu()

            # LightGBM CPU (diversity)
            lp = lgb_base.copy(); lp["random_state"] = seed
            ds_tr  = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=True)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr, free_raw_data=True)
            bst_l = lgb.train(lp, ds_tr, num_boost_round=1500, valid_sets=[ds_val],
                               callbacks=[lgb.early_stopping(50, verbose=False),
                                          lgb.log_evaluation(-1)])
            lgb_fold += bst_l.predict(X_val)
            lgb_models.append(bst_l)
            gc.collect()

        xgb_fold /= len(seeds)
        lgb_fold /= len(seeds)
        fold_probs = XGB_BLEND_WEIGHT * xgb_fold + (1 - XGB_BLEND_WEIGHT) * lgb_fold
        oof_probs[val_idx] = fold_probs

        logger.info("  Fold %d | Blend %.4f | XGB %.4f | LGB %.4f",
                    fold_idx + 1,
                    roc_auc_score(y_val, fold_probs),
                    roc_auc_score(y_val, xgb_fold),
                    roc_auc_score(y_val, lgb_fold))

    logger.info("OOF AUC: %.4f", roc_auc_score(y, oof_probs))
    return oof_probs, xgb_models, lgb_models

# ──────────────────────────────────────────────────────────────
# STACKING + CALIBRATION
# ──────────────────────────────────────────────────────────────
def build_stack_calibrator(oof_probs, y):
    logger.info("Fitting stacking meta-learner ...")
    stacker = LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_STATE)
    stacker.fit(oof_probs.reshape(-1, 1), y)
    stk = stacker.predict_proba(oof_probs.reshape(-1, 1))[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(stk, y)
    return stacker, cal

# ──────────────────────────────────────────────────────────────
# ENSEMBLE PREDICT
# ──────────────────────────────────────────────────────────────
def ensemble_predict(xgb_models, lgb_models, X,
                     stacker=None, calibrator=None) -> np.ndarray:
    dmat    = xgb.DMatrix(X)
    xgb_raw = np.mean([m.predict(dmat) for m in xgb_models], axis=0)
    lgb_raw = np.mean([m.predict(X)    for m in lgb_models], axis=0)
    raw = XGB_BLEND_WEIGHT * xgb_raw + (1 - XGB_BLEND_WEIGHT) * lgb_raw
    if stacker    is not None: raw = stacker.predict_proba(raw.reshape(-1, 1))[:, 1]
    if calibrator is not None: raw = calibrator.transform(raw)
    return raw

# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
def _save_roc_pr(probs, y_true, tag) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, probs)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUROC={roc_auc_score(y_true,probs):.4f}")
    axes[0].plot([0,1],[0,1],"k--"); axes[0].set(title="ROC",xlabel="FPR",ylabel="TPR")
    axes[0].legend()
    prec, rec, _ = precision_recall_curve(y_true, probs)
    axes[1].plot(rec, prec, lw=2, label=f"AUPRC={average_precision_score(y_true,probs):.4f}")
    axes[1].set(title="Precision-Recall",xlabel="Recall",ylabel="Precision"); axes[1].legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"roc_pr_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    logger.info("Saved → %s", path)

def _save_reliability(probs, y_true, tag, n_bins=10) -> None:
    bins  = np.linspace(0, 1, n_bins + 1)
    mid, obs = [], []
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i+1])
        if m.sum() > 0:
            mid.append(probs[m].mean())
            obs.append(y_true[m].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mid, obs, "o-", label="Model")
    ax.plot([0,1],[0,1],"k--", label="Perfect")
    ax.set(title=f"Reliability Diagram ({tag})", xlabel="Mean Predicted Prob",
           ylabel="Observed Readmission Rate"); ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"reliability_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    logger.info("Saved → %s", path)

def _save_shap(xgb_models, X_sample, feature_names) -> None:
    logger.info("Computing SHAP (first XGB model, ≤2000 rows) ...")
    X_s = X_sample[:2000]
    explainer = shap.TreeExplainer(xgb_models[0])
    shap_vals = explainer.shap_values(X_s)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_s, feature_names=feature_names,
                      show=False, max_display=30)
    path = os.path.join(FIGURES_DIR, "shap_baseline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Saved → %s", path)

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def run_training() -> None:
    _set_seed(RANDOM_STATE)
    start = datetime.now()

    # 1. Load
    X_df, y, groups = load_data()
    feature_names   = X_df.columns.tolist()
    hadm_ids        = None  # not needed here but kept for parity
    X_all = X_df.values.astype(np.float32); del X_df; gc.collect()

    # 2. Patient-level split (identical to gated model for fair comparison)
    rng  = np.random.RandomState(RANDOM_STATE)
    pats = list(np.unique(groups.values)); rng.shuffle(pats)
    n_te = int(len(pats) * 0.15);  n_val = int(len(pats) * 0.15)
    te_p  = set(pats[:n_te])
    val_p = set(pats[n_te : n_te + n_val])
    tr_p  = set(pats[n_te + n_val :])

    mask_tr  = groups.isin(tr_p).values
    mask_val = groups.isin(val_p).values
    mask_te  = groups.isin(te_p).values

    X_tr,  y_tr  = X_all[mask_tr],  y.values[mask_tr]
    X_val, y_val = X_all[mask_val], y.values[mask_val]
    X_te,  y_te  = X_all[mask_te],  y.values[mask_te]

    pos_weight = float((y_tr == 0).sum()) / float((y_tr == 1).sum())
    logger.info("pos_weight: %.2f", pos_weight)

    # 3. HPO
    best_xgb = optimize_xgb(X_tr, y_tr, X_val, y_val,
                              pos_weight, n_trials=TRAIN_OPTUNA_TRIALS)
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, "best_xgb_params.pkl"))

    # 4. OOF on train+val combined (same as gated model)
    X_cv      = np.vstack([X_tr, X_val])
    y_cv      = np.concatenate([y_tr, y_val])
    groups_cv = np.concatenate([groups.values[mask_tr], groups.values[mask_val]])

    oof_probs, xgb_models, lgb_models = cross_val_oof(
        X_cv, y_cv, groups_cv, best_xgb.copy(), TRAIN_N_FOLDS, TRAIN_SEEDS)

    # 5. Threshold on OOF
    best_thresh = find_best_threshold(oof_probs, y_cv, strategy=TRAIN_THRESHOLD_STRATEGY)

    # 6. Stack + calibrate
    stacker = calibrator = None
    if TRAIN_ENABLE_STACK:
        stacker, calibrator = build_stack_calibrator(oof_probs, y_cv)
        joblib.dump(stacker,    os.path.join(MODELS_DIR, "stacker.pkl"))
        joblib.dump(calibrator, os.path.join(MODELS_DIR, "calibrator.pkl"))

    # 7. Test
    test_probs     = ensemble_predict(xgb_models, lgb_models, X_te, stacker, calibrator)
    preds          = (test_probs >= best_thresh).astype(int)
    ece_before_cal = compute_ece(
        ensemble_predict(xgb_models, lgb_models, X_te), y_te)
    ece_after_cal  = compute_ece(test_probs, y_te)

    auc   = roc_auc_score(y_te, test_probs)
    ap    = average_precision_score(y_te, test_probs)
    brier = brier_score_loss(y_te, test_probs)
    ll    = log_loss(y_te, test_probs)
    mcc   = matthews_corrcoef(y_te, preds)
    f1    = f1_score(y_te, preds, zero_division=0)
    elapsed = (datetime.now() - start).total_seconds()

    logger.info("=" * 55)
    logger.info("TRANCE BASELINE — TEST RESULTS")
    logger.info("  AUROC  (calibrated): %.4f", auc)
    logger.info("  AUPRC               : %.4f", ap)
    logger.info("  MCC                 : %.4f", mcc)
    logger.info("  F1                  : %.4f", f1)
    logger.info("  Brier score         : %.4f", brier)
    logger.info("  LogLoss             : %.4f", ll)
    logger.info("  ECE before cal      : %.4f", ece_before_cal)
    logger.info("  ECE after cal       : %.4f", ece_after_cal)
    logger.info("  Threshold (%s)     : %.3f", TRAIN_THRESHOLD_STRATEGY, best_thresh)
    logger.info("  Wall time           : %.0f s", elapsed)
    logger.info("=" * 55)

    # 8. Save artefacts
    for i, m in enumerate(xgb_models):
        m.save_model(os.path.join(MODELS_DIR, f"xgb_model_{i}.ubj"))
    for i, m in enumerate(lgb_models):
        m.save_model(os.path.join(MODELS_DIR, f"lgb_model_{i}.txt"))

    # Predictions CSV for final ensembling with gated model
    test_indices = np.where(mask_te)[0]
    pd.DataFrame({
        "idx":       test_indices,
        "y_true":    y_te,
        "prob_raw":  ensemble_predict(xgb_models, lgb_models, X_te),
        "prob_cal":  test_probs,
        "pred":      preds,
    }).to_csv(os.path.join(RESULTS_DIR, "trance_baseline_preds.csv"), index=False)

    # OOF predictions for ensembling
    pd.DataFrame({
        "prob_oof": oof_probs,
        "y_true":   y_cv,
    }).to_csv(os.path.join(RESULTS_DIR, "trance_baseline_oof.csv"), index=False)

    metrics = dict(auroc=auc, auprc=ap, mcc=mcc, f1=f1, brier=brier,
                   logloss=ll, ece_before=ece_before_cal, ece_after=ece_after_cal,
                   threshold=best_thresh, wall_time_s=elapsed)
    with open(os.path.join(RESULTS_DIR, "trance_baseline_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Feature importance
    fi = xgb_models[0].get_score(importance_type="gain")
    (pd.DataFrame({"feature": list(fi.keys()), "gain": list(fi.values())})
       .sort_values("gain", ascending=False)
       .to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False))

    # Plots
    _save_roc_pr(test_probs, y_te, "baseline_test")
    _save_roc_pr(oof_probs,  y_cv, "baseline_oof")
    _save_reliability(test_probs, y_te, "baseline_cal")
    _save_reliability(ensemble_predict(xgb_models, lgb_models, X_te), y_te, "baseline_raw")
    try:
        _save_shap(xgb_models, X_te, feature_names)
    except Exception as exc:
        logger.warning("SHAP skipped: %s", exc)

    with open(os.path.join(MODELS_DIR, "threshold.json"), "w") as fh:
        json.dump({"threshold": best_thresh, "strategy": TRAIN_THRESHOLD_STRATEGY}, fh)

    logger.info("All artefacts saved → %s  %s", MODELS_DIR, RESULTS_DIR)


if __name__ == "__main__":
    run_training()
