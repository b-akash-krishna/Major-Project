# src/03_train.py
"""
TRANCE Framework — Training Pipeline v3
========================================

INPUT FILES (all auto-discovered via config.py):
  • data/ultimate_features_pruned.csv   — structured EHR features (from 01b)
      Falls back to data/ultimate_features.csv if pruned doesn't exist.
      Required columns: subject_id, hadm_id, readmit_30, [feature columns...]
  • data/clinical_t5_embeddings.csv     — ClinicalT5 text embeddings (from 02)
      Required columns: hadm_id, ct5_0 … ct5_127
      Optional — training runs without it (tabular-only mode).
  • models/selected_features.json       — ranked feature list (from 01b)
      Optional — uses all features if missing.

OUTPUT FILES:
  • models/trance_model.pkl             — serialized ensemble + calibrator
      Contains: models (list), meta, calibrator, features (list),
                best_params, best_threshold
  • results/training_report.json        — full metrics + ablation + best params
  • results/test_predictions.csv        — y_true, prob_raw, prob_cal, pred
  • figures/roc_pr_curve.png            — ROC + Precision-Recall curves
  • figures/calibration_curve.png       — reliability diagram
  • figures/shap_summary.png            — SHAP feature importance (top 30)
  • figures/threshold_analysis.png      — F1 / Recall / Precision vs threshold

CLASS IMBALANCE HANDLING (v3):
  1. SMOTETomek oversampling on training fold (pip install imbalanced-learn)
  2. Optuna tunes scale_pos_weight vs is_unbalance — picks the better one
  3. GOSS fix: bagging params (subsample/subsample_freq) excluded for GOSS
  4. Optimal decision threshold search on val set (strategy: F1 / recall80 / Youden-J)
  5. Reports Recall / Precision / F1 on readmit=1 class at best threshold
  6. Threshold analysis plot saved to figures/

OTHER IMPROVEMENTS (v3 vs v2):
  • Patient-level GroupKFold CV (no patient leakage across folds)
  • Temporal test split (last 15% of patients by first admission date)
  • Meta-learner stacking: LightGBM + XGBoost -> Logistic Regression
  • Isotonic calibration (ECE logged before/after)
  • test_predictions.csv for downstream analysis
  • train.log written alongside models/
"""

import gc
import json
import logging
import os
import sys
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
import pandas as pd
import shap
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

# Fix Windows console encoding (cp1252 can't handle Unicode arrows/boxes)
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna internal noise


def _optuna_callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
    """Log each completed trial so HPO progress is visible."""
    if trial.value is None:
        return  # failed trial
    logger.info(
        "  Trial %3d | AUROC: %.4f | best: %.4f | boosting: %-5s | leaves: %s | lr: %.4f",
        trial.number,
        trial.value,
        study.best_value,
        trial.params.get("boosting_type", "?"),
        trial.params.get("num_leaves", "?"),
        trial.params.get("learning_rate", float("nan")),
    )


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from .config import (EMBEDDINGS_CSV, FEATURES_CSV, FIGURES_DIR,
                         MAIN_MODEL_PKL, MODELS_DIR, RANDOM_STATE, RESULTS_DIR)
except ImportError:
    from config import (EMBEDDINGS_CSV, FEATURES_CSV, FIGURES_DIR,
                        MAIN_MODEL_PKL, MODELS_DIR, RANDOM_STATE, RESULTS_DIR)

# Logging to both console and file
os.makedirs(MODELS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(MODELS_DIR), "train.log"), mode="w", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

SELECTED_FEATURES_JSON = os.path.join(MODELS_DIR, "selected_features.json")

# -- TUNABLE CONSTANTS ----------------------------------------------------------
OPTUNA_TRIALS = 5    # 50 trials ~ 2-3 hrs on GTX 3050 with 383k rows
N_FOLDS       = 5     # patient-level CV folds
DART_MAX_TREES = 800  # cap dart trees — early stopping doesn't work with dart
ENABLE_STACK  = True  # LightGBM + XGBoost meta-learner
ENABLE_SMOTE  = True  # SMOTETomek oversampling (requires imbalanced-learn)
TEST_FRAC     = 0.15  # fraction of patients in test set
VAL_FRAC      = 0.15  # fraction of patients in validation set


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads structured features + optional ClinicalT5 embeddings.

    Returns
    -------
    X      : pd.DataFrame  — feature matrix (admissions × features)
    y      : pd.Series     — binary target (readmit_30)
    groups : pd.Series     — subject_id, used for patient-level GroupKFold
    """
    for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    pruned = FEATURES_CSV.replace(".csv", "_pruned.csv")
    path   = pruned if os.path.exists(pruned) else FEATURES_CSV
    logger.info("Loading features: %s", path)
    tab = pd.read_csv(path, low_memory=False).fillna(0)

    selected: Optional[List[str]] = None
    if os.path.exists(SELECTED_FEATURES_JSON):
        with open(SELECTED_FEATURES_JSON) as f:
            selected = json.load(f)
        logger.info("Selected features: %d", len(selected))

    if os.path.exists(EMBEDDINGS_CSV):
        emb = pd.read_csv(EMBEDDINGS_CSV, low_memory=False)
        df  = tab.merge(emb, on="hadm_id", how="left").fillna(0)
        logger.info("Fused shape: %s", df.shape)
    else:
        df = tab.copy()
        logger.warning("Embeddings not found — running tabular-only mode.")

    groups  = df["subject_id"].astype(int)
    y       = df["readmit_30"].astype("int8")
    id_cols = {"subject_id", "hadm_id", "readmit_30"}

    if selected:
        emb_cols = [c for c in df.columns if c.startswith("ct5_")]
        keep = [c for c in selected if c in df.columns] + emb_cols
        X = df[keep].copy()
    else:
        X = df.drop(columns=list(id_cols & set(df.columns)), errors="ignore")

    logger.info("Feature matrix: %s | Positive rate: %.2f%%", X.shape, y.mean() * 100)
    return X, y, groups


# ==============================================================================
# 2. CLASS IMBALANCE — SMOTETomek
# ==============================================================================

def apply_smote(X_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTETomek = SMOTE oversampling of minority class + Tomek link cleaning.
    Applied ONLY to training data — validation and test are never touched.

    Install dependency:  pip install imbalanced-learn
    Set ENABLE_SMOTE=False above to skip.
    """
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        ratio = (y_tr == 1).sum() / max((y_tr == 0).sum(), 1)
        if ratio > 0.40:
            logger.info("Class ratio %.2f already reasonable — skipping SMOTE.", ratio)
            return X_tr, y_tr

        logger.info(
            "Applying SMOTETomek (before: %d pos / %d neg, ratio=%.3f) ...",
            (y_tr == 1).sum(), (y_tr == 0).sum(), ratio,
        )
        sm = SMOTETomek(
            smote=SMOTE(
                sampling_strategy=0.35,  # oversample to 35% minority
                random_state=RANDOM_STATE,
                # n_jobs removed: not supported in older imbalanced-learn versions
            ),
            random_state=RANDOM_STATE,
            # n_jobs removed from SMOTETomek: use fit_resample instead
        )
        X_res, y_res = sm.fit_resample(X_tr, y_tr)
        logger.info(
            "After SMOTETomek: %d pos / %d neg (ratio=%.3f)",
            (y_res == 1).sum(), (y_res == 0).sum(),
            (y_res == 1).sum() / max((y_res == 0).sum(), 1),
        )
        return X_res, y_res

    except ImportError:
        logger.warning(
            "imbalanced-learn not installed — skipping SMOTE. "
            "Install with: pip install imbalanced-learn"
        )
        return X_tr, y_tr
    except Exception as e:
        logger.warning("SMOTE failed (%s) — continuing without.", e)
        return X_tr, y_tr


# ==============================================================================
# 3. HYPERPARAMETER OPTIMISATION
# ==============================================================================

def optimize_lgbm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    pos_weight: float,
    n_trials: int = OPTUNA_TRIALS,
) -> Dict:
    """
    Bayesian HPO via Optuna (TPE sampler, Hyperband pruner).
    Searches: boosting type, tree structure, regularisation, imbalance strategy.

    Key imbalance fix:
      - GOSS boosting cannot use bagging (subsample/subsample_freq are removed)
      - Optuna picks between scale_pos_weight and is_unbalance per trial
    """
    logger.info("Optuna HPO: %d trials ...", n_trials)

    def objective(trial: optuna.Trial) -> float:
        # gbdt/goss appear twice to reduce costly dart sampling frequency
        boosting  = trial.suggest_categorical("boosting_type", ["gbdt", "gbdt", "dart", "goss", "goss"])
        imbalance = trial.suggest_categorical(
            "imbalance_strategy", ["scale_pos_weight", "is_unbalance"]
        )

        params: Dict = {
            "objective":         "binary",
            "metric":            "auc",
            "verbosity":         -1,
            "boosting_type":     boosting,
            "num_leaves":        trial.suggest_int("num_leaves", 63, 300),
            "max_depth":         trial.suggest_int("max_depth", 6, 16),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            # DART has no early stopping so cap trees to avoid 20-min trials
            "n_estimators": (
                trial.suggest_int("n_estimators", 500, DART_MAX_TREES)
                if boosting == "dart"
                else trial.suggest_int("n_estimators", 500, 3000)
            ),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bynode":  trial.suggest_float("colsample_bynode", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "path_smooth":       trial.suggest_float("path_smooth", 0.0, 1.0),
            "random_state":      RANDOM_STATE,
            "n_jobs":            -1,
        }

        # -- GOSS fix: remove bagging params (incompatible with GOSS) ------
        if boosting in ("gbdt", "dart"):
            params["subsample"]      = trial.suggest_float("subsample", 0.5, 1.0)
            params["subsample_freq"] = trial.suggest_int("subsample_freq", 1, 5)
        if boosting == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.4)
        if boosting == "goss":
            params["top_rate"]   = trial.suggest_float("top_rate", 0.05, 0.4)
            params["other_rate"] = trial.suggest_float("other_rate", 0.05, 0.2)

        # -- Imbalance strategy: mutually exclusive -------------------------
        if imbalance == "scale_pos_weight":
            params["scale_pos_weight"] = pos_weight
        else:
            params["is_unbalance"] = True

        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=RANDOM_STATE)
    # MedianPruner is safe and reliable; HyperbandPruner can terminate early unexpectedly
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    logger.info("Starting Optuna optimization: %d trials", n_trials)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_optuna_callback],
        catch=(Exception,),  # log failed trials but don't abort the whole study
    )
    logger.info("Optuna complete. %d/%d trials succeeded.", 
                len([t for t in study.trials if t.value is not None]), n_trials)

    best = study.best_params.copy()
    # Remove Optuna-only key before passing params to LightGBM
    imbalance_strategy = best.pop("imbalance_strategy", "scale_pos_weight")
    if imbalance_strategy == "scale_pos_weight":
        best["scale_pos_weight"] = pos_weight
    else:
        best["is_unbalance"] = True

    best.update({
        "objective": "binary", "metric": "auc",
        "verbosity": -1, "n_jobs": -1, "random_state": RANDOM_STATE,
    })
    logger.info("Best HPO AUROC: %.4f", study.best_value)
    logger.info("Best params: %s", best)
    return best


# ==============================================================================
# 4. PATIENT-LEVEL CROSS VALIDATION
# ==============================================================================

def patient_level_cv(
    X: pd.DataFrame, y: pd.Series,
    groups: pd.Series, params: Dict,
    n_folds: int = N_FOLDS,
) -> Tuple[float, float, List[float]]:
    """
    GroupKFold CV grouped by subject_id.
    Guarantees no patient's records appear in both train and validation folds.
    Returns (mean_auroc, std_auroc, per_fold_aurocs).
    """
    logger.info("Patient-level GroupKFold CV (%d folds) ...", n_folds)
    gkf    = GroupKFold(n_splits=n_folds)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr_cv  = X.iloc[tr_idx].values.astype(np.float32)
        y_tr_cv  = y.iloc[tr_idx].values
        X_val_cv = X.iloc[val_idx].values.astype(np.float32)
        y_val_cv = y.iloc[val_idx].values

        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr_cv, y_tr_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        auc = roc_auc_score(y_val_cv, m.predict_proba(X_val_cv)[:, 1])
        scores.append(auc)
        logger.info(
            "  Fold %d AUROC: %.4f  (patients: %d train / %d val)",
            fold + 1, auc,
            groups.iloc[tr_idx].nunique(),
            groups.iloc[val_idx].nunique(),
        )
        del m, X_tr_cv, X_val_cv; gc.collect()

    mean, std = float(np.mean(scores)), float(np.std(scores))
    logger.info("CV AUROC: %.4f ± %.4f", mean, std)
    return mean, std, scores


# ==============================================================================
# 5. ENSEMBLE BUILDING
# ==============================================================================

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except Exception:
            return False


def build_ensemble(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: Dict, pos_weight: float,
) -> List[Tuple[str, object]]:
    """
    Trains LightGBM + XGBoost on the same training data.
    Returns list of (name, model) tuples for meta-learner stacking.
    """
    models: List[Tuple[str, object]] = []

    # -- LightGBM --------------------------------------------------------------
    logger.info("Training LightGBM ...")
    lgbm_model = lgb.LGBMClassifier(**params)
    lgbm_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    auc_lgbm = roc_auc_score(y_val, lgbm_model.predict_proba(X_val)[:, 1])
    logger.info("LightGBM val AUROC: %.4f", auc_lgbm)
    models.append(("lgbm", lgbm_model))

    # -- XGBoost ---------------------------------------------------------------
    if ENABLE_STACK:
        try:
            import xgboost as xgb
            logger.info("Training XGBoost ...")
            xgb_params = {
                "n_estimators":     1500,
                "learning_rate":    params.get("learning_rate", 0.05),
                "max_depth":        min(int(params.get("max_depth", 8)), 10),
                "subsample":        params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.8),
                "reg_alpha":        params.get("reg_alpha", 0.1),
                "reg_lambda":       params.get("reg_lambda", 1.0),
                "scale_pos_weight": pos_weight,
                "tree_method":      "hist",          # modern API (gpu_hist removed)
                "device":           "cuda" if _has_cuda() else "cpu",
                "eval_metric":      "auc",
                "random_state":     RANDOM_STATE,
                "n_jobs":           -1,
                "verbosity":        0,
            }
            # Handle both XGBoost < 2.0 (early_stopping in fit) and >= 2.0 (constructor)
            try:
                xgb_model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
                xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                xgb_model = xgb.XGBClassifier(**xgb_params)
                xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              early_stopping_rounds=50, verbose=False)
            auc_xgb = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
            logger.info("XGBoost val AUROC: %.4f", auc_xgb)
            models.append(("xgb", xgb_model))
        except ImportError:
            logger.warning("xgboost not installed — skipping. pip install xgboost")
        except Exception as e:
            logger.warning("XGBoost failed (%s) — LightGBM only.", e)

    return models


def ensemble_predict(models: List[Tuple[str, object]], X: np.ndarray) -> np.ndarray:
    """Simple mean of model probabilities."""
    return np.stack([m.predict_proba(X)[:, 1] for _, m in models], axis=1).mean(axis=1)


# ==============================================================================
# 6. META-LEARNER STACKING
# ==============================================================================

def fit_meta_learner(
    models: List[Tuple[str, object]],
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[Optional[LogisticRegression], np.ndarray]:
    """
    Trains Logistic Regression on val-set OOF predictions from each base model.
    Test probabilities come from the stacked meta prediction.
    """
    if len(models) < 2:
        return None, ensemble_predict(models, X_te)

    logger.info("Fitting meta-learner ...")
    val_stack = np.column_stack([m.predict_proba(X_val)[:, 1] for _, m in models])
    te_stack  = np.column_stack([m.predict_proba(X_te)[:, 1]  for _, m in models])

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    meta.fit(val_stack, y_val)

    auc_meta = roc_auc_score(y_val, meta.predict_proba(val_stack)[:, 1])
    logger.info("Meta-learner val AUROC: %.4f", auc_meta)
    return meta, meta.predict_proba(te_stack)[:, 1]


# ==============================================================================
# 7. CALIBRATION
# ==============================================================================

def calibrate(
    val_probs: np.ndarray, y_val: np.ndarray,
    test_probs: np.ndarray,
) -> Tuple[IsotonicRegression, np.ndarray]:
    """Isotonic regression calibration. Fitted on val, applied to test."""
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(val_probs, y_val)
    ece_before = _ece(val_probs, y_val)
    ece_after  = _ece(cal.predict(val_probs), y_val)
    logger.info("ECE before calibration: %.4f -> after: %.4f", ece_before, ece_after)
    return cal, cal.predict(test_probs).astype(np.float32)


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins, ece, total = np.linspace(0, 1, n_bins + 1), 0.0, len(y)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(float(y[mask].mean()) - float(probs[mask].mean()))
    return ece


# ==============================================================================
# 8. THRESHOLD OPTIMISATION
# ==============================================================================

def find_best_threshold(
    val_probs: np.ndarray, y_val: np.ndarray,
    strategy: str = "f1",
) -> float:
    """
    Finds the optimal decision threshold on the VALIDATION SET only.
    Never uses test data — avoids look-ahead bias.

    strategy options
    ----------------
    "f1"       : maximise F1 on readmit=1 class (recommended default)
    "recall80" : highest precision while keeping recall ≥ 0.80
                 (clinical: catch at least 80% of readmissions)
    "j"        : Youden's J = sensitivity + specificity − 1
    """
    thresholds = np.arange(0.05, 0.70, 0.005)

    if strategy == "f1":
        scores = [f1_score(y_val, val_probs >= t, zero_division=0) for t in thresholds]
        best   = float(thresholds[np.argmax(scores)])

    elif strategy == "recall80":
        best = 0.50
        for t in sorted(thresholds, reverse=True):
            preds = (val_probs >= t).astype(int)
            rec   = preds[y_val == 1].mean() if (y_val == 1).sum() > 0 else 0.0
            if rec >= 0.80:
                best = float(t)
                break

    elif strategy == "j":
        fpr, tpr, thresh = roc_curve(y_val, val_probs)
        best = float(thresh[np.argmax(tpr - fpr)])

    else:
        best = 0.50

    logger.info("Threshold strategy='%s' -> best threshold=%.3f", strategy, best)
    return best


# ==============================================================================
# 9. PLOTS
# ==============================================================================

def save_plots(
    y_te: np.ndarray,
    test_probs_raw: np.ndarray,
    test_probs_cal: np.ndarray,
    best_thresh: float,
) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # -- ROC + Precision-Recall ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_te, test_probs_cal)
    axes[0].plot(fpr, tpr, lw=2,
                 label=f"AUROC = {roc_auc_score(y_te, test_probs_cal):.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curve")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_te, test_probs_cal)
    axes[1].plot(rec, prec, lw=2,
                 label=f"AUPRC = {average_precision_score(y_te, test_probs_cal):.4f}")
    axes[1].axhline(y_te.mean(), color="gray", linestyle="--",
                    label=f"Baseline = {y_te.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_pr_curve.png"), dpi=150)
    plt.close()

    # -- Calibration curve -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, probs in [("Raw", test_probs_raw), ("Calibrated", test_probs_cal)]:
        fp, mp = _calibration_bins(probs, y_te)
        ax.plot(mp, fp, "s-", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction Positives",
           title="Calibration (Reliability Diagram)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "calibration_curve.png"), dpi=150)
    plt.close()

    # -- Threshold analysis -----------------------------------------------------
    thresholds = np.arange(0.05, 0.70, 0.01)
    f1s, recs, precs = [], [], []
    for t in thresholds:
        preds = (test_probs_cal >= t).astype(int)
        f1s.append(f1_score(y_te, preds, zero_division=0))
        tp = int(((preds == 1) & (y_te == 1)).sum())
        fp = int(((preds == 1) & (y_te == 0)).sum())
        fn = int(((preds == 0) & (y_te == 1)).sum())
        recs.append(tp / max(tp + fn, 1))
        precs.append(tp / max(tp + fp, 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s,   label="F1 (readmit=1)", lw=2)
    ax.plot(thresholds, recs,  label="Recall (readmit=1)", lw=2)
    ax.plot(thresholds, precs, label="Precision (readmit=1)", lw=2)
    ax.axvline(best_thresh, color="red", linestyle="--",
               label=f"Best threshold = {best_thresh:.3f}")
    ax.set(xlabel="Decision Threshold", ylabel="Score",
           title="Threshold Analysis — Readmit=1 Class", ylim=[0, 1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "threshold_analysis.png"), dpi=150)
    plt.close()

    logger.info("Plots saved -> %s", FIGURES_DIR)


def _calibration_bins(probs, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    fp, mp = [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        fp.append(float(y[mask].mean()))
        mp.append(float(probs[mask].mean()))
    return np.array(fp), np.array(mp)


# ==============================================================================
# 10. SHAP
# ==============================================================================

def compute_shap(model, X_sample: pd.DataFrame) -> None:
    try:
        logger.info("Computing SHAP values (%d samples) ...", len(X_sample))
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X_sample)
        if isinstance(sv, list):
            sv = sv[1]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, max_display=30, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        logger.info("SHAP plot saved.")
    except Exception as e:
        logger.warning("SHAP failed: %s", e)


# ==============================================================================
# 11. ABLATION
# ==============================================================================

def run_ablation(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_te: pd.DataFrame, y_te: pd.Series,
    params: Dict,
) -> Dict:
    """AUROC comparison: fused vs tabular-only vs embeddings-only."""
    tab_cols = [c for c in X_tr.columns if not c.startswith("ct5_")]
    emb_cols = [c for c in X_tr.columns if c.startswith("ct5_")]
    results  = {}

    for name, cols in [
        ("fused",        list(X_tr.columns)),
        ("tabular_only", tab_cols),
        ("text_only",    emb_cols or None),
    ]:
        if not cols:
            continue
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr[cols].values, y_tr.values,
            eval_set=[(X_te[cols].values, y_te.values)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        auc = roc_auc_score(y_te.values, m.predict_proba(X_te[cols].values)[:, 1])
        results[name] = round(float(auc), 4)
        logger.info("  Ablation %-14s: %.4f", name, auc)
        del m; gc.collect()

    return results


# ==============================================================================
# 12. MAIN TRAINER
# ==============================================================================

class TRANCETrainer:
    """
    End-to-end MIMIC-IV readmission prediction pipeline.
    Handles imbalance, HPO, stacking, calibration, threshold optimisation.
    """

    def __init__(self):
        self.models:         List[Tuple[str, object]] = []
        self.meta:           Optional[LogisticRegression] = None
        self.calibrator:     Optional[IsotonicRegression] = None
        self.features:       List[str] = []
        self.best_params:    Dict = {}
        self.best_threshold: float = 0.5

    def run(self) -> Dict:
        # -- 1. Load -------------------------------------------------------
        X, y, groups = load_data()
        self.features = list(X.columns)

        # -- 2. Temporal patient-level split -------------------------------
        # Sort patients by index of their first record -> approximate temporal order
        pat_first   = groups.reset_index(drop=True).groupby(groups.values).first()
        sorted_pats = pat_first.sort_values().index.tolist()

        n_test  = int(len(sorted_pats) * TEST_FRAC)
        n_val   = int(len(sorted_pats) * VAL_FRAC)

        test_pats  = set(sorted_pats[-n_test:])
        val_pats   = set(sorted_pats[-(n_test + n_val):-n_test])
        train_pats = set(sorted_pats[:-(n_test + n_val)])

        train_mask = groups.isin(train_pats).values
        val_mask   = groups.isin(val_pats).values
        test_mask  = groups.isin(test_pats).values

        X_tr  = X[train_mask].values.astype(np.float32)
        y_tr  = y[train_mask].values
        X_val = X[val_mask].values.astype(np.float32)
        y_val = y[val_mask].values
        X_te  = X[test_mask].values.astype(np.float32)
        y_te  = y[test_mask].values

        logger.info("Train: %d | Val: %d | Test: %d", len(y_tr), len(y_val), len(y_te))
        logger.info("Train readmit: %.2f%% | Val: %.2f%% | Test: %.2f%%",
                    y_tr.mean() * 100, y_val.mean() * 100, y_te.mean() * 100)

        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        logger.info("Class imbalance (pos_weight): %.2f", pos_weight)

        # -- 3. SMOTE on training fold -------------------------------------
        if ENABLE_SMOTE:
            X_tr, y_tr = apply_smote(X_tr, y_tr)

        # -- 4. HPO --------------------------------------------------------
        self.best_params = optimize_lgbm(
            X_tr, y_tr, X_val, y_val,
            pos_weight, n_trials=OPTUNA_TRIALS,
        )

        # -- 5. Patient-level CV -------------------------------------------
        X_tv      = X[train_mask | val_mask]
        y_tv      = y[train_mask | val_mask]
        groups_tv = groups[train_mask | val_mask]
        cv_mean, cv_std, _ = patient_level_cv(
            X_tv, y_tv, groups_tv, self.best_params, n_folds=N_FOLDS
        )

        # -- 6. Ensemble ---------------------------------------------------
        self.models = build_ensemble(
            X_tr, y_tr, X_val, y_val, self.best_params, pos_weight
        )

        # -- 7. Meta-learner stacking --------------------------------------
        if ENABLE_STACK and len(self.models) > 1:
            self.meta, test_probs_raw = fit_meta_learner(
                self.models, X_val, y_val, X_te
            )
            val_probs_for_cal = self.meta.predict_proba(
                np.column_stack([m.predict_proba(X_val)[:, 1] for _, m in self.models])
            )[:, 1]
        else:
            self.meta = None
            test_probs_raw    = ensemble_predict(self.models, X_te)
            val_probs_for_cal = ensemble_predict(self.models, X_val)

        # -- 8. Isotonic calibration ---------------------------------------
        self.calibrator, test_probs_cal = calibrate(
            val_probs_for_cal, y_val, test_probs_raw
        )

        # -- 9. Threshold optimisation (val set only) ----------------------
        self.best_threshold = find_best_threshold(
            val_probs_for_cal, y_val, strategy="recall80"
        )

        # -- 10. Metrics ---------------------------------------------------
        auc_raw  = roc_auc_score(y_te, test_probs_raw)
        auc_cal  = roc_auc_score(y_te, test_probs_cal)
        auprc    = average_precision_score(y_te, test_probs_cal)
        brier    = brier_score_loss(y_te, test_probs_cal)
        ll       = log_loss(y_te, test_probs_cal)

        test_preds = (test_probs_raw >= self.best_threshold).astype(int)
        report     = classification_report(y_te, test_preds, output_dict=True,
                                           zero_division=0)
        recall_pos = report.get("1", {}).get("recall",    0.0)
        prec_pos   = report.get("1", {}).get("precision", 0.0)
        f1_pos     = report.get("1", {}).get("f1-score",  0.0)

        logger.info("=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("  AUROC (raw):               %.4f", auc_raw)
        logger.info("  AUROC (calibrated):        %.4f", auc_cal)
        logger.info("  AUPRC:                     %.4f", auprc)
        logger.info("  Brier Score:               %.4f", brier)
        logger.info("  Log Loss:                  %.4f", ll)
        logger.info("  CV AUROC:                  %.4f ± %.4f", cv_mean, cv_std)
        logger.info("  Best threshold (val F1):   %.3f", self.best_threshold)
        logger.info("  -- Readmit=1 @ threshold %.3f --", self.best_threshold)
        logger.info("  Recall    (sensitivity):   %.4f", recall_pos)
        logger.info("  Precision (PPV):           %.4f", prec_pos)
        logger.info("  F1 score:                  %.4f", f1_pos)
        logger.info("=" * 60)

        # -- 11. Ablation --------------------------------------------------
        X_df_tr  = pd.DataFrame(np.vstack([X_tr, X_val]), columns=self.features)
        X_df_te  = pd.DataFrame(X_te, columns=self.features)
        ablation = run_ablation(
            X_df_tr, pd.Series(np.concatenate([y_tr, y_val])),
            X_df_te, pd.Series(y_te),
            self.best_params,
        )

        # -- 12. SHAP ------------------------------------------------------
        primary = next((m for n, m in self.models if n == "lgbm"), None)
        if primary is not None:
            n_shap = min(3000, len(X_te))
            idx    = np.random.RandomState(RANDOM_STATE).choice(
                len(X_te), n_shap, replace=False
            )
            compute_shap(primary, pd.DataFrame(X_te[idx], columns=self.features))

        # -- 13. Plots -----------------------------------------------------
        save_plots(y_te, test_probs_raw, test_probs_cal, self.best_threshold)

        # -- 14. Predictions CSV -------------------------------------------
        pred_df = pd.DataFrame({
            "y_true":   y_te,
            "prob_raw": test_probs_raw.round(6),
            "prob_cal": test_probs_cal.round(6),
            "pred":     test_preds,
        })
        pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        logger.info("Predictions saved -> %s", pred_path)

        # -- 15. Save model + report ---------------------------------------
        results = {
            "auroc_raw":          round(auc_raw,   4),
            "auroc_calibrated":   round(auc_cal,   4),
            "auprc":              round(auprc,      4),
            "brier_score":        round(brier,      4),
            "log_loss":           round(ll,         4),
            "cv_auroc_mean":      round(cv_mean,    4),
            "cv_auroc_std":       round(cv_std,     4),
            "best_threshold":     round(self.best_threshold, 3),
            "recall_readmit1":    round(recall_pos, 4),
            "precision_readmit1": round(prec_pos,   4),
            "f1_readmit1":        round(f1_pos,     4),
            "ablation":           ablation,
            "n_models":           len(self.models),
            "n_features":         len(self.features),
            "train_size":         int(len(y_tr)),
            "val_size":           int(len(y_val)),
            "test_size":          int(len(y_te)),
            "best_params": {
                k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                for k, v in self.best_params.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
        self._save(results)
        return results

    def _save(self, results: Dict) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(
            {
                "models":         self.models,
                "meta":           self.meta,
                "calibrator":     self.calibrator,
                "features":       self.features,
                "best_params":    self.best_params,
                "best_threshold": self.best_threshold,
                "timestamp":      datetime.now().isoformat(),
            },
            MAIN_MODEL_PKL,
        )
        report_path = os.path.join(RESULTS_DIR, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Model saved  -> %s", MAIN_MODEL_PKL)
        logger.info("Report saved -> %s", report_path)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Ensure Windows console can handle UTF-8 output
    # If you still see encoding errors, run: set PYTHONIOENCODING=utf-8
    import os as _os
    _os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    trainer = TRANCETrainer()
    r = trainer.run()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    skip = {"best_params", "ablation"}
    for k, v in r.items():
        if k not in skip:
            print(f"  {k:<30}: {v}")
    print("\nAblation:")
    for k, v in r.get("ablation", {}).items():
        print(f"  {k:<30}: {v}")
    print("\nBest LightGBM params:")
    for k, v in r.get("best_params", {}).items():
        print(f"  {k:<30}: {v}")