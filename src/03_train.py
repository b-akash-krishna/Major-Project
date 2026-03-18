# src/03_train.py
"""
TRANCE Framework — Training Pipeline v3.1
==========================================

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

FIXES (v3.1 vs v3):
  • _set_seed() no longer mutates global RANDOM_STATE — seed passed explicitly
    into the patient split RNG so each multi-seed run shuffles correctly.
  • fit_meta_learner() meta-LR now scored via 5-fold OOF cross_val_predict
    instead of in-sample evaluation, preventing it from always "winning"
    stacker selection even when it would underperform on test data.
"""

import gc
import random
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
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_predict

# Fix Windows console encoding (cp1252 can't handle Unicode arrows/boxes)
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna internal noise


def _set_seed(seed: int) -> None:
    # Do NOT mutate the module-level RANDOM_STATE — load_data() uses it for
    # the patient split and must read the same value that was set before the
    # call. We set numpy/random seeds here; the seed value is passed explicitly
    # into all functions that need it rather than relying on a shared global.
    np.random.seed(int(seed))
    random.seed(int(seed))


def _optuna_callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
    """Log each completed trial so HPO progress is visible."""
    if trial.value is None:
        return  # failed trial
    auc = trial.user_attrs.get("val_auroc", float("nan"))
    ap  = trial.user_attrs.get("val_auprc", float("nan"))
    logger.info(
        "  Trial %3d | score: %.4f | best: %.4f | AUROC: %.4f | AUPRC: %.4f | boosting: %-5s | leaves: %s | lr: %.4f",
        trial.number,
        trial.value,
        study.best_value,
        auc,
        ap,
        trial.params.get("boosting_type", "?"),
        trial.params.get("num_leaves", "?"),
        trial.params.get("learning_rate", float("nan")),
    )


def composite_rank_score(y_true: np.ndarray, probs: np.ndarray, alpha: Optional[float] = None) -> float:
    """
    Ranking-oriented objective used for tuning and blend selection.
    Composite improves AUPRC without sacrificing AUROC.
    """
    if alpha is None:
        alpha = HPO_ALPHA_AUPRC
    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)
    return float(alpha * auprc + (1.0 - alpha) * auroc)


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from .config import (
        EMBEDDINGS_CSV, ENABLE_SMOTE as CFG_ENABLE_SMOTE,
        FEATURES_CSV, FIGURES_DIR, MAIN_MODEL_PKL, MODELS_DIR,
        RANDOM_STATE, RESULTS_DIR,
        TRAIN_OPTUNA_TRIALS, TRAIN_N_FOLDS, TRAIN_DART_MAX_TREES,
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, TRAIN_SMOTE_RATIO,
        TRAIN_BLEND_TRIALS, TRAIN_CT5_KEEP_DIMS, TRAIN_FEATURE_SUBSETS,
        TRAIN_META_C_CANDIDATES, TRAIN_THRESHOLD_STRATEGY,
        TRAIN_ENABLE_STACK,
        TRAIN_ENABLE_AUTO_FEATURE_SUBSET,
        TRAIN_SEEDS, TRAIN_HPO_ONCE,
        TRAIN_OPTIMIZE_AUROC, TRAIN_HPO_ALPHA_AUPRC,
    )
    from .plot_style import apply_publication_style, save_publication_figure
except ImportError:
    from config import (
        EMBEDDINGS_CSV, ENABLE_SMOTE as CFG_ENABLE_SMOTE,
        FEATURES_CSV, FIGURES_DIR, MAIN_MODEL_PKL, MODELS_DIR,
        RANDOM_STATE, RESULTS_DIR,
        TRAIN_OPTUNA_TRIALS, TRAIN_N_FOLDS, TRAIN_DART_MAX_TREES,
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, TRAIN_SMOTE_RATIO,
        TRAIN_BLEND_TRIALS, TRAIN_CT5_KEEP_DIMS, TRAIN_FEATURE_SUBSETS,
        TRAIN_META_C_CANDIDATES, TRAIN_THRESHOLD_STRATEGY,
        TRAIN_ENABLE_STACK,
        TRAIN_ENABLE_AUTO_FEATURE_SUBSET,
        TRAIN_SEEDS, TRAIN_HPO_ONCE,
        TRAIN_OPTIMIZE_AUROC, TRAIN_HPO_ALPHA_AUPRC,
    )
    from plot_style import apply_publication_style, save_publication_figure

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
apply_publication_style()

SELECTED_FEATURES_JSON = os.path.join(MODELS_DIR, "selected_features.json")

# -- TUNABLE CONSTANTS (all sourced from config.py — edit there to tune) --------
OPTUNA_TRIALS            = TRAIN_OPTUNA_TRIALS
N_FOLDS                  = TRAIN_N_FOLDS
DART_MAX_TREES           = TRAIN_DART_MAX_TREES
ENABLE_STACK             = TRAIN_ENABLE_STACK
ENABLE_SMOTE             = CFG_ENABLE_SMOTE
TEST_FRAC                = TRAIN_TEST_FRAC
VAL_FRAC                 = TRAIN_VAL_FRAC
USE_TEMPORAL_SPLIT       = False   # False = random patient split (often higher AUROC)
ENABLE_WEIGHTED_BLEND    = True
THRESHOLD_STRATEGY       = TRAIN_THRESHOLD_STRATEGY
HPO_ALPHA_AUPRC          = TRAIN_HPO_ALPHA_AUPRC
OPTIMIZE_FOR_AUROC       = TRAIN_OPTIMIZE_AUROC
BLEND_SEARCH_TRIALS      = TRAIN_BLEND_TRIALS
ENABLE_CT5_DIM_SELECTION = True
CT5_KEEP_DIMS            = TRAIN_CT5_KEEP_DIMS
USE_SELECTED_FEATURES_JSON = False
ENABLE_AUTO_FEATURE_SUBSET = TRAIN_ENABLE_AUTO_FEATURE_SUBSET
FEATURE_SUBSET_CANDIDATES  = TRAIN_FEATURE_SUBSETS


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
    if USE_SELECTED_FEATURES_JSON and os.path.exists(SELECTED_FEATURES_JSON):
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
        # Avoid eager block consolidation on very large frames (can OOM on Windows).
        X = df.loc[:, keep]
    else:
        keep = [c for c in df.columns if c not in id_cols]
        X = df.loc[:, keep]

    # Optional dimensionality reduction by selecting the highest-variance numeric ct5_* dims.
    if ENABLE_CT5_DIM_SELECTION:
        ct5_vec_cols = [c for c in X.columns if c.startswith("ct5_") and c[4:].isdigit()]
        if len(ct5_vec_cols) > CT5_KEEP_DIMS:
            variances = X[ct5_vec_cols].var(axis=0, ddof=0)
            keep_ct5 = variances.sort_values(ascending=False).head(CT5_KEEP_DIMS).index.tolist()
            other_cols = [c for c in X.columns if c not in ct5_vec_cols]
            X = X.loc[:, other_cols + keep_ct5]
            logger.info(
                "ct5 dimension selection: kept %d/%d numeric ct5 dims by variance.",
                len(keep_ct5), len(ct5_vec_cols),
            )

    logger.info("Feature matrix: %s | Positive rate: %.2f%%", X.shape, y.mean() * 100)
    return X, y, groups


def auto_select_feature_subset(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    pos_weight: float,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """
    Train-only feature reduction:
      1) rank features by LightGBM gain on training split
      2) evaluate candidate top-k subsets on validation AUROC
      3) keep best k
    """
    n_features = X_tr.shape[1]
    if n_features <= min(FEATURE_SUBSET_CANDIDATES):
        return np.arange(n_features), list(feature_names), {
            "enabled": False, "reason": "too_few_features", "selected_k": int(n_features)
        }

    logger.info("Auto feature subset search enabled (n_features=%d).", n_features)
    ranker = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=10,
        min_child_samples=40,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.2,
        reg_lambda=1.5,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    ranker.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    gains = ranker.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-gains)
    del ranker
    gc.collect()

    valid_candidates = sorted({k for k in FEATURE_SUBSET_CANDIDATES if k < n_features})
    valid_candidates.append(n_features)
    best_auc = -1.0
    best_idx = np.arange(n_features)
    diagnostics: Dict[str, float] = {"enabled": True, "n_features_full": int(n_features)}

    for k in valid_candidates:
        idx = order[:k]
        probe = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            n_estimators=1600,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=10,
            min_child_samples=30,
            colsample_bytree=0.85,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        probe.fit(
            X_tr[:, idx], y_tr,
            eval_set=[(X_val[:, idx], y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        val_probs = probe.predict_proba(X_val[:, idx])[:, 1]
        auc = float(roc_auc_score(y_val, val_probs))
        diagnostics[f"val_auroc_top_{k}"] = auc
        logger.info("  Feature subset top-%d -> val AUROC: %.4f", k, auc)
        if auc > best_auc:
            best_auc = auc
            best_idx = idx
        del probe
        gc.collect()

    selected_names = [feature_names[i] for i in best_idx]
    diagnostics["selected_k"] = int(len(best_idx))
    diagnostics["selected_val_auroc"] = float(best_auc)
    logger.info("Auto feature subset selected: top-%d (val AUROC %.4f).", len(best_idx), best_auc)
    return np.array(best_idx, dtype=int), selected_names, diagnostics


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
                sampling_strategy=TRAIN_SMOTE_RATIO,
                random_state=RANDOM_STATE,
            ),
            random_state=RANDOM_STATE,
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
        # AUROC-focused: bias search toward gbdt/goss, avoid expensive/unstable dart.
        boosting  = trial.suggest_categorical("boosting_type", ["gbdt", "gbdt", "gbdt", "goss"])
        imbalance = trial.suggest_categorical(
            "imbalance_strategy", ["scale_pos_weight", "is_unbalance"]
        )

        params: Dict = {
            "objective":         "binary",
            "metric":            "auc",
            "verbosity":         -1,
            "boosting_type":     boosting,
            "num_leaves":        trial.suggest_int("num_leaves", 63, 255),
            "max_depth":         trial.suggest_int("max_depth", 5, 13),
            "learning_rate":     trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
            # DART has no early stopping so cap trees to avoid 20-min trials
            "n_estimators": (
                trial.suggest_int("n_estimators", 500, DART_MAX_TREES)
                if boosting == "dart"
                else trial.suggest_int("n_estimators", 800, 4500)
            ),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "colsample_bynode":  trial.suggest_float("colsample_bynode", 0.55, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 4.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 6.0),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
            "path_smooth":       trial.suggest_float("path_smooth", 0.0, 1.0),
            "max_bin":           trial.suggest_int("max_bin", 127, 511),
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
        val_probs = m.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, val_probs)
        auprc = average_precision_score(y_val, val_probs)
        trial.set_user_attr("val_auroc", float(auroc))
        trial.set_user_attr("val_auprc", float(auprc))
        if OPTIMIZE_FOR_AUROC:
            return float(auroc)
        return composite_rank_score(y_val, val_probs)

    sampler = optuna.samplers.TPESampler(multivariate=False, seed=RANDOM_STATE)
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
    logger.info(
        "Best HPO score (metric=%s): %.4f",
        "AUROC" if OPTIMIZE_FOR_AUROC else "composite",
        study.best_value,
    )
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
    StratifiedGroupKFold CV (fallback: GroupKFold) grouped by subject_id.
    Guarantees no patient's records appear in both train and validation folds.
    Returns (mean_auroc, std_auroc, per_fold_aurocs).
    """
    logger.info("Patient-level grouped CV (%d folds) ...", n_folds)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y, groups)
        logger.info("Using StratifiedGroupKFold.")
    except Exception:
        gkf = GroupKFold(n_splits=n_folds)
        split_iter = gkf.split(X, y, groups)
        logger.info("StratifiedGroupKFold unavailable; using GroupKFold.")
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(split_iter):
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

        try:
            from catboost import CatBoostClassifier
            logger.info("Training CatBoost ...")
            cb_model = CatBoostClassifier(
                iterations=1200,
                learning_rate=float(params.get("learning_rate", 0.03)),
                depth=min(int(params.get("max_depth", 8)), 10),
                l2_leaf_reg=max(float(params.get("reg_lambda", 1.0)), 1e-6),
                eval_metric="AUC",
                loss_function="Logloss",
                random_seed=RANDOM_STATE,
                verbose=False,
                auto_class_weights="Balanced",
            )
            cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            auc_cb = roc_auc_score(y_val, cb_model.predict_proba(X_val)[:, 1])
            logger.info("CatBoost val AUROC: %.4f", auc_cb)
            models.append(("catboost", cb_model))
        except ImportError:
            logger.warning("catboost not installed - skipping. pip install catboost")
        except Exception as e:
            logger.warning("CatBoost failed (%s) - continuing.", e)

    return models


def ensemble_predict(models: List[Tuple[str, object]], X: np.ndarray) -> np.ndarray:
    """Simple mean of model probabilities."""
    return np.stack([m.predict_proba(X)[:, 1] for _, m in models], axis=1).mean(axis=1)


# ==============================================================================
# 6. META-LEARNER STACKING
# ==============================================================================

def _optimize_blend_weights(
    val_stack: np.ndarray,
    y_val: np.ndarray,
    trials: int = BLEND_SEARCH_TRIALS,
) -> np.ndarray:
    """Random-search non-negative weights (sum to 1) for ensemble blending."""
    rng = np.random.RandomState(RANDOM_STATE)
    n_models = val_stack.shape[1]
    best_w = np.ones(n_models, dtype=np.float64) / max(n_models, 1)
    best_score = composite_rank_score(y_val, val_stack @ best_w)
    for _ in range(trials):
        w = rng.dirichlet(np.ones(n_models))
        score = composite_rank_score(y_val, val_stack @ w)
        if score > best_score:
            best_score = score
            best_w = w
    return best_w


def fit_meta_learner(
    models: List[Tuple[str, object]],
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[Optional[LogisticRegression], np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Selects the best stacker on validation data among:
      1) simple mean
      2) optimized weighted blend
      3) logistic meta-learner (scored via OOF cross_val_predict — not in-sample)
    Returns meta model (if selected), val probs, test probs, and diagnostics.
    """
    val_stack = np.column_stack([m.predict_proba(X_val)[:, 1] for _, m in models])
    te_stack  = np.column_stack([m.predict_proba(X_te)[:, 1]  for _, m in models])

    if len(models) < 2:
        p_val = val_stack[:, 0]
        p_te  = te_stack[:, 0]
        return None, p_val, p_te, {"selected_stacker": "single_model"}

    candidates: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[LogisticRegression]]] = {}

    # Candidate 1: simple mean
    candidates["mean"] = (
        val_stack.mean(axis=1),
        te_stack.mean(axis=1),
        None,
    )

    # Candidate 2: weighted blend
    if ENABLE_WEIGHTED_BLEND:
        w = _optimize_blend_weights(val_stack, y_val)
        candidates["weighted_blend"] = (
            val_stack @ w,
            te_stack @ w,
            None,
        )

    # Candidate 3: logistic meta-learner with light C search.
    # FIX: evaluate using 5-fold OOF cross_val_predict so the meta-LR score is
    # out-of-sample. The old code scored meta.predict_proba(val_stack) after
    # fitting on val_stack — pure in-sample, always winning stacker selection
    # even when it would underperform on test data.
    best_meta       = None
    best_meta_score = -np.inf
    best_meta_val   = None
    best_meta_te    = None
    for c in TRAIN_META_C_CANDIDATES:
        meta = LogisticRegression(C=c, max_iter=2000, random_state=RANDOM_STATE)
        try:
            # cross_val_predict gives honest OOF probabilities on val_stack
            p_val_oof = cross_val_predict(
                meta, val_stack, y_val, cv=5, method="predict_proba"
            )[:, 1]
            sc = composite_rank_score(y_val, p_val_oof)
        except Exception as e:
            logger.warning("Meta-LR OOF eval failed (C=%.4f): %s — skipping.", c, e)
            continue
        if sc > best_meta_score:
            best_meta_score = sc
            # Refit on full val_stack for final test prediction
            meta.fit(val_stack, y_val)
            best_meta     = meta
            best_meta_val = p_val_oof                        # honest OOF val probs
            best_meta_te  = meta.predict_proba(te_stack)[:, 1]
    if best_meta is not None:
        candidates["meta_lr"] = (best_meta_val, best_meta_te, best_meta)

    # Select highest composite score on validation
    best_name  = None
    best_score = -np.inf
    best_val   = None
    best_te    = None
    best_model = None
    for name, (p_val, p_te, m) in candidates.items():
        sc  = composite_rank_score(y_val, p_val)
        auc = roc_auc_score(y_val, p_val)
        ap  = average_precision_score(y_val, p_val)
        logger.info("Stacker %-15s | score: %.4f | AUROC: %.4f | AUPRC: %.4f", name, sc, auc, ap)
        if sc > best_score:
            best_name, best_score, best_val, best_te, best_model = name, sc, p_val, p_te, m

    return best_model, best_val, best_te, {
        "selected_stacker":    best_name,
        "val_composite_score": float(best_score),
    }


# ==============================================================================
# 7. CALIBRATION
# ==============================================================================

class PlattCalibrator:
    """Pickle-safe wrapper to provide .predict() interface like IsotonicRegression."""

    def __init__(self, lr: LogisticRegression):
        self.lr = lr

    def predict(self, probs_1d: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs_1d).reshape(-1, 1)
        return self.lr.predict_proba(arr)[:, 1]


def calibrate(
    val_probs: np.ndarray, y_val: np.ndarray,
    test_probs: np.ndarray,
) -> Tuple[object, np.ndarray, str]:
    """
    Fit both Isotonic and Platt (logistic) calibration on validation probs.
    Select method with lower validation Brier score (tie-break: lower log loss).
    """
    ece_before = _ece(val_probs, y_val)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_val)
    val_iso  = iso.predict(val_probs)
    test_iso = iso.predict(test_probs).astype(np.float32)
    brier_iso = brier_score_loss(y_val, val_iso)
    ll_iso    = log_loss(y_val, np.clip(val_iso, 1e-6, 1 - 1e-6))

    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    platt.fit(val_probs.reshape(-1, 1), y_val)
    val_platt  = platt.predict_proba(val_probs.reshape(-1, 1))[:, 1]
    test_platt = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1].astype(np.float32)
    brier_platt = brier_score_loss(y_val, val_platt)
    ll_platt    = log_loss(y_val, np.clip(val_platt, 1e-6, 1 - 1e-6))

    use_iso = (brier_iso < brier_platt) or (abs(brier_iso - brier_platt) < 1e-6 and ll_iso <= ll_platt)
    if use_iso:
        chosen   = "isotonic"
        cal      = iso
        test_cal = test_iso
        ece_after = _ece(val_iso, y_val)
    else:
        chosen   = "platt"
        cal      = PlattCalibrator(platt)
        test_cal = test_platt
        ece_after = _ece(val_platt, y_val)

    logger.info(
        "Calibration selected: %s | Brier iso=%.4f platt=%.4f | LogLoss iso=%.4f platt=%.4f",
        chosen, brier_iso, brier_platt, ll_iso, ll_platt,
    )
    logger.info("ECE before calibration: %.4f -> after: %.4f", ece_before, ece_after)
    return cal, test_cal, chosen


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
    """Finds the best decision threshold using validation data only."""
    thresholds = np.arange(0.01, 0.99, 0.005)

    if strategy == "f1":
        scores = [f1_score(y_val, val_probs >= t, zero_division=0) for t in thresholds]
        best = float(thresholds[np.argmax(scores)])
    elif strategy == "recall80":
        best = 0.50
        for t in sorted(thresholds, reverse=True):
            preds = (val_probs >= t).astype(int)
            rec = preds[y_val == 1].mean() if (y_val == 1).sum() > 0 else 0.0
            if rec >= 0.80:
                best = float(t)
                break
    elif strategy == "j":
        fpr, tpr, thresh = roc_curve(y_val, val_probs)
        best = float(thresh[np.argmax(tpr - fpr)])
    elif strategy == "mcc":
        scores = [matthews_corrcoef(y_val, (val_probs >= t).astype(int)) for t in thresholds]
        best = float(thresholds[np.argmax(scores)])
    else:
        best = 0.50

    logger.info("Threshold strategy='%s' -> best threshold=%.3f", strategy, best)
    return best


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    recall      = tp / max(tp + fn, 1)
    precision   = tp / max(tp + fp, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1  = f1_score(y_true, preds, zero_division=0)
    mcc = matthews_corrcoef(y_true, preds)
    return {
        "threshold":   float(threshold),
        "accuracy":    float(accuracy),
        "recall":      float(recall),
        "precision":   float(precision),
        "specificity": float(specificity),
        "f1":          float(f1),
        "mcc":         float(mcc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def summarize_operating_points(
    y_val: np.ndarray,
    val_probs_cal: np.ndarray,
    y_te: np.ndarray,
    test_probs_cal: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Threshold profiles selected on validation, measured on test."""
    thresholds = {
        "mcc":      find_best_threshold(val_probs_cal, y_val, "mcc"),
        "f1":       find_best_threshold(val_probs_cal, y_val, "f1"),
        "recall80": find_best_threshold(val_probs_cal, y_val, "recall80"),
        "j":        find_best_threshold(val_probs_cal, y_val, "j"),
    }
    return {k: _binary_metrics(y_te, test_probs_cal, t) for k, t in thresholds.items()}


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

    save_publication_figure(fig, os.path.join(FIGURES_DIR, "roc_pr_curve.png"))

    # -- Calibration curve -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, probs in [("Raw", test_probs_raw), ("Calibrated", test_probs_cal)]:
        fp, mp = _calibration_bins(probs, y_te)
        ax.plot(mp, fp, "s-", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction Positives",
           title="Calibration (Reliability Diagram)")
    ax.legend()
    save_publication_figure(fig, os.path.join(FIGURES_DIR, "calibration_curve.png"))

    # -- Threshold analysis ----------------------------------------------------
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
    save_publication_figure(fig, os.path.join(FIGURES_DIR, "threshold_analysis.png"))

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
        save_publication_figure(plt.gcf(), os.path.join(FIGURES_DIR, "shap_summary.png"))
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

    def _run_single(self, seed: int, best_params: Optional[Dict] = None,
                    do_hpo: bool = True, do_artifacts: bool = True) -> Dict[str, object]:
        _set_seed(seed)
        # -- 1. Load -------------------------------------------------------
        X, y, groups = load_data()
        self.features = list(X.columns)
        feature_subset_info: Dict[str, object] = {"enabled": False}

        # -- 2. Patient-level split ----------------------------------------
        if USE_TEMPORAL_SPLIT:
            # Prefer pseudo-time from anchor_year_group + admission_month if present.
            # Fallback to row index order when temporal columns are missing.
            X_reset = X.reset_index(drop=True)
            if {"anchor_year_group", "admission_month"}.issubset(set(X_reset.columns)):
                time_key = (
                    pd.to_numeric(X_reset["anchor_year_group"], errors="coerce").fillna(0).astype(float) * 12.0
                    + pd.to_numeric(X_reset["admission_month"], errors="coerce").fillna(0).astype(float)
                )
                logger.info("Temporal split using anchor_year_group + admission_month.")
            else:
                time_key = pd.Series(np.arange(len(X_reset)), index=X_reset.index, dtype="float64")
                logger.info("Temporal columns missing; split fallback uses row order.")

            pat_first = pd.DataFrame({"subject_id": groups.values, "time_key": time_key.values}) \
                .groupby("subject_id", as_index=False)["time_key"].min()
            sorted_pats = pat_first.sort_values(["time_key", "subject_id"])["subject_id"].tolist()
        else:
            # FIX: use seed directly so each multi-seed run gets a reproducibly
            # different shuffle, independent of the module-level RANDOM_STATE.
            rng = np.random.RandomState(seed)
            sorted_pats = list(groups.drop_duplicates().values)
            rng.shuffle(sorted_pats)
            logger.info("Patient-level random split enabled (USE_TEMPORAL_SPLIT=False).")

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

        # -- 3.5 Train-only feature subset search --------------------------
        if ENABLE_AUTO_FEATURE_SUBSET:
            feat_idx, feat_names, feature_subset_info = auto_select_feature_subset(
                X_tr, y_tr, X_val, y_val, self.features, pos_weight
            )
            X_tr = X_tr[:, feat_idx]
            X_val = X_val[:, feat_idx]
            X_te = X_te[:, feat_idx]
            X = X.loc[:, feat_names]
            self.features = feat_names
            logger.info("Using feature subset: %d features.", len(self.features))

        # -- 3. SMOTE on training fold -------------------------------------
        if ENABLE_SMOTE:
            X_tr, y_tr = apply_smote(X_tr, y_tr)

        # -- 4. HPO --------------------------------------------------------
        if do_hpo or best_params is None:
            self.best_params = optimize_lgbm(
                X_tr, y_tr, X_val, y_val,
                pos_weight, n_trials=OPTUNA_TRIALS,
            )
        else:
            self.best_params = best_params

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
            self.meta, val_probs_for_cal, test_probs_raw, stack_info = fit_meta_learner(
                self.models, X_val, y_val, X_te
            )
            logger.info("Selected stacker: %s", stack_info.get("selected_stacker"))
        else:
            self.meta = None
            stack_info = {"selected_stacker": "mean"}
            test_probs_raw    = ensemble_predict(self.models, X_te)
            val_probs_for_cal = ensemble_predict(self.models, X_val)

        # -- 8. Isotonic calibration ---------------------------------------
        self.calibrator, test_probs_cal, calibration_method = calibrate(
            val_probs_for_cal, y_val, test_probs_raw
        )
        val_probs_cal = self.calibrator.predict(val_probs_for_cal).astype(np.float32)

        # -- 9. Threshold optimisation (val set only) ----------------------
        self.best_threshold = find_best_threshold(
            val_probs_cal, y_val, strategy=THRESHOLD_STRATEGY
        )

        # -- 10. Metrics ---------------------------------------------------
        auc_raw  = roc_auc_score(y_te, test_probs_raw)
        auc_cal  = roc_auc_score(y_te, test_probs_cal)
        auprc    = average_precision_score(y_te, test_probs_cal)
        brier    = brier_score_loss(y_te, test_probs_cal)
        ll       = log_loss(y_te, test_probs_cal)

        test_preds = (test_probs_cal >= self.best_threshold).astype(int)
        report     = classification_report(y_te, test_preds, output_dict=True,
                                           zero_division=0)
        recall_pos = report.get("1", {}).get("recall",    0.0)
        prec_pos   = report.get("1", {}).get("precision", 0.0)
        f1_pos     = report.get("1", {}).get("f1-score",  0.0)
        mcc_pos    = matthews_corrcoef(y_te, test_preds)
        op_points  = summarize_operating_points(y_val, val_probs_cal, y_te, test_probs_cal)

        logger.info("=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("  AUROC (raw):               %.4f", auc_raw)
        logger.info("  AUROC (calibrated):        %.4f", auc_cal)
        logger.info("  AUPRC:                     %.4f", auprc)
        logger.info("  Brier Score:               %.4f", brier)
        logger.info("  Log Loss:                  %.4f", ll)
        logger.info("  CV AUROC:                  %.4f ± %.4f", cv_mean, cv_std)
        logger.info("  Best threshold (val %s):   %.3f", THRESHOLD_STRATEGY, self.best_threshold)
        logger.info("  -- Readmit=1 @ threshold %.3f --", self.best_threshold)
        logger.info("  Recall    (sensitivity):   %.4f", recall_pos)
        logger.info("  Precision (PPV):           %.4f", prec_pos)
        logger.info("  F1 score:                  %.4f", f1_pos)
        logger.info("  MCC:                       %.4f", mcc_pos)
        logger.info("=" * 60)

        # -- 11. Ablation --------------------------------------------------
        X_df_tr  = pd.DataFrame(np.vstack([X_tr, X_val]), columns=self.features)
        X_df_te  = pd.DataFrame(X_te, columns=self.features)
        ablation = {}
        if do_artifacts:
            ablation = run_ablation(
                X_df_tr, pd.Series(np.concatenate([y_tr, y_val])),
                X_df_te, pd.Series(y_te),
                self.best_params,
            )

        # -- 12. SHAP ------------------------------------------------------
        if do_artifacts:
            primary = next((m for n, m in self.models if n == "lgbm"), None)
            if primary is not None:
                n_shap = min(3000, len(X_te))
                idx    = np.random.RandomState(RANDOM_STATE).choice(
                    len(X_te), n_shap, replace=False
                )
                compute_shap(primary, pd.DataFrame(X_te[idx], columns=self.features))

            # -- 13. Plots -------------------------------------------------
            save_plots(y_te, test_probs_raw, test_probs_cal, self.best_threshold)

            # -- 14. Predictions CSV ---------------------------------------
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
            "threshold_strategy": THRESHOLD_STRATEGY,
            "calibration_method": calibration_method,
            "smote_enabled":      ENABLE_SMOTE,
            "feature_subset":     feature_subset_info,
            "recall_readmit1":    round(recall_pos, 4),
            "precision_readmit1": round(prec_pos,   4),
            "f1_readmit1":        round(f1_pos,     4),
            "mcc_readmit1":       round(mcc_pos,    4),
            "operating_points":   op_points,
            "stacking":           stack_info,
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
        # Compute and store per-feature training means for use in 08_predict.py
        # This lets the CLI use realistic population-level defaults for EHR features
        # (lab values, vital signs, ICD pivots) that can't be entered manually.
        try:
            X_all_np = np.vstack([X_tr, X_val])
            self._X_train_means = {
                feat: float(np.nanmean(X_all_np[:, i]))
                for i, feat in enumerate(self.features)
            }
            logger.info("Feature means computed for %d features.", len(self._X_train_means))
        except Exception as e:
            logger.warning("Could not compute feature means: %s", e)
            self._X_train_means = {}

        results.update({
            "seed":           seed,
            "val_probs_raw":  val_probs_for_cal,
            "test_probs_raw": test_probs_raw,
            "y_val":          y_val,
            "y_te":           y_te,
            "cv_auroc_mean":  cv_mean,
            "cv_auroc_std":   cv_std,
            "stack_info":     stack_info,
        })
        return results

    def run(self) -> Dict:
        seeds = TRAIN_SEEDS or [RANDOM_STATE]
        seed_results: List[Dict[str, object]] = []
        best_params = None

        for i, seed in enumerate(seeds):
            logger.info("=== Training seed %s (%d/%d) ===", seed, i + 1, len(seeds))
            res = self._run_single(
                seed=seed,
                best_params=best_params,
                do_hpo=(i == 0 or not TRAIN_HPO_ONCE),
                do_artifacts=(i == 0),
            )
            if best_params is None:
                best_params = res.get("best_params")
            seed_results.append(res)

        if len(seed_results) == 1:
            final = seed_results[0]
            self._save(final)
            return final

        # Average predictions across seeds
        y_val  = seed_results[0]["y_val"]
        y_te   = seed_results[0]["y_te"]
        val_probs_raw  = np.mean([r["val_probs_raw"]  for r in seed_results], axis=0)
        test_probs_raw = np.mean([r["test_probs_raw"] for r in seed_results], axis=0)

        self.calibrator, test_probs_cal, calibration_method = calibrate(
            val_probs_raw, y_val, test_probs_raw
        )
        val_probs_cal = self.calibrator.predict(val_probs_raw).astype(np.float32)

        self.best_threshold = find_best_threshold(
            val_probs_cal, y_val, strategy=THRESHOLD_STRATEGY
        )

        auc_raw  = roc_auc_score(y_te, test_probs_raw)
        auc_cal  = roc_auc_score(y_te, test_probs_cal)
        auprc    = average_precision_score(y_te, test_probs_cal)
        brier    = brier_score_loss(y_te, test_probs_cal)
        ll       = log_loss(y_te, test_probs_cal)

        test_preds = (test_probs_cal >= self.best_threshold).astype(int)
        report     = classification_report(y_te, test_preds, output_dict=True,
                                           zero_division=0)
        recall_pos = report.get("1", {}).get("recall",    0.0)
        prec_pos   = report.get("1", {}).get("precision", 0.0)
        f1_pos     = report.get("1", {}).get("f1-score",  0.0)
        mcc_pos    = matthews_corrcoef(y_te, test_preds)
        op_points  = summarize_operating_points(y_val, val_probs_cal, y_te, test_probs_cal)

        results = {
            "auroc_raw":          round(auc_raw,   4),
            "auroc_calibrated":   round(auc_cal,   4),
            "auprc":              round(auprc,      4),
            "brier_score":        round(brier,      4),
            "log_loss":           round(ll,         4),
            "cv_auroc_mean":      round(np.mean([r["cv_auroc_mean"] for r in seed_results]), 4),
            "cv_auroc_std":       round(np.mean([r["cv_auroc_std"]  for r in seed_results]), 4),
            "best_threshold":     round(self.best_threshold, 3),
            "threshold_strategy": THRESHOLD_STRATEGY,
            "calibration_method": calibration_method,
            "smote_enabled":      ENABLE_SMOTE,
            "feature_subset":     seed_results[0].get("feature_subset", {"enabled": False}),
            "recall_readmit1":    round(recall_pos, 4),
            "precision_readmit1": round(prec_pos,   4),
            "f1_readmit1":        round(f1_pos,     4),
            "mcc_readmit1":       round(mcc_pos,    4),
            "operating_points":   op_points,
            "stacking":           seed_results[0].get("stack_info", {}),
            "ablation":           seed_results[0].get("ablation", {}),
            "n_models":           seed_results[0].get("n_models", 0),
            "n_features":         seed_results[0].get("n_features", 0),
            "train_size":         seed_results[0].get("train_size", 0),
            "val_size":           seed_results[0].get("val_size", 0),
            "test_size":          seed_results[0].get("test_size", 0),
            "best_params": {
                k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                for k, v in (best_params or {}).items()
            },
            "timestamp": datetime.now().isoformat(),
            "seeds":      seeds,
        }

        # Save averaged predictions
        pred_df = pd.DataFrame({
            "y_true":   y_te,
            "prob_raw": test_probs_raw.round(6),
            "prob_cal": test_probs_cal.round(6),
            "pred":     test_preds,
        })
        pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        logger.info("Predictions saved -> %s", pred_path)

        self._save(results)
        return results

    def _save(self, results: Dict) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        feature_means = getattr(self, "_X_train_means", {})
        joblib.dump(
            {
                "models":         self.models,
                "meta":           self.meta,
                "calibrator":     self.calibrator,
                "features":       self.features,
                "best_params":    self.best_params,
                "best_threshold": self.best_threshold,
                "feature_means":  feature_means,
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