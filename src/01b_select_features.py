# src/01b_select_features.py
"""
TRANCE Framework - Feature Selection v2
Multi-method: SHAP + gain importance + mutual information (rank aggregation).
Unchanged from working version — this already works fine.
"""

import gc
import json
import logging
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from .config import DATA_DIR, FEATURES_CSV, MODELS_DIR
except ImportError:
    from config import DATA_DIR, FEATURES_CSV, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SELECTED_FEATURES_JSON = os.path.join(MODELS_DIR, "selected_features.json")
PRUNED_CSV = FEATURES_CSV.replace(".csv", "_pruned.csv")


def select_features(top_n: int = 160) -> bool:
    if not os.path.exists(FEATURES_CSV):
        logger.error("Features file not found: %s", FEATURES_CSV)
        return False

    logger.info("Loading features from %s ...", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV, low_memory=False)

    id_cols = ["subject_id", "hadm_id", "readmit_30"]
    X = df.drop(columns=id_cols, errors="ignore").fillna(0)
    y = df["readmit_30"].astype("int8")

    var = X.var()
    X = X.loc[:, var > 1e-8]
    logger.info("Features after zero-variance removal: %d", X.shape[1])

    X = _remove_correlated(X, threshold=0.97)
    logger.info("Features after correlation pruning: %d", X.shape[1])

    n_folds = 3
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    shap_importance = np.zeros(X.shape[1])
    gain_importance = np.zeros(X.shape[1])

    lgb_params = {
        "objective": "binary", "metric": "auc",
        "num_leaves": 63, "max_depth": 8, "learning_rate": 0.05,
        "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": float((y == 0).sum() / max((y == 1).sum(), 1)),
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("  Feature selection fold %d/%d ...", fold + 1, n_folds)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        try:
            exp = shap.TreeExplainer(model)
            sv = exp.shap_values(X_val)
            if isinstance(sv, list): sv = sv[1]
            shap_importance += np.abs(sv).mean(axis=0)
        except Exception as e:
            logger.warning("SHAP fold %d: %s", fold+1, e)
        gain_importance += model.booster_.feature_importance(importance_type="gain")
        del X_tr, X_val, y_tr, y_val, model; gc.collect()

    shap_importance /= n_folds
    gain_importance /= n_folds

    logger.info("Computing mutual information ...")
    idx = np.random.RandomState(42).choice(len(X), min(50_000, len(X)), replace=False)
    try:
        mi_importance = mutual_info_classif(X.iloc[idx].values, y.iloc[idx].values,
                                            discrete_features=False, random_state=42)
    except Exception as e:
        logger.warning("MI failed: %s", e)
        mi_importance = np.zeros(X.shape[1])

    def rank_norm(arr):
        r = pd.Series(arr).rank(ascending=True)
        return r / r.max()

    combined_score = (
        rank_norm(shap_importance) * 0.5 +
        rank_norm(gain_importance) * 0.3 +
        rank_norm(mi_importance) * 0.2
    )

    feat_df = pd.DataFrame({
        "feature": X.columns,
        "combined_score": combined_score.values,
        "shap_importance": shap_importance,
        "gain_importance": gain_importance,
        "mi_importance": mi_importance,
    }).sort_values("combined_score", ascending=False).reset_index(drop=True)

    selected = feat_df.head(top_n)["feature"].tolist()
    logger.info("Selected %d features. Top 10: %s", len(selected), selected[:10])

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(SELECTED_FEATURES_JSON, "w") as f:
        json.dump(selected, f, indent=2)
    feat_df.to_csv(os.path.join(MODELS_DIR, "feature_importance_report.csv"), index=False)

    save_cols = ["subject_id", "hadm_id", "readmit_30"] + selected
    available = [c for c in save_cols if c in df.columns]
    df[available].to_csv(PRUNED_CSV, index=False)
    logger.info("Pruned features saved → %s (%d cols)", PRUNED_CSV, len(available))
    return True


def _remove_correlated(X: pd.DataFrame, threshold: float = 0.97) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    if to_drop:
        logger.info("Dropping %d highly correlated features.", len(to_drop))
        X = X.drop(columns=to_drop)
    return X


if __name__ == "__main__":
    select_features()