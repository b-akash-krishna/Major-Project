# src/01b_select_features.py
"""
TRANCE Framework - Automated Feature Selection
Uses SHAP and LightGBM to identify and prune low-contribution features.
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
import logging
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, FEATURES_CSV, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURE_SELECTION_JSON = os.path.join(MODELS_DIR, "selected_features.json")

def select_features(top_n=150):
    """Identifies top N features using SHAP importance"""
    if not os.path.exists(FEATURES_CSV):
        logger.error(f"Features file not found: {FEATURES_CSV}")
        return False
        
    logger.info(f"Loading features from {FEATURES_CSV}...")
    df = pd.read_csv(FEATURES_CSV)
    
    # Separate identifiers and target
    drop_cols = ["subject_id", "hadm_id", "readmit_30"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["readmit_30"]
    
    # Train-test split for importance evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training baseline LightGBM on {X_train.shape[1]} features...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        importance_type='gain',
        verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    logger.info("Calculating SHAP values for feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # SHAP values for binary classification can be a list [class0, class1] 
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[1]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
        
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['feature', 'importance'])
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    
    selected_features = feature_importance.head(top_n)['feature'].tolist()
    
    logger.info(f"Selected Top {len(selected_features)} features.")
    
    # Save selected features
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(FEATURE_SELECTION_JSON, 'w') as f:
        json.dump(selected_features, f, indent=4)
        
    logger.info(f"Selected feature list saved to {FEATURE_SELECTION_JSON}")
    
    # Export top N features file for training
    df_pruned = df[drop_cols + selected_features]
    PRUNED_CSV = FEATURES_CSV.replace(".csv", "_pruned.csv")
    df_pruned.to_csv(PRUNED_CSV, index=False)
    logger.info(f"Pruned feature set saved to {PRUNED_CSV}")
    
    return True

if __name__ == "__main__":
    select_features()
