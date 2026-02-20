# src/03_train.py
"""
TRANCE Framework - Enhanced Training Module
Refactored for modularity, experiment tracking, and code quality.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import logging
import json
import matplotlib.pyplot as plt
import shap
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import (
    FEATURES_CSV, EMBEDDINGS_CSV, MAIN_MODEL_PKL, RESULTS_DIR, 
    FIGURES_DIR, RANDOM_STATE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TRANCETrainer:
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings
        self.model = None
        self.best_params = None
        self.df = None
        self.features = []
        
        # Ensure directories
        for d in [RESULTS_DIR, FIGURES_DIR, os.path.dirname(MAIN_MODEL_PKL)]:
            os.makedirs(d, exist_ok=True)

    def load_and_prepare_data(self):
        """Loads and merges tabular features with embeddings"""
        logger.info("Loading features and embeddings...")
        
        # Check for pruned features first (for maximum performance)
        pruned_path = FEATURES_CSV.replace(".csv", "_pruned.csv")
        active_features_path = pruned_path if os.path.exists(pruned_path) else FEATURES_CSV
        
        if os.path.exists(pruned_path):
            logger.info(f"Using Pruned Holistic Feature Set: {pruned_path}")
        else:
            logger.info(f"Using Raw Holistic Feature Set: {FEATURES_CSV}")
            
        tabular = pd.read_csv(active_features_path)
        
        if self.use_embeddings and os.path.exists(EMBEDDINGS_CSV):
            embs = pd.read_csv(EMBEDDINGS_CSV)
            self.df = tabular.merge(embs, on="hadm_id", how="left").fillna(0)
            logger.info(f"Fused data shape: {self.df.shape}")
        else:
            self.df = tabular.fillna(0)
            logger.info(f"Tabular-only data shape: {self.df.shape}")

        # Define feature columns
        id_cols = ["subject_id", "hadm_id"]
        target_col = "readmit_30"
        self.features = [c for c in self.df.columns if c not in id_cols + [target_col]]
        
        return self.df[self.features], self.df[target_col]

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Runs Optuna study for LightGBM"""
        logger.info(f"Starting hyperparameter optimization ({n_trials} trials)...")
        
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'num_leaves': trial.suggest_int('num_leaves', 31, 150),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'n_estimators': 1000,
                'scale_pos_weight': pos_weight,
                'random_state': RANDOM_STATE
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_params.update({'objective': 'binary', 'metric': 'auc', 'scale_pos_weight': pos_weight, 'random_state': RANDOM_STATE})
        logger.info(f"Best AUROC: {study.best_value:.4f}")
        return self.best_params

    def train_and_calibrate(self, X_train, y_train, X_val, y_val):
        """Trains final model and performs calibration"""
        logger.info("Training final model and calibrating probabilities...")
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # Calibrate
        val_probs = self.model.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip').fit(val_probs, y_val)
        platt = LogisticRegression().fit(val_probs.reshape(-1, 1), y_val)
        
        return iso, platt

    def save_artifacts(self, iso, platt, test_metrics):
        """Saves model and training report"""
        logger.info("Saving model artifacts...")
        
        # Save model pack
        joblib.dump({
            'model': self.model,
            'isotonic_scaler': iso,
            'platt_scaler': platt,
            'features': self.features,
            'timestamp': datetime.now().isoformat()
        }, MAIN_MODEL_PKL)
        
        # Save report
        report_path = os.path.join(RESULTS_DIR, "training_report.json")
        with open(report_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
            
        logger.info(f"Artifacts saved to {MAIN_MODEL_PKL}")

    def run_pipeline(self):
        """Execute full training workflow"""
        X, y = self.load_and_prepare_data()
        
        # Splits
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, stratify=y_train_val, random_state=RANDOM_STATE)
        
        self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        iso, platt = self.train_and_calibrate(X_train, y_train, X_val, y_val)
        
        # Final Evaluation
        test_probs = self.model.predict_proba(X_test)[:, 1]
        calibrated_probs = iso.transform(test_probs)
        auc = roc_auc_score(y_test, calibrated_probs)
        
        logger.info(f"Final Test AUROC: {auc:.4f}")
        
        metrics = {
            "test_auroc": auc,
            "best_params": self.best_params,
            "timestamp": datetime.now().isoformat()
        }
        
        self.save_artifacts(iso, platt, metrics)
        return metrics

if __name__ == "__main__":
    trainer = TRANCETrainer()
    trainer.run_pipeline()