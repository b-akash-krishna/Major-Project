# src/05_analyze.py
"""
SHAP Interpretability Analysis for TRANCE Framework
Refactored for modularity and configuration-driven execution.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import logging
from .config import (
    MAIN_MODEL_PKL, FEATURES_CSV, EMBEDDINGS_CSV, 
    FIGURES_DIR, RESULTS_DIR, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    def __init__(self):
        self.model_data = None
        self.df = None
        self.explainer = None
        
    def load_resources(self):
        """Loads model and fuzed data"""
        if not os.path.exists(MAIN_MODEL_PKL):
            raise FileNotFoundError("Model not found. Train first.")
            
        self.model_data = joblib.load(MAIN_MODEL_PKL)
        
        # Load data
        tabular = pd.read_csv(FEATURES_CSV)
        if os.path.exists(EMBEDDINGS_CSV):
            embs = pd.read_csv(EMBEDDINGS_CSV)
            self.df = tabular.merge(embs, on="hadm_id", how="left").fillna(0)
        else:
            self.df = tabular.fillna(0)
            
        return self.df

    def analyze(self, n_samples=500):
        """Runs SHAP analysis on a subsample of data"""
        logger.info(f"Computing SHAP values for {n_samples} samples...")
        
        features = self.model_data['features']
        X = self.df[features]
        
        # Subsample for speed
        X_sample = X.sample(n=min(n_samples, len(X)), random_state=RANDOM_STATE)
        
        self.explainer = shap.TreeExplainer(self.model_data['model'])
        shap_values = self.explainer.shap_values(X_sample)
        
        # For LightGBM, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        return shap_values, X_sample

    def generate_plots(self, shap_values, X_sample):
        """Generates standard interpretability plots"""
        logger.info("Generating SHAP plots...")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # 1. Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        importance = np.abs(shap_values).mean(axis=0)
        feat_imp = pd.DataFrame({'feature': X_sample.columns, 'importance': importance})
        feat_imp = feat_imp.sort_values('importance', ascending=False)
        
        feat_imp.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
        logger.info(f"Saved feature importance report to {RESULTS_DIR}")

    def run(self):
        self.load_resources()
        shap_vals, X_sample = self.analyze()
        self.generate_plots(shap_vals, X_sample)
        logger.info("SHAP Analysis complete.")

if __name__ == "__main__":
    analyzer = SHAPAnalyzer()
    analyzer.run()
