# src/06_visualize.py
"""
Journal-Grade Visualization Suite
Refactored to use centralized configuration and modular data loading.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import logging
from .config import (
    FIGURES_DIR, RESULTS_DIR, MAIN_MODEL_PKL, FEATURES_CSV
)

# Style setup
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JournalVisualizer:
    def __init__(self):
        self.journal_dir = os.path.join(FIGURES_DIR, "journal")
        os.makedirs(self.journal_dir, exist_ok=True)
        self.model_data = None
        
    def load_data(self):
        if os.path.exists(MAIN_MODEL_PKL):
            self.model_data = joblib.load(MAIN_MODEL_PKL)
            
    def plot_top_features(self):
        if not self.model_data: return
        
        logger.info("Generating feature importance plot...")
        model = self.model_data['model']
        feats = self.model_data['features']
        
        imp = pd.DataFrame({'feature': feats, 'importance': model.feature_importances_})
        top = imp.sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, x='importance', y='feature', palette='viridis')
        plt.title("Top 15 Predictive Clinical Features")
        plt.tight_layout()
        plt.savefig(os.path.join(self.journal_dir, "fig_feature_importance.png"), dpi=300)
        plt.close()

    def run(self):
        self.load_data()
        self.plot_top_features()
        logger.info(f"Journal visualizations saved to {self.journal_dir}")

if __name__ == "__main__":
    visualizer = JournalVisualizer()
    visualizer.run()