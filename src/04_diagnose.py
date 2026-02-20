# src/04_diagnose.py
"""
Diagnostic script to check Clinical-T5 embeddings quality
Refactored to use centralized configuration.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import logging
from .config import (
    EMBEDDINGS_CSV, FEATURES_CSV, FIGURES_DIR, RESULTS_DIR
)
from .embedding_utils import validate_embeddings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_diagnostics():
    logger.info("Starting Clinical Embedding Diagnostics...")
    
    # 1. Load Data
    if not os.path.exists(EMBEDDINGS_CSV) or not os.path.exists(FEATURES_CSV):
        logger.error("Missing data files. Run extraction and embedding first.")
        return

    emb_df = pd.read_csv(EMBEDDINGS_CSV)
    feat_df = pd.read_csv(FEATURES_CSV)
    
    # 2. Alignment
    merged = feat_df[['hadm_id', 'readmit_30']].merge(emb_df, on='hadm_id', how='inner')
    logger.info(f"Analyzed {len(merged)} matched samples")
    
    text_cols = [c for c in emb_df.columns if c.startswith('ct5_')]
    X = merged[text_cols].values
    y = merged['readmit_30'].values
    
    # 3. Quality Checks
    is_valid, issues, metrics = validate_embeddings(X, y)
    
    # 4. Visualizations
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA
    pca = PCA(n_components=10)
    pca.fit(X)
    axes[0].bar(range(1, 11), pca.explained_variance_ratio_)
    axes[0].set_title('PCA Explained Variance (Text Embeddings)')
    
    # Means by class
    axes[1].scatter(range(len(text_cols)), X[y==0].mean(axis=0), alpha=0.5, label='Class 0')
    axes[1].scatter(range(len(text_cols)), X[y==1].mean(axis=0), alpha=0.5, label='Class 1')
    axes[1].legend()
    axes[1].set_title('Mean Embedding Values by Class')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'embedding_diagnostics_modular.png'))
    
    # 5. Save Report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "is_valid": is_valid,
        "issues": issues,
        "metrics": metrics
    }
    with open(os.path.join(RESULTS_DIR, "embedding_diagnostics.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Diagnostics complete. Valid: {is_valid}")

if __name__ == "__main__":
    run_diagnostics()
