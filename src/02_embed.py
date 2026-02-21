# src/02_embed.py
"""
Clinical Embedding Generation Pipeline
Uses EmbeddingGenerator and Embedding Validator for industrial throughput.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
try:
    from .config import (
        FEATURES_CSV, EMBEDDINGS_CSV, EMBEDDING_INFO_PKL, 
        MIMIC_NOTE_DIR, MIMIC_BHC_DIR, EMBEDDING_DIM, RANDOM_STATE
    )
    from .embedding_utils import EmbeddingGenerator, validate_embeddings
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import (
        FEATURES_CSV, EMBEDDINGS_CSV, EMBEDDING_INFO_PKL, 
        MIMIC_NOTE_DIR, MIMIC_BHC_DIR, EMBEDDING_DIM, RANDOM_STATE
    )
    from embedding_utils import EmbeddingGenerator, validate_embeddings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, generator: EmbeddingGenerator):
        self.generator = generator

    def load_data(self):
        """Loads features and matches with clinical notes"""
        if not os.path.exists(FEATURES_CSV):
            raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}")
            
        df = pd.read_csv(FEATURES_CSV)
        logger.info(f"Loaded {len(df)} tabular feature samples")

        cohort_hadm = set(df['hadm_id'].astype(int).tolist())
        file_index = {}
        for base_dir in [MIMIC_NOTE_DIR, MIMIC_BHC_DIR]:
            if not os.path.isdir(base_dir): continue
            for root, _, files in os.walk(base_dir):
                for name in files:
                    file_index[name] = os.path.join(root, name)

        notes_df = pd.DataFrame({'hadm_id': list(cohort_hadm)})
        notes_df['text'] = ""
        note_to_hadm = pd.DataFrame(columns=['note_id', 'hadm_id'])

        discharge_path = file_index.get("discharge.csv.gz")
        if discharge_path:
            logger.info("Loading discharge notes...")
            try:
                d = pd.read_csv(discharge_path, usecols=['hadm_id', 'text', 'note_id'], nrows=1_000_000).dropna(subset=['hadm_id', 'text'])
                d = d[d['hadm_id'].isin(cohort_hadm)]
                d_agg = d.groupby('hadm_id')['text'].apply(lambda x: " [DISCHARGE] ".join(x.astype(str))).reset_index()
                notes_df = notes_df.merge(d_agg, on='hadm_id', how='left')
                notes_df['text'] = notes_df['text'] + " " + notes_df['text_y'].fillna("")
                notes_df = notes_df.drop(columns=['text_y'])
                note_to_hadm = d[['note_id', 'hadm_id']].dropna().drop_duplicates()
            except Exception as e:
                logger.error(f"Failed to load discharge notes: {e}")

        rad_path = file_index.get("radiology.csv.gz")
        if rad_path:
            logger.info("Loading radiology notes...")
            try:
                r = pd.read_csv(rad_path, usecols=['hadm_id', 'text'], nrows=1_000_000).dropna()
                r = r[r['hadm_id'].isin(cohort_hadm)]
                r_agg = r.groupby('hadm_id')['text'].apply(lambda x: " [RADIOLOGY] ".join(x.astype(str))).reset_index()
                notes_df = notes_df.merge(r_agg, on='hadm_id', how='left')
                notes_df['text'] = notes_df['text'] + " " + notes_df['text_y'].fillna("")
                notes_df = notes_df.drop(columns=['text_y'])
            except Exception as e:
                logger.error(f"Failed to load radiology notes: {e}")

        bhc_path = file_index.get("mimic-iv-bhc.csv")
        if bhc_path and not note_to_hadm.empty:
            logger.info("Loading BHC notes...")
            try:
                bhc = pd.read_csv(bhc_path, usecols=["note_id", "input_tokens", "target_tokens"], nrows=500_000).dropna(subset=["note_id"])
                bhc = bhc.merge(note_to_hadm, on='note_id', how='inner')
                bhc = bhc[bhc['hadm_id'].isin(cohort_hadm)]
                bhc['bhc_text'] = " [BHC INPUT] " + bhc['input_tokens'].astype(str).fillna("") + " [BHC TARGET] " + bhc['target_tokens'].astype(str).fillna("")
                bhc_agg = bhc.groupby('hadm_id')['bhc_text'].apply(lambda x: " ".join(x)).reset_index()
                notes_df = notes_df.merge(bhc_agg, on='hadm_id', how='left')
                notes_df['text'] = notes_df['text'] + " " + notes_df['bhc_text'].fillna("")
                notes_df = notes_df.drop(columns=['bhc_text'])
            except Exception as e:
                logger.error(f"Failed to load BHC notes: {e}")

        notes_df['text'] = notes_df['text'].str.strip()
        notes_df = notes_df[notes_df['text'] != ""]
        
        logger.info(f"Matched {len(notes_df)} admissions with combined clinical notes")
        
        if notes_df.empty:
            return df, None
        return df, notes_df

    def run_text_generation(self, df, notes_df):
        """Generates text-based embeddings with PCA"""
        logger.info("Starting Text-based Embedding Pipeline...")
        notes_merged = df[['hadm_id']].merge(notes_df, on='hadm_id', how='left')
        notes_merged['text'] = notes_merged['text'].fillna("Patient admission documented.")
        
        raw_embs = self.generator.embed_text(notes_merged['text'].tolist())
        
        # PCA reduction
        pca = PCA(n_components=EMBEDDING_DIM, random_state=RANDOM_STATE)
        reduced = pca.fit_transform(raw_embs)
        
        return reduced, notes_merged['hadm_id'].tolist(), pca

    def run_feature_generation(self, df):
        """Generates feature-based embeddings as fallback"""
        logger.info("Starting Feature-based Embedding Pipeline...")
        
        # Select numeric features only
        X = df.select_dtypes(include=[np.number]).drop(columns=['hadm_id', 'subject_id', 'readmit_30'], errors='ignore').fillna(0)
        
        scaler = StandardScaler()
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        pca = PCA(n_components=EMBEDDING_DIM, random_state=RANDOM_STATE)
        
        X_scaled = scaler.fit_transform(X)
        # Note: Poly features can be huge, in production we might use specific interactions
        # For small-ish feature sets, we use interaction_only=True
        X_reduced = pca.fit_transform(X_scaled)
        
        return X_reduced, df['hadm_id'].tolist(), scaler, poly, pca

    def execute(self):
        """Main execution flow"""
        df, notes_df = self.load_data()
        
        method = "zero"
        model_info = {}
        
        if notes_df is not None:
            embeddings, hadm_ids, pca = self.run_text_generation(df, notes_df)
            is_valid, _, _ = validate_embeddings(embeddings, df['readmit_30'].values)
            
            if is_valid:
                method = "text_model"
                model_info = {"method": method, "pca": pca, "reduced_dim": EMBEDDING_DIM}
            else:
                logger.warning("Text embeddings failed validation. Falling back...")
                
        if method == "zero":
            embeddings, hadm_ids, scaler, poly, pca = self.run_feature_generation(df)
            method = "feature_based"
            model_info = {
                "method": method, "scaler": scaler, "poly": poly, 
                "pca": pca, "reduced_dim": EMBEDDING_DIM
            }

        # Save results
        emb_df = pd.DataFrame(embeddings, columns=[f"ct5_{i}" for i in range(EMBEDDING_DIM)])
        emb_df['hadm_id'] = hadm_ids
        emb_df.to_csv(EMBEDDINGS_CSV, index=False)
        joblib.dump(model_info, EMBEDDING_INFO_PKL)
        
        logger.info(f"Pipeline complete using {method}. Saved to {EMBEDDINGS_CSV}")

if __name__ == "__main__":
    gen = EmbeddingGenerator()
    pipeline = EmbeddingPipeline(gen)
    pipeline.execute()