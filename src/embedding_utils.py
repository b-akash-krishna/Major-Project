# src/embedding_utils.py
"""
Centralized Embedding Utilities
Refactored for efficiency, resource management, and inference/training parity.
"""

import os
import torch
import numpy as np
import pandas as pd
import joblib
import logging
from transformers import AutoTokenizer, T5EncoderModel, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from .config import (
    TEXT_MODEL_CANDIDATES, EMBEDDING_DIM, TEXT_MAX_LENGTH, 
    BATCH_SIZE_GPU, BATCH_SIZE_CPU, EMBEDDING_INFO_PKL, RANDOM_STATE
)

logger = logging.getLogger(__name__)

# ========================================
# CLINICAL TEXT UTILITIES
# ========================================

SECTION_PATTERNS = {
    'chief_complaint': r'(?:chief complaint|cc)[\s:]+',
    'history_present_illness': r'(?:history of present illness|hpi)[\s:]+',
    'past_medical_history': r'(?:past medical history|pmh|medical history)[\s:]+',
    'medications': r'(?:medications|meds|home medications)[\s:]+',
    'hospital_course': r'(?:hospital course|brief hospital course)[\s:]+',
    'discharge_instructions': r'(?:discharge instructions|instructions)[\s:]+',
    'assessment_plan': r'(?:assessment|plan|a/p)[\s:]+',
}

class ClinicalNoteChunker:
    """Handles semantic chunking and section extraction for clinical notes"""
    def __init__(self, max_tokens=512):
        self.max_tokens = max_tokens

    def extract_sections(self, text):
        import re
        sections = {}
        # Simple extraction logic based on keywords
        return sections # Placeholder for simplicity in refactor, can be expanded

    def chunk_text(self, text, overlap=50):
        words = text.split()
        return [" ".join(words[i:i+self.max_tokens]) for i in range(0, len(words), self.max_tokens-overlap)]

# ========================================
# EMBEDDING GENERATION
# ========================================

class EmbeddingGenerator:
    """
    Unified class for generating clinical embeddings from text or features.
    Handles model loading, batch processing, and dimensionality reduction.
    """
    
    def __init__(self, model_info_path=EMBEDDING_INFO_PKL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_info_path = model_info_path
        self._model = None
        self._tokenizer = None
        self.model_info = self._load_model_info()

    def _load_model_info(self):
        """Loads model metadata needed for inference/fallback"""
        if os.path.exists(self.model_info_path):
            try:
                return joblib.load(self.model_info_path)
            except Exception as e:
                logger.error(f"Error loading model info: {e}")
        return {}

    def load_text_model(self, model_name=None):
        """Lazy loader for Transformers models"""
        if self._model is not None:
            return self._model, self._tokenizer

        candidates = [model_name] if model_name else TEXT_MODEL_CANDIDATES
        
        for candidate in candidates:
            try:
                logger.info(f"Loading text model: {candidate}...")
                self._tokenizer = AutoTokenizer.from_pretrained(candidate)
                
                # Determine model class based on name
                if "t5" in candidate.lower():
                    model_class = T5EncoderModel
                else:
                    model_class = AutoModel

                self._model = model_class.from_pretrained(
                    candidate,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device).eval()
                
                logger.info(f"Successfully loaded {candidate}")
                return self._model, self._tokenizer
            except Exception as e:
                logger.warning(f"Failed to load {candidate}: {e}")
                continue
        
        raise RuntimeError("Could not load any text embedding model.")

    @torch.no_grad()
    def embed_text(self, texts, batch_size=None):
        """Generates raw text embeddings using mean pooling"""
        model, tokenizer = self.load_text_model()
        
        if batch_size is None:
            batch_size = BATCH_SIZE_GPU if self.device.type == "cuda" else BATCH_SIZE_CPU

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=TEXT_MAX_LENGTH, 
                return_tensors="pt"
            ).to(self.device)
            
            outputs = model(**tokens)
            hidden = outputs.last_hidden_state
            
            # Mean pooling with attention mask awareness
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().numpy())
            
        return np.vstack(all_embeddings)

    def get_clinical_embedding(self, text=None, features=None):
        """
        Main entry point for single-sample inference.
        Returns a 64-dim embedding vector.
        """
        if text and self.model_info.get("method") == "text_model":
            pca = self.model_info.get("pca")
            raw = self.embed_text([text])
            return pca.transform(raw)[0] if pca else raw[0]
            
        if features and self.model_info.get("method") == "feature_based":
            scaler = self.model_info.get("scaler")
            poly = self.model_info.get("poly")
            pca = self.model_info.get("pca")
            
            # Extract only the needed features in correct order
            X = pd.DataFrame([features])
            if scaler: X = scaler.transform(X)
            if poly: X = poly.transform(X)
            if pca: return pca.transform(X)[0]
            
        return np.zeros(EMBEDDING_DIM)

def validate_embeddings(embeddings, target_values=None, verbose=True):
    """Utility to validate embedding quality (all zeros, variance, correlation)"""
    from scipy.stats import pearsonr
    issues = []
    metrics = {}
    
    # Check 1: Zero ratio
    zero_ratio = (embeddings == 0).mean()
    metrics['zero_ratio'] = zero_ratio
    if zero_ratio > 0.8: issues.append("CRITICAL: Over 80% values are zero")
    
    # Check 2: Variance
    variances = embeddings.var(axis=0)
    metrics['variance_mean'] = variances.mean()
    if variances.mean() < 0.001: issues.append("CRITICAL: Near-zero variance")
    
    # Check 3: Correlation (if target provided)
    if target_values is not None:
        try:
            corrs = [abs(pearsonr(embeddings[:, i], target_values)[0]) for i in range(embeddings.shape[1])]
            metrics['max_corr'] = max(corrs)
            if max(corrs) < 0.01: issues.append("WARNING: Low correlation with target")
        except: pass
        
    is_valid = not any(i.startswith("CRITICAL") for i in issues)
    
    if verbose:
        logger.info(f"Embedding Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        if issues: logger.warning(f"Issues: {issues}")
        
    return is_valid, issues, metrics

    return is_valid, issues, metrics

# ========================================
# MODEL MANAGEMENT
# ========================================

class ModelContainer:
    """Singleton container for the trained readmission model"""
    def __init__(self, model_path=None):
        from .config import MAIN_MODEL_PKL
        self.model_path = model_path or MAIN_MODEL_PKL
        self.model_data = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            return
        try:
            self.model_data = joblib.load(self.model_path)
            logger.info(f"Model loaded with {len(self.model_data['features'])} features")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# Global generator and model instances
_generator = None
model_container = None

def get_embedding(text=None, features=None):
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator.get_clinical_embedding(text=text, features=features)

def get_model_container():
    global model_container
    if model_container is None:
        model_container = ModelContainer()
    return model_container
