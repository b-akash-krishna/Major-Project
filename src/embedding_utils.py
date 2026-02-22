# src/embedding_utils.py
"""
Centralized Embedding Utilities
Handles embedding generation for both training and inference.

Embedding method used during training (02_embed.py):
  - sentence-transformers/all-mpnet-base-v2  (768-dim mean pooling)
  - PCA(128) fitted on training corpus
  - Stored in embedding_info.pkl as:
      {'method': 'sentence_transformers',
       'model_name': 'sentence-transformers/all-mpnet-base-v2',
       'pca': <fitted sklearn PCA>}

The Colab ClinicalT5 conversion path (load_clinical_t5) is also supported
but is only used when embedding_info.pkl explicitly records method='clinical_t5'.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import torch

logger = logging.getLogger(__name__)

# ── Config import (works both as package and as direct script) ─────────────────
try:
    from .config import (
        TEXT_MODEL_CANDIDATES, EMBEDDING_DIM, TEXT_MAX_LENGTH,
        BATCH_SIZE_GPU, BATCH_SIZE_CPU, EMBEDDING_INFO_PKL, RANDOM_STATE,
        MAIN_MODEL_PKL,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        TEXT_MODEL_CANDIDATES, EMBEDDING_DIM, TEXT_MAX_LENGTH,
        BATCH_SIZE_GPU, BATCH_SIZE_CPU, EMBEDDING_INFO_PKL, RANDOM_STATE,
        MAIN_MODEL_PKL,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_primary_model(model_data: dict):
    """
    model_data['models'] is a list of (name, model) tuples as saved by 03_train.py.
    Returns the first LightGBM model (or first model if no LightGBM found).
    """
    models = model_data.get("models", [])
    if not models:
        # Fallback: old pkl schema stored a single model under 'model'
        return model_data.get("model")
    # Prefer LightGBM
    for name, m in models:
        if name == "lgbm":
            return m
    # Any first model
    return models[0][1]


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class ClinicalNoteChunker:
    """Handles chunking for long clinical notes that exceed model max_length."""

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def chunk_text(self, text: str, overlap: int = 50) -> list:
        words = text.split()
        chunks = []
        step = max(self.max_tokens - overlap, 1)
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i: i + self.max_tokens]))
        return chunks or [text]


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingGenerator:
    """
    Generates 128-dim clinical embeddings for single-sample inference,
    matching the method used in 02_embed.py during training.

    Priority order for embedding method:
      1. Read embedding_info.pkl → use stored method + PCA
      2. sentence-transformers/all-mpnet-base-v2  (training default)
      3. ClinicalT5 (if conversion folder exists)
      4. Zero vector fallback (silent, logs a warning)
    """

    def __init__(self, model_info_path: str = EMBEDDING_INFO_PKL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_info_path = model_info_path
        self._st_model = None          # sentence-transformers model
        self._hf_model = None          # HuggingFace Transformer model
        self._hf_tokenizer = None
        self.model_info: dict = self._load_model_info()

    # ── Info loading ──────────────────────────────────────────────────────────

    def _load_model_info(self) -> dict:
        if os.path.exists(self.model_info_path):
            try:
                info = joblib.load(self.model_info_path)
                logger.info("Embedding info loaded: method=%s model=%s",
                            info.get("method"), info.get("model_name"))
                return info
            except Exception as e:
                logger.warning("Could not load embedding_info.pkl: %s", e)
        logger.warning(
            "embedding_info.pkl not found at %s — will use default "
            "sentence-transformers/all-mpnet-base-v2 without PCA.",
            self.model_info_path,
        )
        return {}

    # ── Sentence-Transformers (training default) ──────────────────────────────

    def _load_st_model(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        if self._st_model is not None:
            return self._st_model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer: %s", model_name)
            self._st_model = SentenceTransformer(model_name, device=str(self.device))
            return self._st_model
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def _embed_with_st(self, text: str, model_name: str) -> np.ndarray:
        """768-dim embedding from sentence-transformers (matches 02_embed.py)."""
        model = self._load_st_model(model_name)
        vec = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return vec  # shape (1, 768)

    # ── ClinicalT5 (Colab conversion path) ───────────────────────────────────

    def _embed_with_clinical_t5(self, text: str, converted_dir: str) -> np.ndarray:
        """
        Loads converted ClinicalT5 PyTorch weights (from Colab cell 7) and
        returns a 768-dim mean-pooled embedding.
        """
        from transformers import AutoTokenizer, T5EncoderModel
        if self._hf_model is None:
            logger.info("Loading ClinicalT5 from %s ...", converted_dir)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self._hf_tokenizer = AutoTokenizer.from_pretrained(converted_dir)
            self._hf_model = T5EncoderModel.from_pretrained(
                converted_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()

        tokens = self._hf_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden = self._hf_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return pooled.cpu().float().numpy()  # shape (1, 768)

    # ── HuggingFace fallback (Bio_ClinicalBERT etc.) ──────────────────────────

    def _embed_with_hf(self, text: str, model_name: str) -> np.ndarray:
        from transformers import AutoTokenizer, AutoModel
        if self._hf_model is None:
            logger.info("Loading HuggingFace model: %s", model_name)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        tokens = self._hf_tokenizer(
            text, padding=True, truncation=True,
            max_length=TEXT_MAX_LENGTH, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden = self._hf_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return pooled.cpu().float().numpy()

    # ── Main inference entry point ────────────────────────────────────────────

    def get_clinical_embedding(self, text: str = None, features: dict = None) -> np.ndarray:
        """
        Returns a 128-dim embedding vector matching what was used during training.

        Logic:
          1. Determine method from embedding_info.pkl
          2. Generate 768-dim raw embedding from text (or zeros if no text)
          3. Apply stored PCA to compress to 128-dim
          4. If no PCA available, truncate/pad to EMBEDDING_DIM
        """
        method     = self.model_info.get("method", "sentence_transformers")
        model_name = self.model_info.get("model_name",
                                         "sentence-transformers/all-mpnet-base-v2")
        pca        = self.model_info.get("pca")          # fitted sklearn PCA or None
        scaler     = self.model_info.get("scaler")       # optional StandardScaler

        raw_768: np.ndarray = None

        # ── Step 1: generate 768-dim raw embedding ────────────────────────
        if text and text.strip():
            try:
                if method == "sentence_transformers":
                    raw_768 = self._embed_with_st(text, model_name)

                elif method == "clinical_t5":
                    converted_dir = self.model_info.get(
                        "converted_dir", "clinical_t5_pytorch"
                    )
                    raw_768 = self._embed_with_clinical_t5(text, converted_dir)

                elif method == "huggingface":
                    raw_768 = self._embed_with_hf(text, model_name)

                else:
                    # Unknown method — try sentence-transformers as safe default
                    logger.warning(
                        "Unknown embedding method '%s', falling back to "
                        "sentence-transformers/all-mpnet-base-v2", method
                    )
                    raw_768 = self._embed_with_st(
                        text, "sentence-transformers/all-mpnet-base-v2"
                    )

            except Exception as e:
                logger.warning("Text embedding failed (%s) — using zero vector.", e)
                raw_768 = np.zeros((1, 768), dtype=np.float32)
        else:
            # No clinical note provided — zero vector (model trained on sparse text too)
            raw_768 = np.zeros((1, 768), dtype=np.float32)

        # ── Step 2: apply scaler + PCA (must match 02_embed.py exactly) ──
        if scaler is not None:
            try:
                raw_768 = scaler.transform(raw_768)
            except Exception as e:
                logger.warning("Scaler transform failed: %s", e)

        if pca is not None:
            try:
                result = pca.transform(raw_768)[0]  # (128,)
                return result.astype(np.float32)
            except Exception as e:
                logger.warning("PCA transform failed: %s", e)

        # Fallback: truncate/pad to EMBEDDING_DIM
        vec = raw_768[0]
        if len(vec) >= EMBEDDING_DIM:
            return vec[:EMBEDDING_DIM].astype(np.float32)
        return np.pad(vec, (0, EMBEDDING_DIM - len(vec))).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_embeddings(embeddings: np.ndarray, target_values=None,
                        verbose: bool = True):
    """
    Validates embedding quality: zero ratio, variance, target correlation.
    Returns (is_valid, issues_list, metrics_dict).
    """
    from scipy.stats import pearsonr
    issues, metrics = [], {}

    zero_ratio = float((embeddings == 0).mean())
    metrics["zero_ratio"] = zero_ratio
    if zero_ratio > 0.8:
        issues.append("CRITICAL: Over 80% values are zero")

    var_mean = float(embeddings.var(axis=0).mean())
    metrics["variance_mean"] = var_mean
    if var_mean < 0.001:
        issues.append("CRITICAL: Near-zero variance")

    if target_values is not None:
        try:
            corrs = [
                abs(pearsonr(embeddings[:, i], target_values)[0])
                for i in range(embeddings.shape[1])
            ]
            metrics["max_corr"] = float(max(corrs))
            if max(corrs) < 0.01:
                issues.append("WARNING: Low correlation with target")
        except Exception:
            pass

    is_valid = not any(i.startswith("CRITICAL") for i in issues)

    if verbose:
        status = "PASS" if is_valid else "FAIL"
        logger.info("Embedding Validation: %s", status)
        if issues:
            logger.warning("Issues: %s", issues)

    return is_valid, issues, metrics   # single return — duplicate removed


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

class ModelContainer:
    """
    Loads and holds the trained readmission model (trance_framework.pkl).
    Exposes the primary LightGBM model via self.primary_model for SHAP / inference.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or MAIN_MODEL_PKL
        self.model_data: dict = {}
        self.primary_model = None   # the actual sklearn/lgbm estimator
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.error("Model file not found at %s", self.model_path)
            return
        try:
            self.model_data = joblib.load(self.model_path)
            self.primary_model = _extract_primary_model(self.model_data)
            logger.info(
                "Model loaded: %d features | %d ensemble members",
                len(self.model_data.get("features", [])),
                len(self.model_data.get("models", [])),
            )
        except Exception as e:
            logger.error("Error loading model: %s", e)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Full ensemble prediction (LightGBM + XGBoost meta) with calibration.
        Returns probability of readmission for each row.
        """
        models     = self.model_data.get("models", [])
        meta       = self.model_data.get("meta")
        calibrator = self.model_data.get("calibrator")

        if not models:
            raise RuntimeError("No models found in pkl. Retrain first.")

        X_arr = X.values.astype(np.float32)

        if meta is not None and len(models) > 1:
            stack = np.column_stack([m.predict_proba(X_arr)[:, 1] for _, m in models])
            raw   = meta.predict_proba(stack)[:, 1]
        else:
            raw = np.mean(
                [m.predict_proba(X_arr)[:, 1] for _, m in models], axis=0
            )

        if calibrator is not None:
            return calibrator.predict(raw).astype(np.float32)
        return raw.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_generator: EmbeddingGenerator = None
_model_container: ModelContainer = None


def get_embedding(text: str = None, features: dict = None) -> np.ndarray:
    """Module-level convenience — returns 128-dim embedding vector."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator.get_clinical_embedding(text=text, features=features)


def get_model_container() -> ModelContainer:
    """Module-level convenience — returns singleton ModelContainer."""
    global _model_container
    if _model_container is None:
        _model_container = ModelContainer()
    return _model_container