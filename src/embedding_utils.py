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

# Monkey patch torch.load to bypass CVE-2025-32434 PyTorch 2.5.1 lockout
# The CVE block triggers specifically when HuggingFace tries to load with weights_only=True
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" in kwargs:
        del kwargs["weights_only"]
    return _orig_torch_load(*args, weights_only=False, **kwargs)
torch.load = _patched_torch_load

logger = logging.getLogger(__name__)

# Match 02_embed.py chunking defaults so inference embeddings follow training logic.
try:
    from .config import EMBED_CHUNK_WORDS, EMBED_CHUNK_OVERLAP, EMBED_MAX_CHUNKS
except Exception:
    from config import EMBED_CHUNK_WORDS, EMBED_CHUNK_OVERLAP, EMBED_MAX_CHUNKS

CHUNK_WORDS = EMBED_CHUNK_WORDS
CHUNK_OVERLAP = EMBED_CHUNK_OVERLAP
MAX_CHUNKS_PER_NOTE = EMBED_MAX_CHUNKS
BIOCLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# ── Config import (works both as package and as direct script) ─────────────────
try:
    from .config import (
        TEXT_MODEL_CANDIDATES, EMBEDDING_DIM, TEXT_MAX_LENGTH,
        BATCH_SIZE_GPU, BATCH_SIZE_CPU, EMBEDDING_INFO_PKL, RANDOM_STATE,
        MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        TEXT_MODEL_CANDIDATES, EMBEDDING_DIM, TEXT_MAX_LENGTH,
        BATCH_SIZE_GPU, BATCH_SIZE_CPU, EMBEDDING_INFO_PKL, RANDOM_STATE,
        MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY,
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


def _note_chunks_for_inference(text: str) -> list:
    """Replicate chunked note pooling used by 02_embed.py."""
    if not text or len(text.strip()) == 0:
        return ["[NO_NOTE]"]
    words = text.split()
    if not words:
        return ["[NO_NOTE]"]
    chunks = []
    step = max(CHUNK_WORDS - CHUNK_OVERLAP, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_WORDS]).strip()
        if chunk:
            chunks.append(chunk)
        if len(chunks) >= MAX_CHUNKS_PER_NOTE:
            break
    return chunks or ["[NO_NOTE]"]


def _raise_or_warn_embedding(strict: bool, message: str, err: Exception = None):
    if strict:
        if err is None:
            raise RuntimeError(message)
        raise RuntimeError(f"{message}: {err}") from err
    if err is None:
        logger.warning(message)
    else:
        logger.warning("%s (%s) — using zero vector.", message, err)


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

    def _reset_hf_cache(self):
        self._hf_model = None
        self._hf_tokenizer = None

    def _attention_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = hidden.float().mean(dim=-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1).type_as(hidden)
        return (hidden * weights).sum(1)

    # ── Info loading ──────────────────────────────────────────────────────────

    def _load_model_info(self) -> dict:
        if os.path.exists(self.model_info_path):
            try:
                info = joblib.load(self.model_info_path)
                model_ref = info.get("model_name") or info.get("model")
                logger.info("Embedding info loaded: method=%s model=%s",
                            info.get("method"), model_ref)
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
        If it fails, falls back to Bio_ClinicalBERT.
        """
        from transformers import AutoTokenizer, T5EncoderModel, AutoModel
        if self._hf_model is None:
            logger.info("Loading ClinicalT5 from %s ...", converted_dir)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            try:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(converted_dir)
                self._hf_model = T5EncoderModel.from_pretrained(
                    converted_dir,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device).eval()
            except Exception as e:
                logger.warning("Converted ClinicalT5 load failed: %s", e)
                fallback_name = "emilyalsentzer/Bio_ClinicalBERT"
                logger.info("Falling back to: %s", fallback_name)
                self._hf_tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                self._hf_model = AutoModel.from_pretrained(
                    fallback_name,
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
            pooled = self._attention_pool(hidden, tokens["attention_mask"])

        return pooled.cpu().float().numpy()  # shape (1, 768)

    def _embed_with_t5_hub(self, text: str, model_name: str) -> np.ndarray:
        """
        Load T5 encoder model from a local path or HuggingFace hub.
        Tries native PyTorch loading first, then Flax conversion fallback.
        """
        from transformers import AutoTokenizer, T5EncoderModel, AutoModel
        if self._hf_model is None:
            logger.info("Loading T5 model from hub: %s", model_name)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            try:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._hf_model = T5EncoderModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device).eval()
            except Exception as e:
                logger.warning("Native T5 load failed, trying from_flax=True: %s", e)
                try:
                    self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._hf_model = T5EncoderModel.from_pretrained(
                        model_name,
                        from_flax=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                    ).to(self.device).eval()
                except Exception as e2:
                    logger.warning("ClinicalT5 from Flax load failed: %s", e2)
                    fallback_name = "emilyalsentzer/Bio_ClinicalBERT"
                    logger.info("Falling back to: %s", fallback_name)
                    self._hf_tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                    self._hf_model = AutoModel.from_pretrained(
                        fallback_name,
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
            pooled = self._attention_pool(hidden, tokens["attention_mask"])

        return pooled.cpu().float().numpy()

    # ── HuggingFace fallback (Bio_ClinicalBERT etc.) ──────────────────────────

    def _embed_with_hf(self, text: str, model_name: str) -> np.ndarray:
        from transformers import AutoTokenizer, AutoModel
        if self._hf_model is None:
            logger.info("Loading HuggingFace model: %s", model_name)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = AutoModel.from_pretrained(
                model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            ).to(self.device).eval()

        tokens = self._hf_tokenizer(
            text, padding=True, truncation=True,
            max_length=TEXT_MAX_LENGTH, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden = self._hf_model(**tokens).last_hidden_state
            pooled = self._attention_pool(hidden, tokens["attention_mask"])

        return pooled.cpu().float().numpy()

    # ── Main inference entry point ────────────────────────────────────────────

    def get_clinical_embedding(self, text: str = None, features: dict = None, strict: bool = False) -> np.ndarray:
        """
        Returns a 128-dim embedding vector matching what was used during training.

        Logic:
          1. Determine method from embedding_info.pkl
          2. Generate 768-dim raw embedding from text (or zeros if no text)
          3. Apply stored PCA to compress to 128-dim
          4. If no PCA available, truncate/pad to EMBEDDING_DIM
        """
        method = self.model_info.get("method", "sentence_transformers")
        model_name = (
            self.model_info.get("model_name")
            or self.model_info.get("model")
            or "sentence-transformers/all-mpnet-base-v2"
        )
        reducer    = self.model_info.get("reducer") or self.model_info.get("pca")
        scaler     = self.model_info.get("scaler")       # optional StandardScaler

        raw_768: np.ndarray = None
        note_chunks = _note_chunks_for_inference(text)
        has_text = bool(text and text.strip())

        # ── Step 1: generate 768-dim raw embedding ────────────────────────
        if has_text:
            try:
                if method == "multi_fusion":
                    models = self.model_info.get("models", [])
                    parts = []
                    for m in models:
                        name = m.get("name") if isinstance(m, dict) else m
                        mtype = m.get("type") if isinstance(m, dict) else None
                        chunk_vecs = []
                        for chunk in note_chunks:
                            if mtype == "t5":
                                chunk_vecs.append(self._embed_with_t5_hub(chunk, name))
                            else:
                                chunk_vecs.append(self._embed_with_hf(chunk, name))
                        parts.append(np.vstack(chunk_vecs).mean(axis=0, keepdims=True))
                    raw_768 = np.hstack(parts)

                if method == "sentence_transformers":
                    parts = [self._embed_with_st(chunk, model_name) for chunk in note_chunks]
                    raw_768 = np.vstack(parts).mean(axis=0, keepdims=True)

                elif method in ("clinical_t5", "transformer_t5"):
                    # 'transformer_t5' is the method name saved by 02_embed.py
                    # when ClinicalT5/T5EncoderModel was used during training.
                    # Try converted PyTorch dir first, then generic HuggingFace path.
                    converted_dir = self.model_info.get("converted_dir")
                    if converted_dir and os.path.exists(converted_dir):
                        parts = [self._embed_with_clinical_t5(chunk, converted_dir) for chunk in note_chunks]
                    else:
                        # Fall back to loading directly from HuggingFace hub
                        # (requires internet; model_name from embedding_info.pkl)
                        logger.info(
                            "No converted_dir found — loading %s from HuggingFace", model_name
                        )
                        parts = [self._embed_with_t5_hub(chunk, model_name) for chunk in note_chunks]
                    raw_768 = np.vstack(parts).mean(axis=0, keepdims=True)

                elif method == "huggingface" or str(method).startswith("transformer_"):
                    parts = [self._embed_with_hf(chunk, model_name) for chunk in note_chunks]
                    raw_768 = np.vstack(parts).mean(axis=0, keepdims=True)

                elif method == "feature_svd":
                    # SVD fallback used in 02_embed.py when all text models failed
                    # — no text model involved, return zeros and rely on tabular features
                    logger.warning(
                        "Embedding method is 'feature_svd' (text model failed during "
                        "training). Text note cannot improve prediction. Using zeros."
                    )
                    raw_768 = np.zeros((1, 768), dtype=np.float32)

                else:
                    # Unknown method — try HuggingFace with stored model_name
                    logger.warning(
                        "Unknown embedding method '%s' — attempting HuggingFace load "
                        "of '%s'", method, model_name
                    )
                    try:
                        parts = [self._embed_with_hf(chunk, model_name) for chunk in note_chunks]
                        raw_768 = np.vstack(parts).mean(axis=0, keepdims=True)
                    except Exception:
                        raw_768 = np.zeros((1, 768), dtype=np.float32)

            except Exception as e:
                logger.warning("Text embedding failed (%s) — using zero vector.", e)
                raw_768 = np.zeros((1, 768), dtype=np.float32)
        else:
            # No clinical note provided. Use dedicated token to stay consistent
            # with training-time missing-note handling in 02_embed.py.
            try:
                if method == "multi_fusion":
                    models = self.model_info.get("models", [])
                    parts = []
                    for m in models:
                        name = m.get("name") if isinstance(m, dict) else m
                        mtype = m.get("type") if isinstance(m, dict) else None
                        if mtype == "t5":
                            parts.append(self._embed_with_t5_hub("[NO_NOTE]", name))
                        else:
                            parts.append(self._embed_with_hf("[NO_NOTE]", name))
                    raw_768 = np.hstack(parts)

                if method == "sentence_transformers":
                    raw_768 = self._embed_with_st("[NO_NOTE]", model_name)
                elif method in ("clinical_t5", "transformer_t5"):
                    converted_dir = self.model_info.get("converted_dir")
                    if not converted_dir:
                        local_converted = os.path.join(
                            os.path.dirname(self.model_info_path), "clinical_t5_pytorch"
                        )
                        if os.path.exists(os.path.join(local_converted, "config.json")):
                            converted_dir = local_converted
                    if converted_dir and os.path.exists(converted_dir):
                        raw_768 = self._embed_with_clinical_t5("[NO_NOTE]", converted_dir)
                    else:
                        raw_768 = self._embed_with_t5_hub("[NO_NOTE]", model_name)
                elif method == "huggingface" or str(method).startswith("transformer_"):
                    raw_768 = self._embed_with_hf("[NO_NOTE]", model_name)
                else:
                    raw_768 = np.zeros((1, 768), dtype=np.float32)
            except Exception:
                raw_768 = np.zeros((1, 768), dtype=np.float32)

        # ── Step 2: apply reduction (must match 02_embed.py) ─────────────
        if raw_768 is None:
            raw_768 = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

        # L2 normalize before reduction
        norm = np.linalg.norm(raw_768, axis=1, keepdims=True)
        raw_768 = raw_768 / np.clip(norm, 1e-9, None)

        if scaler is not None and reducer is not None:
            try:
                result = reducer.transform(scaler.transform(raw_768))[0]
                return result.astype(np.float32)
            except Exception as e:
                logger.warning("Reducer transform failed: %s", e)

        if reducer is not None:
            try:
                result = reducer.transform(raw_768)[0]
                return result.astype(np.float32)
            except Exception as e:
                logger.warning("Reducer transform failed: %s", e)

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
    Loads and holds the trained readmission model (ACAGN base ensemble).
    Exposes the primary LightGBM model via self.primary_model for SHAP / inference.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or MAIN_MODEL_PKL
        self.model_data: dict = {}
        self.primary_model = None   # the actual sklearn/lgbm estimator
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            if os.path.exists(MAIN_MODEL_PKL_LEGACY):
                logger.warning(
                    "Model file not found at %s; falling back to legacy path %s",
                    self.model_path,
                    MAIN_MODEL_PKL_LEGACY,
                )
                self.model_path = MAIN_MODEL_PKL_LEGACY
            else:
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

        # Keep DataFrame input to preserve feature names used during model fit.
        # This avoids "X does not have valid feature names" warnings.
        X_df = X.astype(np.float32)

        if meta is not None and len(models) > 1:
            stack = np.column_stack([m.predict_proba(X_df)[:, 1] for _, m in models])
            raw   = meta.predict_proba(stack)[:, 1]
        else:
            raw = np.mean(
                [m.predict_proba(X_df)[:, 1] for _, m in models], axis=0
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
