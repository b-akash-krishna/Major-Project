# src/02_embed.py
"""
TRANCE Framework - Optimized Clinical Embedding Generation v2

Critical fixes from v1:
  1. ClinicalT5 uses Flax weights → load with from_flax=True
  2. Also tries safetensors format
  3. Falls back to Bio_ClinicalBERT (PyTorch native) if T5 fails
  4. Proper clinical note preprocessing (section extraction)
  5. Batch size 4 with FP16 for GTX 3050 8GB
  6. 128-dim PCA output

Note on ClinicalT5 (luqh/ClinicalT5-base):
  This model is stored as Flax (JAX) weights only on HuggingFace.
  We must use from_flax=True OR install flax+jax.
  Alternatively Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) is PyTorch-native
  and achieves very similar quality for embedding extraction.
"""

import gc
import logging
import os
import re
import sys
import warnings
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from .config import (DATA_DIR, EMBEDDINGS_CSV, EMBEDDING_INFO_PKL,
                         FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_NOTE_DIR, RANDOM_STATE)
except ImportError:
    from config import (DATA_DIR, EMBEDDINGS_CSV, EMBEDDING_INFO_PKL,
                        FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_NOTE_DIR, RANDOM_STATE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBEDDING_DIM   = 128          # final PCA dimension
MAX_SEQ_LEN     = 512
GPU_BATCH       = 4            # safe for 3050 8GB
CPU_BATCH       = 8
MIN_TEXT_LEN    = 50
MAX_TEXT_CHARS  = 3000         # chars per admission

# Model priority order — first one that loads wins
MODEL_CANDIDATES = [
    ("luqh/ClinicalT5-base",              "t5",    True),    # T5 encoder, Flax
    ("emilyalsentzer/Bio_ClinicalBERT",   "bert",  False),   # BERT, PyTorch ✓
    ("allenai/biomed_roberta_base",       "bert",  False),   # BioMed RoBERTa
    ("sentence-transformers/all-mpnet-base-v2", "bert", False),  # general fallback
]

HIGH_VALUE_SECTIONS = [
    "brief hospital course", "hospital course", "discharge diagnosis",
    "discharge condition", "assessment and plan", "assessment/plan",
    "pertinent results", "discharge medications", "history of present illness",
    "past medical history", "social history", "discharge disposition",
    "major surgical", "reason for hospitalization",
]


# ── TEXT PREPROCESSING ────────────────────────────────────────────────────────

def preprocess_note(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(text, str) or len(text.strip()) < MIN_TEXT_LEN:
        return ""
    # Remove MIMIC de-id placeholders
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Try extracting high-value sections
    extracted = []
    text_lower = text.lower()
    for sec in HIGH_VALUE_SECTIONS:
        pat = re.compile(
            rf"(?:^|\n|\r){re.escape(sec)}\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z /]+[:\-]|\Z)",
            re.IGNORECASE | re.DOTALL
        )
        m = pat.search(text_lower)
        if m:
            s, e = m.span(1)
            extracted.append(text[s:e].strip()[:1000])

    result = " [SEP] ".join(extracted) if extracted else text
    return result[:max_chars]


def build_file_index(dirs: List[str]) -> Dict[str, str]:
    idx = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                idx.setdefault(f, os.path.join(root, f))
    return idx


def load_notes(cohort_hadm: set, file_index: Dict[str, str]) -> pd.DataFrame:
    frames = []

    # Discharge notes (primary)
    for fn in ["discharge.csv.gz", "discharge.csv"]:
        if fn in file_index:
            logger.info("Loading discharge notes ...")
            try:
                df = pd.read_csv(file_index[fn], usecols=["hadm_id", "text"],
                                 nrows=1_500_000, low_memory=True)
                df = df[df["hadm_id"].isin(cohort_hadm)].dropna(subset=["text"])
                df["text"] = df["text"].apply(preprocess_note)
                df = df[df["text"].str.len() >= MIN_TEXT_LEN]
                agg = df.groupby("hadm_id")["text"].apply(" [DC] ".join).reset_index()
                frames.append(agg)
                logger.info("  %d discharge notes", len(agg))
            except Exception as e:
                logger.error("Discharge: %s", e)
            break

    # Radiology (supplement)
    for fn in ["radiology.csv.gz", "radiology.csv"]:
        if fn in file_index:
            logger.info("Loading radiology notes ...")
            try:
                df = pd.read_csv(file_index[fn], usecols=["hadm_id", "text"],
                                 nrows=800_000, low_memory=True)
                df = df[df["hadm_id"].isin(cohort_hadm)].dropna(subset=["text"])
                df["text"] = df["text"].apply(lambda t: preprocess_note(t, max_chars=800))
                df = df[df["text"].str.len() >= MIN_TEXT_LEN]
                agg = df.groupby("hadm_id")["text"].apply(" [RAD] ".join).reset_index()
                frames.append(agg)
                logger.info("  %d radiology notes", len(agg))
            except Exception as e:
                logger.error("Radiology: %s", e)
            break

    if not frames:
        return pd.DataFrame(columns=["hadm_id", "text"])

    notes = frames[0]
    for fr in frames[1:]:
        notes = notes.merge(fr, on="hadm_id", how="outer", suffixes=("", "_y"))
        if "text_y" in notes.columns:
            notes["text"] = notes["text"].fillna("") + " " + notes["text_y"].fillna("")
            notes = notes.drop(columns=["text_y"])
    notes["text"] = notes["text"].str.strip()
    notes = notes[notes["text"].str.len() >= MIN_TEXT_LEN].drop_duplicates("hadm_id")
    return notes


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_encoder(device: torch.device):
    """
    Try to load ClinicalT5 (with from_flax=True) or fall back to Bio_ClinicalBERT.
    Returns (model, tokenizer, model_type).
    """
    from transformers import (AutoModel, AutoTokenizer, T5EncoderModel)

    for model_name, mtype, try_flax in MODEL_CANDIDATES:
        logger.info("Trying %s ...", model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if device.type == "cuda" else torch.float32

            if mtype == "t5" and try_flax:
                # ClinicalT5 is Flax-only on HuggingFace
                model = T5EncoderModel.from_pretrained(
                    model_name, from_flax=True,
                    dtype=dtype, low_cpu_mem_usage=True
                ).to(device).eval()
            else:
                model = AutoModel.from_pretrained(
                    model_name, dtype=dtype, low_cpu_mem_usage=True
                ).to(device).eval()

            logger.info("Loaded %s on %s", model_name, device)
            return model, tokenizer, mtype, model_name

        except Exception as e:
            logger.warning("  Failed %s: %s", model_name, str(e)[:120])
            continue

    raise RuntimeError("All models failed to load.")


# ── ENCODING ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(model, tokenizer, texts: List[str], device: torch.device,
                 mtype: str) -> np.ndarray:
    tokens = tokenizer(texts, padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN, return_tensors="pt")
    input_ids      = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    if mtype == "t5":
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    hidden = out.last_hidden_state                          # (B, L, H)
    mask   = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    result = pooled.cpu().float().numpy()
    del input_ids, attention_mask, out, hidden, mask, pooled
    return result


def encode_all(model, tokenizer, texts: List[str], device: torch.device,
               mtype: str, batch_size: int) -> np.ndarray:
    all_emb = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = [t if t and len(t.strip()) > 0 else "patient admission" for t in texts[i:i+batch_size]]
        try:
            emb = encode_batch(model, tokenizer, batch, device, mtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM at batch %d; trying batch_size=1", i)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                emb = np.vstack([encode_batch(model, tokenizer, [t], device, mtype) for t in batch])
            else:
                raise
        all_emb.append(emb)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        if (i // batch_size) % 50 == 0:
            logger.info("  Encoded %d/%d (%.1f%%)", min(i+batch_size, n), n, min(i+batch_size, n)/n*100)
    return np.vstack(all_emb).astype(np.float32)


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

class EmbeddingPipeline:

    def __init__(self):
        self.file_index = build_file_index([MIMIC_NOTE_DIR, MIMIC_BHC_DIR])

    def run(self):
        if not os.path.exists(FEATURES_CSV):
            raise FileNotFoundError(f"Run 01_extract.py first. {FEATURES_CSV} not found.")

        feat_df = pd.read_csv(FEATURES_CSV, usecols=["hadm_id", "readmit_30"])
        cohort_hadm = set(feat_df["hadm_id"].astype(int))
        feat_hadm   = feat_df["hadm_id"].tolist()
        logger.info("Cohort: %d admissions", len(cohort_hadm))

        notes        = load_notes(cohort_hadm, self.file_index)
        hadm_to_text = dict(zip(notes["hadm_id"], notes["text"]))
        texts        = [hadm_to_text.get(h, "patient hospitalization") for h in feat_hadm]
        n_matched    = sum(h in hadm_to_text for h in feat_hadm)
        logger.info("Notes matched: %d/%d (%.1f%%)", n_matched, len(feat_hadm), n_matched/len(feat_hadm)*100)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        embeddings = None
        method     = "feature_fallback"
        model_name_used = "none"

        # ── Try transformer encoding ──────────────────────────────────────
        try:
            model, tokenizer, mtype, model_name_used = load_encoder(device)
            batch_size = GPU_BATCH if device.type == "cuda" else CPU_BATCH
            logger.info("Encoding %d texts with %s (batch=%d) ...", len(texts), model_name_used, batch_size)

            raw = encode_all(model, tokenizer, texts, device, mtype, batch_size)
            logger.info("Raw embeddings: %s", raw.shape)

            zero_ratio = (raw == 0).mean()
            var_mean   = raw.var(axis=0).mean()
            logger.info("Quality check: zero_ratio=%.3f var_mean=%.5f", zero_ratio, var_mean)

            if zero_ratio > 0.9 or var_mean < 1e-7:
                raise ValueError(f"Poor embedding quality: zero={zero_ratio:.3f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            # PCA
            scaler  = StandardScaler()
            raw_std = scaler.fit_transform(raw)
            del raw; gc.collect()

            n_comp = min(EMBEDDING_DIM, raw_std.shape[1], raw_std.shape[0]-1)
            pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE, svd_solver="randomized")
            embeddings = pca.fit_transform(raw_std).astype(np.float32)
            logger.info("PCA explained variance: %.1f%%", pca.explained_variance_ratio_.sum()*100)
            del raw_std; gc.collect()

            method     = f"transformer_{mtype}"
            model_info = {"method": method, "model": model_name_used,
                          "scaler": scaler, "pca": pca, "dim": n_comp}

        except Exception as e:
            logger.error("Transformer encoding failed: %s", e)
            logger.warning("Falling back to structured-feature SVD embeddings.")
            embeddings = None

        # ── Feature-based fallback ────────────────────────────────────────
        if embeddings is None:
            logger.info("Building SVD embeddings from structured features ...")
            ff = pd.read_csv(FEATURES_CSV, low_memory=False)
            num_cols = [c for c in ff.select_dtypes(include=[np.number]).columns
                        if c not in ["hadm_id","subject_id","readmit_30"]]
            X = ff[num_cols].fillna(0).values
            scaler = StandardScaler()
            Xs     = scaler.fit_transform(X)
            n_comp = min(EMBEDDING_DIM, Xs.shape[1], Xs.shape[0]-1)
            svd    = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
            embeddings = svd.fit_transform(Xs).astype(np.float32)
            feat_hadm  = ff["hadm_id"].tolist()
            model_info = {"method": "feature_svd", "scaler": scaler, "pca": svd, "dim": n_comp}
            del ff, X, Xs; gc.collect()
            model_name_used = "feature_svd"

        # ── Save ──────────────────────────────────────────────────────────
        n_dims = embeddings.shape[1]
        emb_df = pd.DataFrame(embeddings, columns=[f"ct5_{i}" for i in range(n_dims)])
        emb_df["hadm_id"] = feat_hadm
        emb_df.to_csv(EMBEDDINGS_CSV, index=False)
        joblib.dump(model_info, EMBEDDING_INFO_PKL)
        logger.info("Saved embeddings → %s (%d dims, method=%s)", EMBEDDINGS_CSV, n_dims, model_name_used)


if __name__ == "__main__":
    EmbeddingPipeline().run()