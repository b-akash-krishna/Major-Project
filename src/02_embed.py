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
    from .config import (
        DATA_DIR, EMBEDDINGS_CSV, EMBEDDING_INFO_PKL,
        FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_NOTE_DIR, RANDOM_STATE,
        CLINICAL_T5_LARGE_DIR, CLINICAL_T5_BASE_DIR, CLINICAL_T5_SCI_DIR,
        EMBED_DIM, EMBED_MAX_SEQ_LEN, EMBED_GPU_BATCH, EMBED_CPU_BATCH,
        EMBED_MIN_TEXT_LEN, EMBED_MAX_CHARS, EMBED_CHUNK_WORDS,
        EMBED_CHUNK_OVERLAP, EMBED_MAX_CHUNKS,
    )
except ImportError:
    from config import (
        DATA_DIR, EMBEDDINGS_CSV, EMBEDDING_INFO_PKL,
        FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_NOTE_DIR, RANDOM_STATE,
        CLINICAL_T5_LARGE_DIR, CLINICAL_T5_BASE_DIR, CLINICAL_T5_SCI_DIR,
        EMBED_DIM, EMBED_MAX_SEQ_LEN, EMBED_GPU_BATCH, EMBED_CPU_BATCH,
        EMBED_MIN_TEXT_LEN, EMBED_MAX_CHARS, EMBED_CHUNK_WORDS,
        EMBED_CHUNK_OVERLAP, EMBED_MAX_CHUNKS,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FINETUNED_ENCODER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "clinical_t5_finetuned",
    "encoder",
)

# ── CONFIG (all values sourced from config.py) ───────────────────────────────
EMBEDDING_DIM       = EMBED_DIM
MAX_SEQ_LEN         = EMBED_MAX_SEQ_LEN
GPU_BATCH           = EMBED_GPU_BATCH
CPU_BATCH           = EMBED_CPU_BATCH
MIN_TEXT_LEN        = EMBED_MIN_TEXT_LEN
MAX_TEXT_CHARS      = EMBED_MAX_CHARS
CHUNK_WORDS         = EMBED_CHUNK_WORDS
CHUNK_OVERLAP       = EMBED_CHUNK_OVERLAP
MAX_CHUNKS_PER_NOTE = EMBED_MAX_CHUNKS
NOTES_CACHE_PATH    = os.path.join(DATA_DIR, "embed_cache", "notes_preprocessed.csv.gz")

def _model_candidates() -> List[tuple]:
    # Prioritize finetuned encoder first, then local Clinical-T5 models (Large > Base > Sci).
    cand: List[tuple] = []
    if os.path.exists(os.path.join(FINETUNED_ENCODER_DIR, "config.json")):
        cand.append((FINETUNED_ENCODER_DIR, "t5", False))
    if os.path.exists(os.path.join(CLINICAL_T5_LARGE_DIR, "config.json")):
        cand.append((CLINICAL_T5_LARGE_DIR, "t5", False))
    if os.path.exists(os.path.join(CLINICAL_T5_BASE_DIR, "config.json")):
        cand.append((CLINICAL_T5_BASE_DIR, "t5", False))
    if os.path.exists(os.path.join(CLINICAL_T5_SCI_DIR, "config.json")):
        cand.append((CLINICAL_T5_SCI_DIR, "t5", False))
    cand.extend([
        ("luqh/ClinicalT5-base", "t5", True),    # HF Flax weights
        ("emilyalsentzer/Bio_ClinicalBERT", "bert", False),
        ("allenai/biomed_roberta_base", "bert", False),
        ("sentence-transformers/all-mpnet-base-v2", "bert", False),
    ])
    return cand

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
    text = text.replace("\r", "\n")

    # Try extracting high-value sections
    extracted = []
    text_lower = text.lower()  # same length as text; safe for span mapping
    for sec in HIGH_VALUE_SECTIONS:
        pat = re.compile(
            rf"{re.escape(sec)}\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z /]{2,40}\s*[:\-]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        m = pat.search(text_lower)
        if m:
            s, e = m.span(1)
            block = re.sub(r"\s+", " ", text[s:e]).strip()
            if block:
                extracted.append(block[:1200])

    if extracted:
        result = " [SEP] ".join(extracted)
    else:
        result = re.sub(r"\s+", " ", text).strip()
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
    # Fast path: reuse cached preprocessed notes if available
    if os.path.exists(NOTES_CACHE_PATH):
        logger.info("Loading cached notes → %s", NOTES_CACHE_PATH)
        try:
            cached = pd.read_csv(NOTES_CACHE_PATH, usecols=["hadm_id", "text"])
            cached = cached[cached["hadm_id"].isin(cohort_hadm)].dropna(subset=["text"])
            cached = cached[cached["text"].str.len() >= MIN_TEXT_LEN].drop_duplicates("hadm_id")
            logger.info("  Cached notes matched: %d", len(cached))
            return cached
        except Exception as e:
            logger.warning("Failed to read notes cache, rebuilding. %s", str(e)[:120])

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

    # Save cache for faster restarts
    try:
        os.makedirs(os.path.dirname(NOTES_CACHE_PATH), exist_ok=True)
        notes.to_csv(NOTES_CACHE_PATH, index=False, compression="gzip")
        logger.info("Saved preprocessed notes cache → %s", NOTES_CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to save notes cache: %s", str(e)[:120])

    return notes


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_encoder(device: torch.device):
    """
    Try to load ClinicalT5 (with from_flax=True) or fall back to Bio_ClinicalBERT.
    Returns (model, tokenizer, model_type).
    """
    from transformers import (AutoModel, AutoTokenizer, T5EncoderModel)

    for model_name, mtype, try_flax in _model_candidates():
        logger.info("Trying %s ...", model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if device.type == "cuda" else torch.float32

            if mtype == "t5":
                if try_flax:
                    # ClinicalT5 is Flax-only on HuggingFace
                    model = T5EncoderModel.from_pretrained(
                        model_name, from_flax=True,
                        dtype=dtype, low_cpu_mem_usage=True
                    ).to(device).eval()
                else:
                    model = T5EncoderModel.from_pretrained(
                        model_name,
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


from torch.utils.data import Dataset, DataLoader

class NoteDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        chunks = _note_chunks(text)
        if not chunks:
            chunks = ["[NO_NOTE]"]
        return chunks

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


def _note_chunks(text: str) -> List[str]:
    if not text or len(text.strip()) == 0:
        return []
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(CHUNK_WORDS - CHUNK_OVERLAP, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_WORDS]).strip()
        if chunk:
            chunks.append(chunk)
        if len(chunks) >= MAX_CHUNKS_PER_NOTE:
            break
    return chunks


def encode_all(model, tokenizer, texts: List[str], device: torch.device,
               mtype: str, batch_size: int, cache_dir: str) -> np.ndarray:
    """
    Encode each admission note using DataLoader for efficient processing
    and chunked pooling. Saves intermediate results to a cache directory 
    to allow resuming.
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_emb = []
    n = len(texts)
    
    chunk_size = 10000  # Increased to 10k for faster checkpointing with DataLoader
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        cache_file = os.path.join(cache_dir, f"emb_chunk_{chunk_start}_{chunk_end}.npy")
        
        if os.path.exists(cache_file):
            logger.info("  Loaded cached chunks [%d:%d]", chunk_start, chunk_end)
            all_emb.append(np.load(cache_file))
            continue
            
        logger.info("  Processing chunks [%d:%d]", chunk_start, chunk_end)
        chunk_texts = texts[chunk_start:chunk_end]
        
        # DataLoader handles parallel extraction of chunks, though tokenization
        # still happens mostly sequentially over the batch in this simple setup.
        # It still reduces Python loop overhead.
        dataset = NoteDataset(chunk_texts)
        # Using a custom collate_fn so we just get a list of chunk lists
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, 
                                collate_fn=lambda x: x[0])
        
        chunk_emb_list = []

        for i, chunks in enumerate(dataloader):
            try:
                emb_parts = []
                # To maximize GPU utilization, we pass as many chunks as possible
                # up to batch_size directly to tokenizer
                for j in range(0, len(chunks), batch_size):
                    part = chunks[j:j + batch_size]
                    emb_parts.append(encode_batch(model, tokenizer, part, device, mtype))
                emb_note = np.vstack(emb_parts).mean(axis=0, keepdims=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM at note %d; trying chunk-by-chunk", chunk_start + i)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    emb_note = np.vstack(
                        [encode_batch(model, tokenizer, [c], device, mtype) for c in chunks]
                    ).mean(axis=0, keepdims=True)
                else:
                    raise

            chunk_emb_list.append(emb_note.astype(np.float32))

        chunk_emb = np.vstack(chunk_emb_list).astype(np.float32)
        np.save(cache_file, chunk_emb)
        all_emb.append(chunk_emb)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("  Saved chunk [%d:%d] (%.1f%% overall)", chunk_start, chunk_end, chunk_end / n * 100)

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
        texts        = [hadm_to_text.get(h, "") for h in feat_hadm]
        n_matched    = sum(h in hadm_to_text for h in feat_hadm)
        logger.info("Notes matched: %d/%d (%.1f%%)", n_matched, len(feat_hadm), n_matched/len(feat_hadm)*100)
        has_note = np.array([1.0 if t and len(t.strip()) >= MIN_TEXT_LEN else 0.0 for t in texts], dtype=np.float32)
        note_len_chars = np.array([len(t) for t in texts], dtype=np.float32)
        note_len_tokens = np.array([len(t.split()) if isinstance(t, str) else 0 for t in texts], dtype=np.float32)

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

            raw = encode_all(model, tokenizer, texts, device, mtype, batch_size,
                             cache_dir=os.path.join(DATA_DIR, "embed_cache"))
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
        # Explicit text-quality signals improve fusion robustness.
        emb_df["ct5_has_note"] = has_note
        emb_df["ct5_note_len_chars"] = np.log1p(note_len_chars).astype(np.float32)
        emb_df["ct5_note_len_tokens"] = np.log1p(note_len_tokens).astype(np.float32)
        emb_df["hadm_id"] = feat_hadm
        emb_df.to_csv(EMBEDDINGS_CSV, index=False)
        joblib.dump(model_info, EMBEDDING_INFO_PKL)
        logger.info("Saved embeddings → %s (%d dims, method=%s)", EMBEDDINGS_CSV, n_dims, model_name_used)


if __name__ == "__main__":
    EmbeddingPipeline().run()
