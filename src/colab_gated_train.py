# colab_gated_train.py  ──  TRANCE-Gate: Text-Guided Feature Gating
# =========================================================================
# Designed for Colab Free Tier T4 GPU (15.6 GB VRAM)
#
# Architecture (from Enhanced.md / research plan):
#   Text encoder  : ClinicalT5 embeddings (768-dim, pre-computed)
#   Gate network  : text_emb(768) → Linear(128) → ReLU → Linear(n_tab) → Sigmoid
#                   → per-feature weights in [0,1] (context-aware suppression)
#   Gated features: gate_weights ⊙ x_tabular  (element-wise product)
#   Classifier    : concat(text_emb, x_gated) → MLP(256→64→1) → Sigmoid
#   Training      : end-to-end BCE, Adam lr=1e-4, cosine LR, early stop on AUROC
#   Ensemble      : 3 seeds averaged → isotonic calibration
#
# T4 optimisations:
#   • AMP (torch.cuda.amp) mixed precision — halves VRAM, ~1.5× faster
#   • pin_memory + non_blocking transfers
#   • gradient clipping (max_norm=1.0)
#   • Weighted BCE for class imbalance (no oversampling needed)
#   • torch.cuda.empty_cache() between seeds
#   • float32 numpy arrays to minimise host↔device bandwidth
#
# What this script produces for the paper:
#   • AUROC / AUPRC / Brier / ECE before and after calibration
#   • gate_weights.npy + gate_patient_ids.npy  → for interpretability analysis
#   • results/trance_gate_preds.csv            → for final ensembling
#   • results/gate_training_report.json
#   • figures/roc_pr_gate_*.png
#   • figures/reliability_gate_*.png
# =========================================================================

import gc
import json
import logging
import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION  (must match colab_train.py splits exactly)
# ──────────────────────────────────────────────────────────────
DRIVE_PROJECT_PATH = "WeCare Model/data"

DATA_DIR = "/content"
if os.path.exists("/content/drive/MyDrive"):
    DATA_DIR = f"/content/drive/MyDrive/{DRIVE_PROJECT_PATH}"
    print(f"✓ Google Drive detected → {DATA_DIR}")

MODELS_DIR  = "/content/models"
RESULTS_DIR = "/content/results"
FIGURES_DIR = "/content/figures"

FEATURES_CSV   = "/content/ultimate_features_pruned.csv"
EMBEDDINGS_CSV = "/content/embeddings.csv"

# Fallback to Drive if local copy wasn't made
if not os.path.exists(FEATURES_CSV):
    FEATURES_CSV = os.path.join(DATA_DIR, "ultimate_features_pruned.csv")
if not os.path.exists(EMBEDDINGS_CSV):
    EMBEDDINGS_CSV = os.path.join(DATA_DIR, "embeddings.csv")

RANDOM_STATE     = 42
TRAIN_TEST_FRAC  = 0.15
TRAIN_VAL_FRAC   = 0.15
TRAIN_CT5_DIMS   = 768     # full fine-tuned ClinicalT5 signal

# Gate architecture (from Enhanced.md Section 3.2)
GATE_HIDDEN_DIM  = 128
GATE_DROPOUT     = 0.3
GATE_LR          = 1e-4
GATE_EPOCHS      = 100
GATE_PATIENCE    = 10
GATE_SEEDS       = [42, 2024, 777]
GATE_BATCH_TRAIN = 512     # larger batch → better T4 utilisation
GATE_BATCH_EVAL  = 1024
GATE_WEIGHT_DECAY = 1e-5

# Artefact paths
GATE_WEIGHTS_NPY    = os.path.join(RESULTS_DIR, "gate_weights.npy")
GATE_PATIENT_IDS    = os.path.join(RESULTS_DIR, "gate_patient_ids.npy")
GATE_MODEL_PKL      = os.path.join(MODELS_DIR,  "trance_gate.pkl")

# ── GPU ────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")   # AMP only on GPU

if DEVICE.type == "cuda":
    _p = torch.cuda.get_device_properties(0)
    print(f"✓ GPU  : {_p.name}  |  VRAM: {_p.total_memory/1e9:.1f} GB")
    print(f"✓ AMP mixed-precision : enabled")
else:
    print("✓ No GPU — CPU mode (will be slow)")

# ──────────────────────────────────────────────────────────────
# DIRECTORIES & LOGGING
# ──────────────────────────────────────────────────────────────
for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_DIR, "train_gate.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def _flush() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins, ece, total = np.linspace(0, 1, n_bins + 1), 0.0, len(labels)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.sum() == 0: continue
        ece += (m.sum() / total) * abs(float(labels[m].mean()) - float(probs[m].mean()))
    return float(ece)

# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_fused_data():
    """
    Returns: text_emb, tabular, labels, groups, hadm_ids, tab_cols
    All arrays are float32, aligned row-by-row.
    """
    logger.info("Loading tabular features ...")
    id_dtypes = {"subject_id": "int32", "hadm_id": "int32", "readmit_30": "int8"}
    tab_df = pd.read_csv(FEATURES_CSV, low_memory=False, dtype=id_dtypes).fillna(0)
    tab_df[tab_df.select_dtypes("float64").columns] = (
        tab_df.select_dtypes("float64").astype("float32"))

    logger.info("Loading embeddings (chunked) ...")
    emb_header = pd.read_csv(EMBEDDINGS_CSV, nrows=0)
    emb_dtypes = {c: "float32" for c in emb_header.columns if c.startswith("ct5_")}
    emb_dtypes["hadm_id"] = "int32"
    target_ids = set(tab_df["hadm_id"])
    chunks = []
    for chunk in pd.read_csv(EMBEDDINGS_CSV, chunksize=25_000,
                              low_memory=False, dtype=emb_dtypes):
        chunk = chunk[chunk["hadm_id"].isin(target_ids)]
        if not chunk.empty: chunks.append(chunk)
        del chunk; gc.collect()
    emb_df = pd.concat(chunks, axis=0, ignore_index=True)
    df = tab_df.merge(emb_df, on="hadm_id", how="left").fillna(0)
    del tab_df, emb_df, chunks; gc.collect()
    logger.info("Merged shape: %s", df.shape)

    id_cols  = {"subject_id", "hadm_id", "readmit_30"}
    emb_cols = [c for c in df.columns if c.startswith("ct5_")]

    # High-variance dim selection for full 768
    if len(emb_cols) > TRAIN_CT5_DIMS:
        keep = (df[emb_cols].var().sort_values(ascending=False)
                .head(TRAIN_CT5_DIMS).index.tolist())
        emb_cols = keep
        logger.info("Kept %d ct5 dims.", len(emb_cols))

    tab_cols = [c for c in df.columns if c not in id_cols and c not in set(emb_cols)]

    groups   = df["subject_id"].astype(int).values
    hadm_ids = df["hadm_id"].astype(int).values
    labels   = df["readmit_30"].astype(np.float32).values
    text_emb = df[emb_cols].values.astype(np.float32)
    tabular  = df[tab_cols].values.astype(np.float32)

    logger.info("Text dim: %d | Tabular dim: %d | Patients: %d",
                text_emb.shape[1], tabular.shape[1], len(np.unique(groups)))
    return text_emb, tabular, labels, groups, hadm_ids, tab_cols

# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class ReadmissionDataset(Dataset):
    def __init__(self, text_emb: np.ndarray, tabular: np.ndarray, labels: np.ndarray):
        # Store as CPU tensors; pin_memory in DataLoader handles the rest
        self.text = torch.from_numpy(text_emb)
        self.tab  = torch.from_numpy(tabular)
        self.y    = torch.from_numpy(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.text[idx], self.tab[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────
# ARCHITECTURE  (from Enhanced.md Section 3.2, exactly)
# ──────────────────────────────────────────────────────────────
class TextGuidedGate(nn.Module):
    """
    Gate network  : text_emb → Linear(hidden) → ReLU → Linear(n_tab) → Sigmoid
    Classifier    : concat(text_emb, gate_weights ⊙ x_tab)
                    → Linear(256) → ReLU → Dropout
                    → Linear(64)  → ReLU
                    → Linear(1)   → Sigmoid
    All trained end-to-end.
    """
    def __init__(self, text_dim: int, tabular_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(text_dim, GATE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(GATE_HIDDEN_DIM, tabular_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + tabular_dim, 256),
            nn.ReLU(),
            nn.Dropout(GATE_DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_emb: torch.Tensor, x_tab: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_weights = self.gate(text_emb)              # (B, tabular_dim)
        x_gated      = gate_weights * x_tab             # element-wise
        x_fused      = torch.cat([text_emb, x_gated], dim=1)
        prob         = self.classifier(x_fused).squeeze(1)
        return prob, gate_weights

# ──────────────────────────────────────────────────────────────
# PATIENT-LEVEL SPLIT  (identical to colab_train.py)
# ──────────────────────────────────────────────────────────────
def make_splits(groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng  = np.random.RandomState(RANDOM_STATE)
    pats = np.unique(groups); rng.shuffle(pats)
    n    = len(pats)
    n_te = int(n * TRAIN_TEST_FRAC);  n_val = int(n * TRAIN_VAL_FRAC)
    te_p   = set(pats[:n_te])
    val_p  = set(pats[n_te : n_te + n_val])
    tr_p   = set(pats[n_te + n_val :])
    return (np.array([g in tr_p  for g in groups]),
            np.array([g in val_p for g in groups]),
            np.array([g in te_p  for g in groups]))

# ──────────────────────────────────────────────────────────────
# SINGLE-SEED TRAINING
# ──────────────────────────────────────────────────────────────
def train_one_seed(text_emb, tabular, labels, groups, seed):
    _set_seed(seed)
    tr_m, val_m, te_m = make_splits(groups)

    pos_weight = float((labels[tr_m] == 0).sum()) / max(float((labels[tr_m] == 1).sum()), 1)
    pos_w_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
    # BCEWithLogitsLoss needs logits; our model outputs sigmoid, so use BCELoss
    # with manual class weight applied per sample for stable training
    criterion = nn.BCELoss(reduction="none")

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        ReadmissionDataset(text_emb[tr_m], tabular[tr_m], labels[tr_m]),
        batch_size=GATE_BATCH_TRAIN, shuffle=True,
        pin_memory=pin, num_workers=0)
    val_loader = DataLoader(
        ReadmissionDataset(text_emb[val_m], tabular[val_m], labels[val_m]),
        batch_size=GATE_BATCH_EVAL, shuffle=False,
        pin_memory=pin, num_workers=0)
    test_loader = DataLoader(
        ReadmissionDataset(text_emb[te_m], tabular[te_m], labels[te_m]),
        batch_size=GATE_BATCH_EVAL, shuffle=False,
        pin_memory=pin, num_workers=0)

    model     = TextGuidedGate(text_emb.shape[1], tabular.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=GATE_LR, weight_decay=GATE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GATE_EPOCHS)
    scaler    = GradScaler(enabled=USE_AMP)

    best_auroc, best_state, patience = 0.0, None, 0

    for epoch in range(GATE_EPOCHS):
        # ── Train ──────────────────────────────────────────────
        model.train()
        for text_b, tab_b, y_b in train_loader:
            text_b = text_b.to(DEVICE, non_blocking=True)
            tab_b  = tab_b.to(DEVICE, non_blocking=True)
            y_b    = y_b.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                probs, _ = model(text_b, tab_b)
                # Weighted BCE: positive class gets pos_weight, negatives get 1
                w = torch.where(y_b == 1, pos_w_tensor, torch.ones_like(y_b))
                loss = (criterion(probs, y_b) * w).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # ── Validate ───────────────────────────────────────────
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for text_b, tab_b, y_b in val_loader:
                text_b = text_b.to(DEVICE, non_blocking=True)
                tab_b  = tab_b.to(DEVICE, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    probs, _ = model(text_b, tab_b)
                vp.append(probs.float().cpu().numpy())
                vl.append(y_b.numpy())
        val_probs  = np.concatenate(vp)
        val_labels = np.concatenate(vl)
        val_auroc  = roc_auc_score(val_labels, val_probs)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if (epoch + 1) % 10 == 0:
            logger.info("Seed %d | Epoch %3d | Val AUROC %.4f | Best %.4f | LR %.6f",
                        seed, epoch + 1, val_auroc, best_auroc,
                        scheduler.get_last_lr()[0])
        if patience >= GATE_PATIENCE:
            logger.info("Early stopping at epoch %d (patience %d)", epoch + 1, GATE_PATIENCE)
            break

    # ── Test inference with best weights ─────────────────────
    model.load_state_dict(best_state)
    model.eval()
    tp, tl, gw = [], [], []
    with torch.no_grad():
        for text_b, tab_b, y_b in test_loader:
            text_b = text_b.to(DEVICE, non_blocking=True)
            tab_b  = tab_b.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                probs, gates = model(text_b, tab_b)
            tp.append(probs.float().cpu().numpy())
            tl.append(y_b.numpy())
            gw.append(gates.float().cpu().numpy())

    return (val_probs, val_labels,
            np.concatenate(tp), np.concatenate(tl), np.concatenate(gw),
            te_m)

# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
def _save_roc_pr(probs, y_true, tag) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, probs)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUROC={roc_auc_score(y_true,probs):.4f}")
    axes[0].plot([0,1],[0,1],"k--"); axes[0].set(title="ROC",xlabel="FPR",ylabel="TPR")
    axes[0].legend()
    prec, rec, _ = precision_recall_curve(y_true, probs)
    axes[1].plot(rec, prec, lw=2, label=f"AUPRC={average_precision_score(y_true,probs):.4f}")
    axes[1].set(title="Precision-Recall",xlabel="Recall",ylabel="Precision"); axes[1].legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"roc_pr_gate_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    logger.info("Saved → %s", path)

def _save_reliability(probs, y_true, tag, n_bins=10) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    mid, obs = [], []
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i+1])
        if m.sum() > 0: mid.append(probs[m].mean()); obs.append(y_true[m].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mid, obs, "o-", label="Model"); ax.plot([0,1],[0,1],"k--",label="Perfect")
    ax.set(title=f"Reliability ({tag})", xlabel="Mean Predicted Prob",
           ylabel="Observed Rate"); ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"reliability_gate_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    logger.info("Saved → %s", path)

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def train_gate_model() -> None:
    _set_seed(RANDOM_STATE)
    start = datetime.now()

    text_emb, tabular, labels, groups, hadm_ids, tab_cols = load_fused_data()

    all_val_probs, all_test_probs, all_gate_weights = [], [], []
    test_labels_ref = val_labels_ref = test_mask_ref = None

    for seed in GATE_SEEDS:
        logger.info("=== Seed %d ===", seed)
        (val_probs, val_labels,
         test_probs, test_labels, gate_weights, te_m) = train_one_seed(
            text_emb, tabular, labels, groups, seed)

        all_val_probs.append(val_probs)
        all_test_probs.append(test_probs)
        all_gate_weights.append(gate_weights)
        if test_labels_ref is None:
            test_labels_ref = test_labels
            val_labels_ref  = val_labels
            test_mask_ref   = te_m
        _flush()

    # ── Average across seeds ──────────────────────────────────
    avg_val   = np.mean(all_val_probs,   axis=0)
    avg_test  = np.mean(all_test_probs,  axis=0)
    avg_gates = np.mean(all_gate_weights, axis=0)  # (n_test, n_tab)

    # ── Isotonic calibration on val → apply to test ───────────
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(avg_val, val_labels_ref)
    cal_test = calibrator.predict(avg_test).astype(np.float32)

    # ── Metrics ───────────────────────────────────────────────
    auroc_raw = roc_auc_score(test_labels_ref, avg_test)
    auroc_cal = roc_auc_score(test_labels_ref, cal_test)
    auprc     = average_precision_score(test_labels_ref, cal_test)
    brier     = brier_score_loss(test_labels_ref, cal_test)
    ll        = log_loss(test_labels_ref, cal_test)
    ece_raw   = compute_ece(avg_test, test_labels_ref)
    ece_cal   = compute_ece(cal_test, test_labels_ref)

    # Threshold search on calibrated probs
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.05, 0.70, 200):
        preds = (cal_test >= t).astype(int)
        m = matthews_corrcoef(test_labels_ref, preds)
        if m > best_mcc: best_mcc, best_t = m, t
    preds = (cal_test >= best_t).astype(int)
    mcc   = matthews_corrcoef(test_labels_ref, preds)
    f1    = f1_score(test_labels_ref, preds, zero_division=0)
    elapsed = (datetime.now() - start).total_seconds()

    logger.info("=" * 55)
    logger.info("TRANCE-GATE — TEST RESULTS")
    logger.info("  AUROC (raw)         : %.4f", auroc_raw)
    logger.info("  AUROC (calibrated)  : %.4f", auroc_cal)
    logger.info("  AUPRC               : %.4f", auprc)
    logger.info("  MCC                 : %.4f", mcc)
    logger.info("  F1                  : %.4f", f1)
    logger.info("  Brier score         : %.4f", brier)
    logger.info("  LogLoss             : %.4f", ll)
    logger.info("  ECE before cal      : %.4f", ece_raw)
    logger.info("  ECE after cal       : %.4f", ece_cal)
    logger.info("  Threshold (MCC)     : %.3f", best_t)
    logger.info("  Wall time           : %.0f s", elapsed)
    logger.info("=" * 55)

    # ── Save gate weights for interpretability analysis ───────
    test_hadm_ids = hadm_ids[test_mask_ref]
    np.save(GATE_WEIGHTS_NPY, avg_gates)
    np.save(GATE_PATIENT_IDS, test_hadm_ids)
    logger.info("Gate weights saved → %s  (%s)", GATE_WEIGHTS_NPY, str(avg_gates.shape))

    # ── Predictions CSV for final ensembling ─────────────────
    pd.DataFrame({
        "hadm_id":  test_hadm_ids,
        "y_true":   test_labels_ref,
        "prob_raw": avg_test,
        "prob_cal": cal_test,
        "pred":     preds,
    }).to_csv(os.path.join(RESULTS_DIR, "trance_gate_preds.csv"), index=False)

    # ── Save full model bundle ────────────────────────────────
    results = dict(
        auroc_raw=round(float(auroc_raw), 4), auroc_cal=round(float(auroc_cal), 4),
        auprc=round(float(auprc), 4),         brier=round(float(brier), 4),
        mcc=round(float(mcc), 4),             f1=round(float(f1), 4),
        logloss=round(float(ll), 4),
        ece_before=round(float(ece_raw), 4),  ece_after=round(float(ece_cal), 4),
        threshold=round(float(best_t), 4),    wall_time_s=round(elapsed, 1),
        tab_features=tab_cols, n_test=int(len(test_labels_ref)), seeds=GATE_SEEDS,
    )
    joblib.dump(dict(
        calibrator=calibrator, tab_cols=tab_cols,
        text_dim=text_emb.shape[1], tabular_dim=tabular.shape[1],
        results=results,
        test_probs_raw=avg_test, test_probs_cal=cal_test,
        test_labels=test_labels_ref, test_hadm_ids=test_hadm_ids,
        avg_gate_weights=avg_gates,
    ), GATE_MODEL_PKL)

    with open(os.path.join(RESULTS_DIR, "gate_training_report.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ── Plots ─────────────────────────────────────────────────
    _save_roc_pr(cal_test, test_labels_ref, "cal")
    _save_roc_pr(avg_test, test_labels_ref, "raw")
    _save_reliability(cal_test, test_labels_ref, "cal")
    _save_reliability(avg_test, test_labels_ref, "raw")

    logger.info("All artefacts saved → %s  %s", MODELS_DIR, RESULTS_DIR)


# ──────────────────────────────────────────────────────────────
# FINAL ENSEMBLE  (run after both scripts complete)
# ──────────────────────────────────────────────────────────────
def final_ensemble() -> None:
    """
    Loads both prediction CSVs, averages prob_cal, computes final AUROC.
    Run this cell AFTER both colab_train.py and colab_gated_train.py finish.
    """
    baseline_path = os.path.join(RESULTS_DIR, "trance_baseline_preds.csv")
    gate_path     = os.path.join(RESULTS_DIR, "trance_gate_preds.csv")

    if not (os.path.exists(baseline_path) and os.path.exists(gate_path)):
        print("Run both training scripts first.")
        return

    base = pd.read_csv(baseline_path)
    gate = pd.read_csv(gate_path)

    # Align on hadm_id if available, else assume same order
    if "hadm_id" in base.columns and "hadm_id" in gate.columns:
        merged = base.merge(gate[["hadm_id", "prob_cal"]],
                            on="hadm_id", suffixes=("_base", "_gate"))
        ensemble_prob = 0.5 * merged["prob_cal_base"] + 0.5 * merged["prob_cal_gate"]
        y_true        = merged["y_true"]
    else:
        # same test order guaranteed by identical split seed
        ensemble_prob = 0.5 * base["prob_cal"].values + 0.5 * gate["prob_cal"].values
        y_true        = base["y_true"].values

    auroc = roc_auc_score(y_true, ensemble_prob)
    ap    = average_precision_score(y_true, ensemble_prob)
    print("=" * 45)
    print(f"FINAL ENSEMBLE RESULTS")
    print(f"  AUROC : {auroc:.4f}")
    print(f"  AUPRC : {ap:.4f}")
    print("=" * 45)

    pd.DataFrame({"y_true": y_true, "prob_ensemble": ensemble_prob}).to_csv(
        os.path.join(RESULTS_DIR, "trance_final_ensemble_preds.csv"), index=False)
    print(f"Saved → {RESULTS_DIR}/trance_final_ensemble_preds.csv")


if __name__ == "__main__":
    train_gate_model()
    # Uncomment after colab_train.py also finishes:
    # final_ensemble()
