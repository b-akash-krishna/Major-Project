"""
ACAGN-Gate: Text-Guided Feature Gating Model
==============================================
Architecture:
  - ClinicalT5 text embedding (256-dim, pre-computed) acts as context signal
  - Gate network: text_emb -> per-feature weights in [0,1]
  - Gated features: gate_weights * tabular_features (element-wise)
  - Classifier: concat(text_emb, gated_features) -> MLP -> readmission prob

The gate is trained jointly with the classifier end-to-end.
Gate weights are saved per patient for interpretability analysis.
"""

import os
import sys
import gc
import json
import logging
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.calibration import IsotonicRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import (
        FEATURES_CSV, EMBEDDINGS_CSV, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, RESULTS_DIR,
        GATE_HIDDEN_DIM, GATE_TEXT_DIM, GATE_DROPOUT,
        GATE_LR, GATE_EPOCHS, GATE_PATIENCE, GATE_SEEDS,
        GATE_WEIGHTS_NPY, GATE_PATIENT_IDS_NPY,
        GATE_ENABLE_SHAP, GATE_SHAP_N_BACKGROUND, GATE_SHAP_N_SAMPLES,
        GATE_SHAP_IMPORTANCE_CSV, GATE_SHAP_SUMMARY_PNG,
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,
        THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK,
    )
except ImportError:
    from .config import (
        FEATURES_CSV, EMBEDDINGS_CSV, GATE_MODEL_PKL, GATE_MODEL_PKL_LEGACY, RESULTS_DIR,
        GATE_HIDDEN_DIM, GATE_TEXT_DIM, GATE_DROPOUT,
        GATE_LR, GATE_EPOCHS, GATE_PATIENCE, GATE_SEEDS,
        GATE_WEIGHTS_NPY, GATE_PATIENT_IDS_NPY,
        GATE_ENABLE_SHAP, GATE_SHAP_N_BACKGROUND, GATE_SHAP_N_SAMPLES,
        GATE_SHAP_IMPORTANCE_CSV, GATE_SHAP_SUMMARY_PNG,
        TRAIN_TEST_FRAC, TRAIN_VAL_FRAC, RANDOM_STATE,
        THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Output safety ─────────────────────────────────────────────────────────────

def _ensure_writable(path: str, overwrite: bool) -> None:
    if path is None:
        return
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {path} (pass --overwrite to replace)")

# ── Dataset ───────────────────────────────────────────────────────────────────

class ReadmissionDataset(Dataset):
    """
    Holds text embeddings, tabular features, and labels for one split.
    Returns tensors of (text_emb, tabular, label) per patient.
    """
    def __init__(self, text_emb: np.ndarray, tabular: np.ndarray, labels: np.ndarray):
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.tabular  = torch.tensor(tabular,  dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_emb[idx], self.tabular[idx], self.labels[idx]

# ── Architecture ──────────────────────────────────────────────────────────────

class TextGuidedGate(nn.Module):
    """
    Full gated fusion model.

    Gate network:
        text_emb (256) -> Linear(256, 128) -> ReLU
                       -> Linear(128, n_tab) -> Sigmoid
        output: gate_weights in [0,1] for each tabular feature

    Gating:
        x_gated = gate_weights * x_tabular   (element-wise product)

    Classifier:
        concat(text_emb, x_gated) -> Linear(256+n_tab, 256) -> ReLU -> Dropout
                                   -> Linear(256, 64) -> ReLU
                                   -> Linear(64, 1) -> Sigmoid
    """

    def __init__(self, text_dim: int, tabular_dim: int,
                 hidden_dim: int = GATE_HIDDEN_DIM, dropout: float = GATE_DROPOUT):
        super().__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tabular_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(text_dim + tabular_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb: torch.Tensor, x_tab: torch.Tensor):
        gate_weights = self.gate_network(text_emb)       # (batch, tabular_dim)
        x_gated      = gate_weights * x_tab              # element-wise
        x_fused      = torch.cat([text_emb, x_gated], dim=1)
        prob         = self.classifier(x_fused).squeeze(1)
        return prob, gate_weights

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_fused_data():
    """
    Loads and merges tabular features with text embeddings.
    Returns aligned arrays for text, tabular, labels, groups (subject_id),
    hadm_ids, and the lists of feature names (embedding and tabular).
    """
    pruned = FEATURES_CSV.replace(".csv", "_pruned.csv")
    feat_path = pruned if os.path.exists(pruned) else FEATURES_CSV
    logger.info("Loading features from %s", feat_path)
    tab_df = pd.read_csv(feat_path, low_memory=False).fillna(0)

    logger.info("Loading embeddings from %s", EMBEDDINGS_CSV)
    emb_df = pd.read_csv(EMBEDDINGS_CSV, low_memory=False)

    df = tab_df.merge(emb_df, on="hadm_id", how="left").fillna(0)
    logger.info("Merged shape: %s", df.shape)

    # Separate columns
    id_cols  = {"subject_id", "hadm_id", "readmit_30"}
    emb_cols = [c for c in emb_df.columns if c.startswith("ct5_")]
    tab_cols = [c for c in df.columns if c not in id_cols and c not in emb_cols]

    groups   = df["subject_id"].astype(int).values
    hadm_ids = df["hadm_id"].astype(int).values
    labels   = df["readmit_30"].astype(np.float32).values

    text_emb = df[emb_cols].values.astype(np.float32)
    tabular  = df[tab_cols].values.astype(np.float32)

    logger.info("Text embedding dim: %d | Tabular features: %d", text_emb.shape[1], tabular.shape[1])
    return text_emb, tabular, labels, groups, hadm_ids, emb_cols, tab_cols


class _GateProbWrapper(nn.Module):
    """Wrap TextGuidedGate to return only probabilities for SHAP."""
    def __init__(self, gate_model: nn.Module):
        super().__init__()
        self.gate_model = gate_model

    def forward(self, text_emb: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        prob, _ = self.gate_model(text_emb, x_tab)
        # DeepExplainer expects a 2D output for single-output models.
        return prob.unsqueeze(1)


def compute_gate_shap(
    model: nn.Module,
    text_emb: np.ndarray,
    tabular: np.ndarray,
    emb_cols: list,
    tab_cols: list,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    device: torch.device,
    force: bool = False,
) -> None:
    """
    Compute SHAP values for ACAGN-Gate using DeepExplainer and save:
      - figures/gate_shap_summary.png
      - results/gate_shap_importance.csv (mean |SHAP| per feature)

    This is optional and can be expensive; controlled by GATE_ENABLE_SHAP.
    """
    try:
        if not (GATE_ENABLE_SHAP or force):
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        os.makedirs(os.path.dirname(GATE_SHAP_IMPORTANCE_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(GATE_SHAP_SUMMARY_PNG), exist_ok=True)

        rng = np.random.RandomState(RANDOM_STATE)

        tr_idx = np.where(train_mask)[0]
        te_idx = np.where(test_mask)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            logger.warning("SHAP skipped: empty train/test split.")
            return

        n_bg = int(min(GATE_SHAP_N_BACKGROUND, len(tr_idx)))
        n_ex = int(min(GATE_SHAP_N_SAMPLES, len(te_idx)))
        bg_sel = rng.choice(tr_idx, n_bg, replace=False)
        ex_sel = rng.choice(te_idx, n_ex, replace=False)

        # SHAP can be fragile on GPU for some setups; run the explainer on CPU.
        shap_device = torch.device("cpu")
        model = model.to(shap_device).eval()

        text_bg = torch.tensor(text_emb[bg_sel], dtype=torch.float32, device=shap_device)
        tab_bg  = torch.tensor(tabular[bg_sel],  dtype=torch.float32, device=shap_device)
        text_ex = torch.tensor(text_emb[ex_sel], dtype=torch.float32, device=shap_device)
        tab_ex  = torch.tensor(tabular[ex_sel],  dtype=torch.float32, device=shap_device)

        wrapped = _GateProbWrapper(model).to(shap_device).eval()
        logger.info("Computing Gate SHAP (background=%d, samples=%d) ...", n_bg, n_ex)

        explainer = shap.DeepExplainer(wrapped, [text_bg, tab_bg])
        shap_vals = explainer.shap_values([text_ex, tab_ex])

        # DeepExplainer may return:
        # - list[input] for single-output models, or
        # - list[output][input] for multi-output models.
        if isinstance(shap_vals, list) and len(shap_vals) > 0 and isinstance(shap_vals[0], list):
            shap_vals = shap_vals[0]

        if not isinstance(shap_vals, list) or len(shap_vals) != 2:
            logger.warning("Unexpected SHAP output format for gate model; skipping plot.")
            return

        sv_text = np.asarray(shap_vals[0])
        sv_tab  = np.asarray(shap_vals[1])
        if sv_text.ndim == 3:
            sv_text = sv_text.squeeze(-1)
        if sv_tab.ndim == 3:
            sv_tab = sv_tab.squeeze(-1)

        X_text = text_emb[ex_sel]
        X_tab  = tabular[ex_sel]

        sv = np.concatenate([sv_text, sv_tab], axis=1)
        X  = np.concatenate([X_text, X_tab], axis=1)
        feature_names = list(emb_cols) + list(tab_cols)

        imp = np.abs(sv).mean(axis=0)
        imp_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
        )
        imp_df.to_csv(GATE_SHAP_IMPORTANCE_CSV, index=False)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X, feature_names=feature_names, max_display=30, show=False)
        plt.tight_layout()
        plt.savefig(GATE_SHAP_SUMMARY_PNG, dpi=150)
        plt.close()
        logger.info("Gate SHAP saved -> %s", GATE_SHAP_SUMMARY_PNG)
    except Exception as e:
        logger.warning("Gate SHAP failed: %s", e)


def _state_dict_from_numpy(state: dict) -> dict:
    return {k: torch.tensor(v) for k, v in state.items()}

def make_splits(groups, labels):
    """
    Patient-level train/val/test split.
    Identical strategy to 03_train.py so results are comparable.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    unique_patients = np.unique(groups)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_test = int(n * TRAIN_TEST_FRAC)
    n_val  = int(n * TRAIN_VAL_FRAC)

    test_pats  = set(unique_patients[-n_test:])
    val_pats   = set(unique_patients[-(n_test + n_val):-n_test])
    train_pats = set(unique_patients[:-(n_test + n_val)])

    train_mask = np.array([g in train_pats for g in groups])
    val_mask   = np.array([g in val_pats   for g in groups])
    test_mask  = np.array([g in test_pats  for g in groups])

    return train_mask, val_mask, test_mask


def run_gate_shap_only() -> None:
    """
    Compute ACAGN-Gate SHAP from a previously saved gate bundle (no training).

    Requires `models/acagn_gate.pkl` to contain `best_seed_state_dict` (added in
    the updated pipeline). If missing, rerun training once to generate and save
    the weights.
    """
    model_path = GATE_MODEL_PKL
    if not os.path.exists(model_path) and os.path.exists(GATE_MODEL_PKL_LEGACY):
        logger.warning(
            "Gate model bundle not found at %s; falling back to legacy path %s",
            model_path,
            GATE_MODEL_PKL_LEGACY,
        )
        model_path = GATE_MODEL_PKL_LEGACY
    if not os.path.exists(model_path):
        logger.error("Gate model bundle not found: %s", model_path)
        return

    bundle = joblib.load(model_path)
    best_state = bundle.get("best_seed_state_dict")
    if best_state is None:
        logger.error(
            "Gate bundle missing best_seed_state_dict. Re-run training once (with the updated code) to save weights."
        )
        return

    text_emb, tabular, labels, groups, hadm_ids, emb_cols, tab_cols = load_fused_data()
    train_mask, _, test_mask = make_splits(groups, labels)

    model = TextGuidedGate(text_emb.shape[1], tabular.shape[1]).to(torch.device("cpu"))
    model.load_state_dict(_state_dict_from_numpy(best_state))
    model.eval()

    compute_gate_shap(
        model,
        text_emb=text_emb,
        tabular=tabular,
        emb_cols=emb_cols,
        tab_cols=tab_cols,
        train_mask=train_mask,
        test_mask=test_mask,
        device=torch.device("cpu"),
        force=True,
    )

# ── Training ──────────────────────────────────────────────────────────────────

def train_one_seed(text_emb, tabular, labels, groups, seed: int, device: torch.device):
    """
    Trains one instance of TextGuidedGate with a given random seed.
    Returns the trained model, validation probabilities, and test probabilities.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_mask, val_mask, test_mask = make_splits(groups, labels)

    pos_weight = (labels[train_mask] == 0).sum() / max((labels[train_mask] == 1).sum(), 1)
    criterion  = nn.BCELoss(weight=None)   # isotonic calibration handles imbalance post-hoc
    # For training we use weighted BCE to handle imbalance
    criterion_train = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    # But our model outputs sigmoid already, so use BCELoss with manual weighting
    # Simpler: just use BCELoss and rely on calibration
    criterion = nn.BCELoss()

    train_ds = ReadmissionDataset(text_emb[train_mask], tabular[train_mask], labels[train_mask])
    val_ds   = ReadmissionDataset(text_emb[val_mask],   tabular[val_mask],   labels[val_mask])
    test_ds  = ReadmissionDataset(text_emb[test_mask],  tabular[test_mask],  labels[test_mask])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

    text_dim    = text_emb.shape[1]
    tabular_dim = tabular.shape[1]
    model       = TextGuidedGate(text_dim, tabular_dim).to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=GATE_LR, weight_decay=1e-5)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GATE_EPOCHS)

    best_val_auroc = 0.0
    best_state     = None
    patience_count = 0

    for epoch in range(GATE_EPOCHS):
        # Training
        model.train()
        for text_b, tab_b, label_b in train_loader:
            text_b, tab_b, label_b = text_b.to(device), tab_b.to(device), label_b.to(device)
            optimizer.zero_grad()
            probs, _ = model(text_b, tab_b)
            loss = criterion(probs, label_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_probs_list = []
        val_labels_list = []
        with torch.no_grad():
            for text_b, tab_b, label_b in val_loader:
                text_b, tab_b = text_b.to(device), tab_b.to(device)
                probs, _ = model(text_b, tab_b)
                val_probs_list.append(probs.cpu().numpy())
                val_labels_list.append(label_b.numpy())

        val_probs  = np.concatenate(val_probs_list)
        val_labels = np.concatenate(val_labels_list)
        val_auroc  = roc_auc_score(val_labels, val_probs)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            logger.info("Seed %d | Epoch %d | Val AUROC: %.4f | Best: %.4f",
                        seed, epoch, val_auroc, best_val_auroc)

        if patience_count >= GATE_PATIENCE:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Get test predictions and gate weights
    test_probs_list  = []
    test_labels_list = []
    gate_weights_list = []

    with torch.no_grad():
        for text_b, tab_b, label_b in test_loader:
            text_b, tab_b = text_b.to(device), tab_b.to(device)
            probs, gates = model(text_b, tab_b)
            test_probs_list.append(probs.cpu().numpy())
            test_labels_list.append(label_b.numpy())
            gate_weights_list.append(gates.cpu().numpy())

    test_probs   = np.concatenate(test_probs_list)
    test_labels  = np.concatenate(test_labels_list)
    gate_weights = np.concatenate(gate_weights_list)

    return model, val_probs, val_labels, test_probs, test_labels, gate_weights, test_mask

# ── ECE ───────────────────────────────────────────────────────────────────────

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, total = 0.0, len(labels)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)

# ── Main Training Entry Point ─────────────────────────────────────────────────

def train_gate_model(
    model_out: str = GATE_MODEL_PKL,
    report_out: Optional[str] = None,
    weights_out: str = GATE_WEIGHTS_NPY,
    ids_out: str = GATE_PATIENT_IDS_NPY,
    overwrite: bool = False,
    save_gate_arrays: bool = True,
):
    """
    Trains ACAGN-Gate across multiple seeds, averages predictions,
    applies isotonic calibration, and saves everything needed for analysis.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if report_out is None:
        report_out = os.path.join(RESULTS_DIR, "gate_training_report.json")

    _ensure_writable(model_out, overwrite=overwrite)
    _ensure_writable(report_out, overwrite=overwrite)
    if save_gate_arrays:
        _ensure_writable(weights_out, overwrite=overwrite)
        _ensure_writable(ids_out, overwrite=overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    text_emb, tabular, labels, groups, hadm_ids, emb_cols, tab_cols = load_fused_data()
    train_mask, _, test_mask = make_splits(groups, labels)

    all_val_probs  = []
    all_test_probs = []
    all_gate_weights = []
    seed_state_dicts = {}
    best_seed = None
    best_seed_state_dict = None
    best_seed_model = None
    best_seed_val_auc = -1.0
    test_labels_ref  = None
    val_labels_ref   = None
    test_mask_ref    = None

    for seed in GATE_SEEDS:
        logger.info("=== Training seed %d ===", seed)
        model, val_probs, val_labels, test_probs, test_labels, gate_weights, test_mask = \
            train_one_seed(text_emb, tabular, labels, groups, seed, device)

        all_val_probs.append(val_probs)
        all_test_probs.append(test_probs)
        all_gate_weights.append(gate_weights)
        seed_state_dicts[int(seed)] = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

        try:
            v_auc = float(roc_auc_score(val_labels, val_probs))
        except Exception:
            v_auc = float("nan")
        if np.isfinite(v_auc) and v_auc > best_seed_val_auc:
            best_seed_val_auc = v_auc
            if best_seed_model is not None:
                del best_seed_model
            best_seed_model = model
            best_seed = int(seed)
            best_seed_state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

        if test_labels_ref is None:
            test_labels_ref = test_labels
            val_labels_ref  = val_labels
            test_mask_ref   = test_mask

        if best_seed_model is not model:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_seed_model is not None:
        compute_gate_shap(
            best_seed_model,
            text_emb=text_emb,
            tabular=tabular,
            emb_cols=emb_cols,
            tab_cols=tab_cols,
            train_mask=train_mask,
            test_mask=test_mask_ref,
            device=device,
        )
        del best_seed_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Average across seeds
    avg_val_probs  = np.mean(all_val_probs,  axis=0)
    avg_test_probs = np.mean(all_test_probs, axis=0)
    avg_gate_weights = np.mean(all_gate_weights, axis=0)

    # Isotonic calibration fitted on val, applied to test
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(avg_val_probs, val_labels_ref)
    cal_test_probs = calibrator.predict(avg_test_probs).astype(np.float32)

    # Metrics
    auroc_raw = roc_auc_score(test_labels_ref, avg_test_probs)
    auroc_cal = roc_auc_score(test_labels_ref, cal_test_probs)
    auprc     = average_precision_score(test_labels_ref, cal_test_probs)
    brier     = brier_score_loss(test_labels_ref, cal_test_probs)
    ece_before = compute_ece(avg_test_probs, test_labels_ref)
    ece_after  = compute_ece(cal_test_probs,  test_labels_ref)

    logger.info("=" * 55)
    logger.info("ACAGN-Gate Results")
    logger.info("  AUROC (raw):        %.4f", auroc_raw)
    logger.info("  AUROC (calibrated): %.4f", auroc_cal)
    logger.info("  AUPRC:              %.4f", auprc)
    logger.info("  Brier score:        %.4f", brier)
    logger.info("  ECE before cal:     %.4f", ece_before)
    logger.info("  ECE after cal:      %.4f", ece_after)
    logger.info("=" * 55)

    # Save gate weights and patient ids for interpretability analysis
    test_hadm_ids = hadm_ids[test_mask_ref]
    if save_gate_arrays:
        np.save(weights_out, avg_gate_weights)
        np.save(ids_out, test_hadm_ids)
        logger.info("Gate weights saved -> %s", weights_out)

    # Save model bundle
    results = {
        "auroc_raw":    round(float(auroc_raw), 4),
        "auroc_cal":    round(float(auroc_cal), 4),
        "auprc":        round(float(auprc),     4),
        "brier":        round(float(brier),     4),
        "ece_before":   round(float(ece_before), 4),
        "ece_after":    round(float(ece_after),  4),
        "text_features": emb_cols,
        "tab_features": tab_cols,
        "n_test":       int(len(test_labels_ref)),
        "seeds":        GATE_SEEDS,
    }

    joblib.dump({
        "calibrator":      calibrator,
        "tab_cols":        tab_cols,
        "emb_cols":        emb_cols,
        "text_dim":        text_emb.shape[1],
        "tabular_dim":     tabular.shape[1],
        "results":         results,
        "test_probs_raw":  avg_test_probs,
        "test_probs_cal":  cal_test_probs,
        "test_labels":     test_labels_ref,
        "test_hadm_ids":   test_hadm_ids,
        "avg_gate_weights": avg_gate_weights,
        "best_seed":        best_seed,
        "best_seed_state_dict": best_seed_state_dict,
        "seed_state_dicts": seed_state_dicts,
        "gate_hidden_dim": int(GATE_HIDDEN_DIM),
        "gate_dropout": float(GATE_DROPOUT),
    }, model_out)

    with open(report_out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Gate model saved -> %s", model_out)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-only", action="store_true", help="Compute Gate SHAP from saved bundle (no training).")
    parser.add_argument("--model-out", default=GATE_MODEL_PKL, help="Path to write the gate model bundle (.pkl).")
    parser.add_argument(
        "--report-out",
        default=None,
        help="Path to write the gate JSON metric report (default: results/gate_training_report.json).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument(
        "--no-save-gate-arrays",
        action="store_true",
        help="Do not write gate_weights.npy / gate_patient_ids.npy (useful for inference-only bundles).",
    )
    args = parser.parse_args()

    if args.shap_only:
        run_gate_shap_only()
    else:
        train_gate_model(
            model_out=args.model_out,
            report_out=args.report_out,
            overwrite=args.overwrite,
            save_gate_arrays=not args.no_save_gate_arrays,
        )
