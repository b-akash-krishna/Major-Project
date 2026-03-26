"""
Concat-MLP baseline (no gating)
==============================

Purpose:
  Match TRANCE-Gate (aka ACAGN-Gate) as closely as possible, but remove the
  gating mechanism. Instead of gating tabular features with a text-derived gate
  vector, this baseline directly concatenates:

    fused = concat(text_embedding, tabular_features)

  and feeds fused into the SAME MLP head used in the gated model.

Outputs are written to separate files so existing gate models/results remain
untouched unless --overwrite is explicitly provided.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

try:
    from config import (
        CONCAT_MLP_MODEL_PKL,
        CONCAT_MLP_REPORT_JSON,
        GATE_DROPOUT,
        GATE_EPOCHS,
        GATE_LR,
        GATE_PATIENCE,
        GATE_SEEDS,
        RESULTS_DIR,
    )
    from gated_fusion_model import ReadmissionDataset, compute_ece, load_fused_data, make_splits
except ImportError:
    from .config import (
        CONCAT_MLP_MODEL_PKL,
        CONCAT_MLP_REPORT_JSON,
        GATE_DROPOUT,
        GATE_EPOCHS,
        GATE_LR,
        GATE_PATIENCE,
        GATE_SEEDS,
        RESULTS_DIR,
    )
    from .gated_fusion_model import ReadmissionDataset, compute_ece, load_fused_data, make_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ConcatMLP(nn.Module):
    """
    Baseline: concat(text_emb, tabular) -> MLP -> readmission prob

    The MLP head is intentionally identical to TextGuidedGate.classifier.
    """

    def __init__(self, text_dim: int, tabular_dim: int, dropout: float = GATE_DROPOUT):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + tabular_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_emb: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        x_fused = torch.cat([text_emb, x_tab], dim=1)
        prob = self.classifier(x_fused).squeeze(1)
        return prob


def _ensure_writable(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {path} (pass --overwrite to replace)")


def train_one_seed_concat(
    text_emb: np.ndarray,
    tabular: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train one Concat-MLP instance with a given random seed.
    Mirrors src/gated_fusion_model.py:train_one_seed as closely as possible.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_mask, val_mask, test_mask = make_splits(groups, labels)

    criterion = nn.BCELoss()

    train_ds = ReadmissionDataset(text_emb[train_mask], tabular[train_mask], labels[train_mask])
    val_ds = ReadmissionDataset(text_emb[val_mask], tabular[val_mask], labels[val_mask])
    test_ds = ReadmissionDataset(text_emb[test_mask], tabular[test_mask], labels[test_mask])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    text_dim = text_emb.shape[1]
    tabular_dim = tabular.shape[1]
    model = ConcatMLP(text_dim, tabular_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GATE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GATE_EPOCHS)

    best_val_auroc = 0.0
    best_state: Optional[dict] = None
    patience_count = 0

    for epoch in range(GATE_EPOCHS):
        model.train()
        for text_b, tab_b, label_b in train_loader:
            text_b, tab_b, label_b = text_b.to(device), tab_b.to(device), label_b.to(device)
            optimizer.zero_grad()
            probs = model(text_b, tab_b)
            loss = criterion(probs, label_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_probs_list = []
        val_labels_list = []
        with torch.no_grad():
            for text_b, tab_b, label_b in val_loader:
                text_b, tab_b = text_b.to(device), tab_b.to(device)
                probs = model(text_b, tab_b)
                val_probs_list.append(probs.cpu().numpy())
                val_labels_list.append(label_b.numpy())

        val_probs = np.concatenate(val_probs_list)
        val_labels = np.concatenate(val_labels_list)
        val_auroc = roc_auc_score(val_labels, val_probs)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            logger.info(
                "Seed %d | Epoch %d | Val AUROC: %.4f | Best: %.4f",
                seed,
                epoch,
                val_auroc,
                best_val_auroc,
            )

        if patience_count >= GATE_PATIENCE:
            logger.info("Early stopping at epoch %d", epoch)
            break

    if best_state is None:
        raise RuntimeError("No best_state captured during training; check labels/val split.")

    model.load_state_dict(best_state)
    model.eval()

    test_probs_list = []
    test_labels_list = []
    with torch.no_grad():
        for text_b, tab_b, label_b in test_loader:
            text_b, tab_b = text_b.to(device), tab_b.to(device)
            probs = model(text_b, tab_b)
            test_probs_list.append(probs.cpu().numpy())
            test_labels_list.append(label_b.numpy())

    test_probs = np.concatenate(test_probs_list)
    test_labels = np.concatenate(test_labels_list)
    return model, val_probs, val_labels, test_probs, test_labels, test_mask


def train_concat_mlp_model(
    model_out: str = CONCAT_MLP_MODEL_PKL,
    report_out: str = CONCAT_MLP_REPORT_JSON,
    overwrite: bool = False,
) -> dict:
    """
    Multi-seed training + ensembling + isotonic calibration, on the exact same
    train/val/test split strategy used by TRANCE-Gate.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _ensure_writable(model_out, overwrite=overwrite)
    _ensure_writable(report_out, overwrite=overwrite)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    text_emb, tabular, labels, groups, hadm_ids, emb_cols, tab_cols = load_fused_data()

    all_val_probs = []
    all_test_probs = []
    best_seed = None
    best_seed_state_dict = None
    best_seed_model = None
    best_seed_val_auc = -1.0

    test_labels_ref = None
    val_labels_ref = None
    test_mask_ref = None

    for seed in GATE_SEEDS:
        logger.info("=== Training seed %d ===", seed)
        model, val_probs, val_labels, test_probs, test_labels, test_mask = train_one_seed_concat(
            text_emb, tabular, labels, groups, seed, device
        )

        all_val_probs.append(val_probs)
        all_test_probs.append(test_probs)

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
            val_labels_ref = val_labels
            test_mask_ref = test_mask

        if best_seed_model is not model:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if test_labels_ref is None or val_labels_ref is None or test_mask_ref is None:
        raise RuntimeError("Failed to capture reference labels/mask.")

    if best_seed_model is not None:
        del best_seed_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_val_probs = np.mean(all_val_probs, axis=0)
    avg_test_probs = np.mean(all_test_probs, axis=0)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(avg_val_probs, val_labels_ref)
    cal_test_probs = calibrator.predict(avg_test_probs).astype(np.float32)

    auroc_raw = roc_auc_score(test_labels_ref, avg_test_probs)
    auroc_cal = roc_auc_score(test_labels_ref, cal_test_probs)
    auprc = average_precision_score(test_labels_ref, cal_test_probs)
    brier = brier_score_loss(test_labels_ref, cal_test_probs)
    ece_before = compute_ece(avg_test_probs, test_labels_ref)
    ece_after = compute_ece(cal_test_probs, test_labels_ref)

    logger.info("=" * 55)
    logger.info("Concat-MLP Results (no gating)")
    logger.info("  AUROC (raw):        %.4f", auroc_raw)
    logger.info("  AUROC (calibrated): %.4f", auroc_cal)
    logger.info("  AUPRC:              %.4f", auprc)
    logger.info("  Brier score:        %.4f", brier)
    logger.info("  ECE before cal:     %.4f", ece_before)
    logger.info("  ECE after cal:      %.4f", ece_after)
    logger.info("=" * 55)

    test_hadm_ids = hadm_ids[test_mask_ref]
    results = {
        "auroc_raw": round(float(auroc_raw), 4),
        "auroc_cal": round(float(auroc_cal), 4),
        "auprc": round(float(auprc), 4),
        "brier": round(float(brier), 4),
        "ece_before": round(float(ece_before), 4),
        "ece_after": round(float(ece_after), 4),
        "text_features": emb_cols,
        "tab_features": tab_cols,
        "n_test": int(len(test_labels_ref)),
        "seeds": list(GATE_SEEDS),
        "note": "Baseline identical to TRANCE-Gate MLP head; no text-derived gating (direct concatenation).",
    }

    joblib.dump(
        {
            "calibrator": calibrator,
            "tab_cols": tab_cols,
            "emb_cols": emb_cols,
            "text_dim": int(text_emb.shape[1]),
            "tabular_dim": int(tabular.shape[1]),
            "results": results,
            "test_probs_raw": avg_test_probs,
            "test_probs_cal": cal_test_probs,
            "test_labels": test_labels_ref,
            "test_hadm_ids": test_hadm_ids,
            "best_seed": best_seed,
            "best_seed_state_dict": best_seed_state_dict,
        },
        model_out,
    )

    with open(report_out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Concat-MLP model saved -> %s", model_out)
    logger.info("Concat-MLP report saved -> %s", report_out)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument("--model-out", default=CONCAT_MLP_MODEL_PKL, help="Path to write the model bundle (.pkl).")
    parser.add_argument(
        "--report-out", default=CONCAT_MLP_REPORT_JSON, help="Path to write the JSON metric report."
    )
    args = parser.parse_args()

    train_concat_mlp_model(model_out=args.model_out, report_out=args.report_out, overwrite=args.overwrite)

