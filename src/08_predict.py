# src/08_predict.py
"""
TRANCE Framework — Interactive Prediction CLI.
Run as:  python src/08_predict.py   OR   python -m src.08_predict

Embedding method matches training exactly:
  - sentence-transformers/all-mpnet-base-v2 -> 768-dim mean pool -> PCA(128)
  - Reads embedding_info.pkl for stored PCA (fitted during 02_embed.py)
  - Falls back to ClinicalT5 PyTorch conversion if embedding_info says so
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

# ── Import fix ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config import DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK
    from embedding_utils import get_embedding, get_model_container
except ImportError:
    from .config import DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK
    from .embedding_utils import get_embedding, get_model_container

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Load model once at import time
model_container = get_model_container()


# ── User input ────────────────────────────────────────────────────────────────

def get_user_input() -> dict:
    """Interactive CLI prompt for patient data."""
    print("\n" + "=" * 52)
    print("  TRANCE RISK PREDICTION — INPUT TERMINAL")
    print("=" * 52)

    try:
        age    = int(input("  Age [18-120]:                      ") or 65)
        gender = int(input("  Gender (0=Female, 1=Male) [0]:     ") or 0)
        los    = float(input("  Length of Stay in days [3]:        ") or 3.0)
        prev   = int(input("  Previous Admissions [0]:           ") or 0)
        adm_t  = int(input("  Admission type (1=Emer,2=Elec) [1]:") or 1)
        note   = input("  Clinical Note (optional, press Enter to skip):\n  > ")

        return {
            "anchor_age":      age,
            "gender":          gender,
            "los_days":        los,
            "los_hours":       los * 24,
            "prev_admissions": prev,
            "admission_type":  adm_t,
            "clinical_note":   note.strip() if note.strip() else None,
        }
    except (ValueError, KeyboardInterrupt):
        print("\n  Invalid input or cancelled.")
        return {}


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(data: dict) -> None:
    """
    Runs the full ensemble prediction for one patient.

    Steps:
      1. Merge user input with DEFAULTS (covers all 288 model features)
      2. Generate 128-dim text embedding via embedding_utils (matches training)
      3. Fill ct5_0 … ct5_127 into the feature row
      4. Select model feature columns in correct order
      5. Full ensemble predict_proba with calibration
    """
    if not model_container.model_data:
        print("  Model not loaded. Run 03_train.py first.")
        return

    # ── Step 1: Build full feature dict ──────────────────────────────────────
    full = {**DEFAULTS, **data}
    full["los_hours"] = data.get("los_days", 3.0) * 24

    # ── Step 2+3: Generate and inject embeddings ──────────────────────────────
    clinical_note = data.get("clinical_note")
    emb = get_embedding(text=clinical_note, features=full)   # 128-dim np.ndarray

    if clinical_note:
        print(f"  Embedding generated from clinical note ({len(clinical_note)} chars).")
    else:
        print("  No clinical note — using zero embeddings for ct5_* features.")

    for i, val in enumerate(emb):
        full[f"ct5_{i}"] = float(val)

    # ── Step 4: Align feature columns with model ──────────────────────────────
    features = model_container.model_data["features"]
    missing  = [f for f in features if f not in full]
    if missing:
        print(f"  Note: {len(missing)} features defaulted to 0: {missing[:5]} ...")
    row = {f: full.get(f, 0) for f in features}
    X   = pd.DataFrame([row])[features]

    # ── Step 5: Full ensemble prediction ─────────────────────────────────────
    try:
        proba = float(model_container.predict_proba(X)[0])
    except Exception as e:
        print(f"  Prediction error: {e}")
        return

    if proba >= THRESHOLD_HIGH_RISK:
        risk  = "HIGH"
        color = "!!!"
    elif proba >= THRESHOLD_MEDIUM_RISK:
        risk  = "MEDIUM"
        color = "??"
    else:
        risk  = "LOW"
        color = "OK"

    print("\n" + "*" * 52)
    print(f"  RESULT: {color} {risk} RISK {color}")
    print(f"  30-day readmission probability: {proba:.1%}")
    print(f"  Threshold: High>={THRESHOLD_HIGH_RISK:.0%}  Medium>={THRESHOLD_MEDIUM_RISK:.0%}")
    print("*" * 52 + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not model_container.model_data:
        print("ERROR: Model not loaded. Ensure models/trance_framework.pkl exists.")
        sys.exit(1)

    print(f"\nModel loaded: {len(model_container.model_data.get('features', []))} features"
          f" | {len(model_container.model_data.get('models', []))} ensemble members")

    while True:
        data = get_user_input()
        if data:
            run_inference(data)

        try:
            cont = input("  Predict another patient? (y/n) [y]: ").strip().lower()
        except KeyboardInterrupt:
            break
        if cont == "n":
            break

    print("  Goodbye.")