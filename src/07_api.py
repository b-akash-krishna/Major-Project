# src/07_api.py
"""
ACAGN Readmission Prediction API — FastAPI v2 compatible.
Run as:  uvicorn src.07_api:app --host 0.0.0.0 --port 8000
      OR python src/07_api.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Annotated

# ── Import fix: works both as package module and direct script ─────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import (
        MAIN_MODEL_PKL, API_HOST, API_PORT, CORS_ORIGINS,
        DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK,
        EMBEDDING_DIM,
    )
    from embedding_utils import get_embedding, get_model_container
    from hybrid_predictor import GatePredictor, hybrid_combine
except ImportError:
    from .config import (
        MAIN_MODEL_PKL, API_HOST, API_PORT, CORS_ORIGINS,
        DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK,
        EMBEDDING_DIM,
    )
    from .embedding_utils import get_embedding, get_model_container
    from .hybrid_predictor import GatePredictor, hybrid_combine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ACAGN Readmission Prediction API",
    description=(
        "30-day hospital readmission risk prediction using "
        "LightGBM + XGBoost ensemble with clinical note embeddings."
    ),
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup (singleton)
model_container = get_model_container()
_gate_predictor = None


# ── Request schema (Pydantic v2 compatible) ───────────────────────────────────
class PatientInput(BaseModel):
    """Input features for a single patient admission."""

    # ── FIX: Pydantic v2 dropped `example` kwarg from Field().
    #    Use json_schema_extra on the model instead, or Annotated[..., Field(...)].
    anchor_age:       int   = Field(..., ge=18, le=120,
                                    description="Patient age at admission")
    gender:           int   = Field(..., ge=0,  le=1,
                                    description="0=Female, 1=Male")
    los_days:         float = Field(..., ge=0,
                                    description="Length of stay in days")
    admission_type:   int   = Field(1,
                                    description="Admission type code (1=emergency)")
    prev_admissions:  int   = Field(0, ge=0,
                                    description="Number of prior admissions")
    clinical_note:    Optional[str] = Field(
        None,
        description="Discharge/clinical note text for embedding generation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "anchor_age": 65,
                "gender": 1,
                "los_days": 5.5,
                "admission_type": 1,
                "prev_admissions": 2,
                "clinical_note": "Patient has a history of heart failure...",
            }
        }
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_feature_row(input_data: PatientInput) -> dict:
    """
    Merges user input with DEFAULTS, computes derived fields,
    and fills in ClinicalT5 embeddings (ct5_0 … ct5_127).
    """
    data = input_data.model_dump()
    data["los_hours"] = data["los_days"] * 24

    # Start from DEFAULTS so every feature the model expects is present
    full = {**DEFAULTS, **data}

    note_text = data.get("clinical_note") or ""

    # Generate text embeddings
    emb = get_embedding(text=note_text, features=full)
    for i, val in enumerate(emb):
        full[f"ct5_{i}"] = float(val)

    # Metadata features expected by ACAGN models trained with embeddings.csv
    full["ct5_has_note"] = 1 if str(note_text).strip() else 0
    full["ct5_note_len_chars"] = int(len(str(note_text)))
    full["ct5_note_len_tokens"] = int(len(str(note_text).split()))

    return full


def _get_gate_predictor() -> GatePredictor:
    global _gate_predictor
    if _gate_predictor is None:
        _gate_predictor = GatePredictor()
    return _gate_predictor


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", summary="Predict 30-day readmission risk")
async def predict(input_data: PatientInput, model: str = "base"):
    """
    Returns readmission probability and risk tier (High / Medium / Low).

    The full ensemble (LightGBM + XGBoost + isotonic calibration) is used,
    matching the trained model exactly.
    """
    if not model_container.model_data:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        full = _build_feature_row(input_data)

        features = model_container.model_data["features"]

        # Build DataFrame row — fill any still-missing columns with 0
        row = {f: full.get(f, 0) for f in features}
        X   = pd.DataFrame([row])[features]

        model_key = (model or "base").strip().lower()

        # Base ensemble probability (calibrated)
        p_base = float(model_container.predict_proba(X)[0])

        if model_key == "base":
            proba = p_base
            model_used = "ACAGN-Base"
        elif model_key == "gate":
            p_gate = float(_get_gate_predictor().predict_proba_from_full(full))
            proba = p_gate
            model_used = "ACAGN-Gate"
        elif model_key == "hybrid":
            p_gate = float(_get_gate_predictor().predict_proba_from_full(full))
            proba = hybrid_combine(p_base=p_base, p_gate=p_gate, w_base=0.5)
            model_used = "ACAGN-Hybrid"
        else:
            raise HTTPException(status_code=400, detail="Invalid model. Use: base | gate | hybrid")

        if proba >= THRESHOLD_HIGH_RISK:
            risk = "High"
        elif proba >= THRESHOLD_MEDIUM_RISK:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "prediction":  f"{risk} Risk",
            "probability": round(proba, 4),
            "risk_level":  risk,
            "confidence":  round(max(proba, 1 - proba), 4),
            "model":       model_used,
            "status":      "success",
        }

    except HTTPException:
        raise
    except (FileNotFoundError, RuntimeError) as e:
        logger.exception("Prediction dependency error")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/health", summary="Health check")
async def health():
    models = model_container.model_data.get("models", []) if model_container.model_data else []
    return {
        "status":        "healthy",
        "model_loaded":  model_container.model_data is not None,
        "n_models":      len(models),
        "n_features":    len(model_container.model_data.get("features", []))
                         if model_container.model_data else 0,
    }


@app.get("/features", summary="List model features")
async def list_features():
    if not model_container.model_data:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "features":   model_container.model_data.get("features", []),
        "n_features": len(model_container.model_data.get("features", [])),
    }


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
