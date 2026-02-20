# src/07_api.py
"""
TRANCE Readmission Prediction API
Production-grade FastAPI implementation with configuration-driven settings.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Add current dir to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    MAIN_MODEL_PKL, API_HOST, API_PORT, CORS_ORIGINS, 
    DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK
)
from embedding_utils import get_embedding, get_model_container

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TRANCE Readmission Prediction API",
    description="Industry-grade hospital readmission risk prediction using BERT/T5 and EHR features.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_container = get_model_container()

class PatientInput(BaseModel):
    anchor_age: int = Field(..., ge=18, le=120, example=65)
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male", example=1)
    los_days: float = Field(..., ge=0, example=5.5)
    admission_type: int = Field(1, description="Categorical code for admission type", example=1)
    prev_admissions: int = Field(0, example=2)
    clinical_note: Optional[str] = Field(None, example="Patient has a history of heart failure...")

@app.post("/predict")
async def predict(input_data: PatientInput):
    """Predicts 30-day readmission risk levels"""
    if not model_container.model_data:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        # Pre-processing
        data = input_data.dict()
        data['los_hours'] = data['los_days'] * 24
        
        # Merge with defaults for calculated features
        full_data = {**DEFAULTS, **data}
        
        # Embedding Generation
        emb = get_embedding(text=input_data.clinical_note, features=full_data)
        for i, val in enumerate(emb):
            full_data[f"ct5_{i}"] = val
            
        # Inference
        features = model_container.model_data['features']
        X = pd.DataFrame([full_data])[features]
        
        proba = model_container.model_data['model'].predict_proba(X)[0, 1]
        
        # Threshold logic
        risk = "High" if proba > THRESHOLD_HIGH_RISK else "Medium" if proba > THRESHOLD_MEDIUM_RISK else "Low"
        
        return {
            "prediction": f"{risk} Risk",
            "probability": float(proba),
            "risk_level": risk,
            "confidence": float(max(proba, 1-proba)),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_container.model_data is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)