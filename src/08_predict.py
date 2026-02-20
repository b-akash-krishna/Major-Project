# src/08_predict.py
"""
TRANCE Framework - Interactive Prediction CLI
Refactored for UX and modularity.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULTS
from embedding_utils import get_embedding, get_model_container

model_container = get_model_container()

logging.basicConfig(level=logging.WARNING)

def get_user_input():
    """Interactive CLI prompt for patient data"""
    print("\n" + "="*50)
    print("  TRANCE RISK PREDICTION - INPUT TERMINAL")
    print("="*50)
    
    try:
        age = int(input("  Age [18-120]: ") or 65)
        gender = int(input("  Gender (0=F, 1=M) [0]: ") or 0)
        los = float(input("  Length of Stay (Days) [3]: ") or 3.0)
        prev = int(input("  Previous Admissions [0]: ") or 0)
        note = input("  Clinical Note (optional): ")
        
        return {
            "anchor_age": age,
            "gender": gender,
            "los_days": los,
            "prev_admissions": prev,
            "clinical_note": note
        }
    except ValueError:
        print("  ❌ Invalid input. Using defaults.")
        return None

def run_inference(data):
    """Executes the prediction logic locally"""
    if not model_container.model_data:
        print("  ❌ Model not loaded. Run training first.")
        return

    # Prep features
    full_data = {**DEFAULTS, **data}
    full_data['los_hours'] = data['los_days'] * 24
    
    # Embedding
    emb = get_embedding(text=data['clinical_note'], features=full_data)
    for i, val in enumerate(emb):
        full_data[f"ct5_{i}"] = val
        
    # Predict
    features = model_container.model_data['features']
    X = pd.DataFrame([full_data])[features]
    proba = model_container.model_data['model'].predict_proba(X)[0, 1]
    
    risk = "High" if proba > 0.5 else "Medium" if proba > 0.3 else "Low"
    
    print("\n" + "*"*50)
    print(f"  RESULT: {risk} Risk Prediction")
    print(f"  Probability: {proba:.2%}")
    print("*"*50 + "\n")

if __name__ == "__main__":
    if not model_container.model_data:
        model_container.load_model()
        
    while True:
        data = get_user_input()
        if data:
            run_inference(data)
        
        cont = input("  Predict another? (y/n) [y]: ")
        if cont.lower() == 'n':
            break