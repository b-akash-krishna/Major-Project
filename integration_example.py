"""
ACAGN - Prediction Model Integration Guide
=====================================================

This script serves as a complete guide for integrating the ACAGN Readmission Prediction model
into external systems. It demonstrates:
1. What input features are strictly required from the user/external system.
2. How to handle missing secondary features (using baselines/defaults).
3. How to process unstructured clinical notes into embeddings (ClinicalT5).
4. How to load the pre-trained ensemble model and generate a prediction.
5. How to interpret the model's output probabilities and risk tiers.

--- Core Requirements for Integration ---
- The model expects exactly 259 features in a specific order.
- To avoid passing 259 fields manually, this script uses a `DEFAULTS` dictionary for the ~253
  secondary features (like rare lab tests, specific medications) while taking the 6 core features
  from the user.
- If you have real data for secondary features (e.g., 'lab_hematocrit_min'), you simply overwrite
  the default value before predicting.
"""

import sys
import os
import pandas as pd
import warnings
import torch

# Fix CVE-2025-32434 PyTorch vulnerability lockout when loading Text embeddings
torch.serialization.add_safe_globals([
    'torch.nn.modules.module.Module',
    'torch._utils._rebuild_tensor_v2', 
    'torch._utils._rebuild_parameter'
])

# Suppress verbose PyTorch/HuggingFace warnings during ClinicalT5 model loading
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the 'src' directory to the Python path so we can import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import ACAGN utilities
from config import DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK
from embedding_utils import get_embedding, get_model_container

def get_user_inputs():
    """
    Collect the 6 core features required for a baseline prediction.
    In a real system (like an EHR integration), these would be queried directly from the database.
    
    Feature Explanations:
    - anchor_age: Patient's age at admission (Integer, typically 18-120)
    - gender: 0 for Female, 1 for Male (Integer)
    - los_days: Length of stay in the hospital so far, in days (Float)
    - admission_type: 1=Emergency, 2=Urgent, 3=Planned (Integer)
    - prev_admissions: Count of previous hospital admissions for this patient (Integer)
    - clinical_note: The raw text from discharge summaries, nursing notes, or radiology reports.
                     Used to generate 128 text embedding features using ClinicalT5.
    """
    print("\n--- CLINICAL INPUT GATHERING ---")
    print("Please provide the following patient details:")
    
    try:
        age_input = input("1. Patient Age (e.g., 65): ")
        anchor_age = int(age_input) if age_input.strip() else 65
        
        gender_input = input("2. Gender (0=Female, 1=Male) [Default=0]: ")
        gender = int(gender_input) if gender_input.strip() else 0
        
        los_input = input("3. Length of stay in days (e.g., 5.5): ")
        los_days = float(los_input) if los_input.strip() else 5.5
        
        adm_input = input("4. Admission Type (1=Emergency, 2=Urgent, 3=Planned) [Default=1]: ")
        admission_type = int(adm_input) if adm_input.strip() else 1
        
        prev_input = input("5. Number of previous hospital admissions [Default=0]: ")
        prev_admissions = int(prev_input) if prev_input.strip() else 0
        
        # Optional high-impact secondary EHR features
        print("\n--- OPTIONAL SECONDARY EHR FEATURES (Press Enter to use Dataset Median Defaults) ---")
        
        sec_days = input(" > Days since last hospital discharge [Default=0]: ")
        days_since_last = float(sec_days) if sec_days.strip() else 0.0

        sec_abnormal = input(" > Abnormal lab ratio (0.0 to 1.0) [Default=0.0]: ")
        lab_abnormal_rate = float(sec_abnormal) if sec_abnormal.strip() else 0.0

        sec_meds = input(" > Medication orders count [Default=0]: ")
        rx_count = int(sec_meds) if sec_meds.strip() else 0

        sec_icu = input(" > Number of ICU stays this visit [Default=0]: ")
        icu_stays_count = int(sec_icu) if sec_icu.strip() else 0

        sec_proc = input(" > Number of procedures this stay [Default=0]: ")
        proc_count = int(sec_proc) if sec_proc.strip() else 0

        sec_readmit = input(" > Past readmission rate (0.0 to 1.0) [Default=0.0]: ")
        prev_readmit_rate = float(sec_readmit) if sec_readmit.strip() else 0.0
        
        print("\n6. Clinical Note snippet (e.g., 'Patient admitted for acute heart failure...'):")
        clinical_note = input("> ")
        
        return {
            "anchor_age": anchor_age,
            "gender": gender,
            "los_days": los_days,
            "admission_type": admission_type,
            "prev_admissions": prev_admissions,
            "days_since_last": days_since_last,
            "lab_abnormal_rate": lab_abnormal_rate,
            "rx_count": rx_count,
            "icu_stays_count": icu_stays_count,
            "proc_count": proc_count,
            "prev_readmit_rate": prev_readmit_rate,
            "clinical_note": clinical_note
        }
    except ValueError:
        print("\n[!] Invalid input detected. Please enter numbers where requested.")
        sys.exit(1)


def build_full_feature_vector(user_data):
    """
    Combines core inputs, derived mathematical features, defaults, and text embeddings
    into the full 259-feature dictionary required by the model.
    """
    print("\n--- FEATURE PROCESSING ---")
    
    # 1. Start with the defaults to ensure no expected model features are missing (NaNs)
    # DEFAULTS contain median/mode values observed in the training data for all 250+ features.
    full_features = DEFAULTS.copy()
    
    # 2. Overwrite defaults with explicitly known patient data
    full_features["anchor_age"] = user_data["anchor_age"]
    full_features["gender"] = user_data["gender"]
    full_features["los_days"] = user_data["los_days"]
    full_features["admission_type"] = user_data["admission_type"]
    full_features["prev_admissions"] = user_data["prev_admissions"]
    
    # Map high-priority secondary features
    full_features["days_since_last"] = user_data["days_since_last"]
    full_features["lab_abnormal_rate"] = user_data["lab_abnormal_rate"]
    full_features["rx_count"] = user_data["rx_count"]
    full_features["icu_stays_count"] = user_data["icu_stays_count"]
    full_features["proc_count"] = user_data["proc_count"]
    full_features["prev_readmit_rate"] = user_data["prev_readmit_rate"]
    
    # 3. Compute simple derived features
    full_features["los_hours"] = user_data["los_days"] * 24.0
    full_features["proc_per_day"] = user_data["proc_count"] / max(1.0, user_data["los_days"])
    full_features["med_per_day"] = user_data["rx_count"] / max(1.0, user_data["los_days"])
    
    # 4. Generate Text Embeddings for the unstructured note
    # Extracted text is passed through the pre-trained ClinicalT5 transformer model.
    # It outputs an array of 128 numerical values (ct5_0 to ct5_127).
    print("Generating ClinicalT5 embeddings from text... (this may take a moment)")
    embeddings = get_embedding(
        text=user_data["clinical_note"], 
        features=full_features  # Context features used if text is empty
    )
    
    # Append the 128 embedding dimensions to our feature dictionary
    for i, val in enumerate(embeddings):
        full_features[f"ct5_{i}"] = float(val)
        
    print("Feature vector successfully constructed.")
    return full_features


def run_prediction():
    """Main execution flow for prediction integration."""
    
    print("==================================================")
    print("  ACAGN INTEGRATION GUIDE - PREDICTION PIPELINE  ")
    print("==================================================")
    
    # Step A: Collect data (User input / Database Query)
    user_inputs = get_user_inputs()
    
    # Step B: Load the trained Meta-Learner Model
    # `get_model_container()` initializes the singleton class that loads `models/acagn_framework.pkl`
    # (legacy path `models/trance_framework.pkl` is supported as a fallback).
    # It contains LightGBM, XGBoost, and the isotonic calibration wrapper.
    print("\nLoading ACAGN ensemble model from disk...")
    model_container = get_model_container()
    
    if not model_container.model_data:
        print("[!] Error: Model failed to load. Have you run 'python src/03_train.py' yet?")
        sys.exit(1)
        
    # The container securely holds the exact list of 259 features the model was trained on
    ordered_feature_names = model_container.model_data["features"]
    
    # Step C: Preprocess the input into the 259-dimensional vector
    full_feature_dict = build_full_feature_vector(user_inputs)
    
    # Convert dictionary into a Pandas DataFrame row with exact column ordering
    input_row = {feat: full_feature_dict.get(feat, 0) for feat in ordered_feature_names}
    X_input = pd.DataFrame([input_row])[ordered_feature_names]
    
    # Step D: Execute Model Inference
    print("\nRunning ensemble inference...")
    
    # .predict_proba() yields array of [Prob_Class_0, Prob_Class_1]
    # We want index 1: The probability of a 30-day readmission
    readmission_prob = float(model_container.predict_proba(X_input)[0])
    
    # Step E: Interpret Outputs
    # We use scientifically calibrated thresholds to classify the risk tier
    if readmission_prob >= THRESHOLD_HIGH_RISK:   # Usually 0.70 (70%)
        risk_tier = "HIGH"
    elif readmission_prob >= THRESHOLD_MEDIUM_RISK: # Usually 0.40 (40%)
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"
        
    print("\n==================================================")
    print("                 PREDICTION RESULTS                 ")
    print("==================================================")
    print(f"Risk Probability : {readmission_prob * 100:.2f}% chance of 30-day readmission")
    print(f"Risk Tier        : {risk_tier} RISK")
    print("==================================================")
    print("\nIntegration output definitions:")
    print(" - Probability: Calibrated likelihood (0.0 to 1.0) of patient returning within 30 days.")
    print(" - Tier: Categorical bucket ideal for triggering clinical workflows (e.g. HIGH -> Consult social worker).")
    print("\nTo integrate this logic, copy the steps from this script (Steps A through E).")

if __name__ == "__main__":
    run_prediction()
