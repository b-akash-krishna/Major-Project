import os
import joblib
import pandas as pd
import numpy as np

from src.config import RANDOM_STATE, TRAIN_VAL_FRAC, TRAIN_TEST_FRAC, FEATURES_CSV, MAIN_MODEL_PKL, EMBEDDINGS_CSV
import src.gated_fusion_model

np.random.seed(RANDOM_STATE)

def get_masks(groups):
    rng = np.random.RandomState(RANDOM_STATE)
    unique_patients = np.unique(groups)
    rng.shuffle(unique_patients)
    n = len(unique_patients)
    n_test = int(n * TRAIN_TEST_FRAC)
    n_val = int(n * TRAIN_VAL_FRAC)
    
    test_pats = set(unique_patients[-n_test:])
    val_pats = set(unique_patients[-(n_test + n_val):-n_test])
    
    return np.array([g in val_pats for g in groups]), np.array([g in test_pats for g in groups])

def main():
    print("Loading original feature and embeddings data to reconstruct the split...")
    pruned = FEATURES_CSV.replace(".csv", "_pruned.csv")
    path = pruned if os.path.exists(pruned) else FEATURES_CSV
    tab = pd.read_csv(path, low_memory=False).fillna(0)
    
    if os.path.exists(EMBEDDINGS_CSV):
        emb = pd.read_csv(EMBEDDINGS_CSV, low_memory=False)
        df = tab.merge(emb, on="hadm_id", how="left").fillna(0)
    else:
        df = tab.copy()

    groups = df["subject_id"].astype(int)
    y = df["readmit_30"].astype("int8")
    
    val_mask, test_mask = get_masks(groups.values)
    
    # Extract metadata blocks
    val_meta = df[val_mask][["hadm_id", "subject_id", "anchor_age", "gender", "race_enc"]].copy()
    test_meta = df[test_mask][["hadm_id", "subject_id", "anchor_age", "gender", "race_enc"]].copy()
    
    val_meta.to_csv("results/val_meta.csv", index=False)
    test_meta.to_csv("results/test_meta.csv", index=False)
    
    val_labels = df[val_mask][["readmit_30"]].rename(columns={"readmit_30":"y_true"})
    test_labels = df[test_mask][["readmit_30"]].rename(columns={"readmit_30":"y_true"})
    
    val_labels.to_csv("results/val_labels.csv", index=False)
    test_labels.to_csv("results/test_labels.csv", index=False)

    print("Loading frozen ACAGN framework model...")
    state = joblib.load(MAIN_MODEL_PKL)
    features = state["features"]
    models = state["models"]
    meta = state.get("meta")
    calibrator = state["calibrator"]
    
    X_val = df[val_mask][features].values.astype(np.float32)
    X_test = df[test_mask][features].values.astype(np.float32)
    
    print("Re-evaluating base models to reconstruct frozen probabilities...")
    val_stack = np.column_stack([m.predict_proba(X_val)[:, 1] for _, m in models])
    test_stack = np.column_stack([m.predict_proba(X_test)[:, 1] for _, m in models])
    
    if meta is not None:
        val_probs_raw = meta.predict_proba(val_stack)[:, 1]
        test_probs_raw = meta.predict_proba(test_stack)[:, 1]
    else:
        val_probs_raw = val_stack.mean(axis=1)
        test_probs_raw = test_stack.mean(axis=1)
        
    val_probs_cal = calibrator.predict(val_probs_raw).astype(np.float32)
    test_probs_cal = calibrator.predict(test_probs_raw).astype(np.float32)
    
    pd.DataFrame({"prob_cal": val_probs_cal}).to_csv("results/val_probs.csv", index=False)
    pd.DataFrame({"prob_cal": test_probs_cal}).to_csv("results/test_probs.csv", index=False)
    
    print("Successfully generated all frozen dataset files in 'results/' directory:")
    print(" - results/val_probs.csv")
    print(" - results/val_labels.csv")
    print(" - results/val_meta.csv")
    print(" - results/test_probs.csv")
    print(" - results/test_labels.csv")
    print(" - results/test_meta.csv")

if __name__ == "__main__":
    main()
