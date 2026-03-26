# src/08_predict.py
"""
ACAGN - Interactive Prediction CLI.
Run as:  python src/08_predict.py   OR   python -m src.08_predict

Embedding method matches training exactly:
  - sentence-transformers/all-mpnet-base-v2 -> 768-dim mean pool -> PCA(128)
  - Reads embedding_info.pkl for stored PCA (fitted during 02_embed.py)
  - Falls back to ClinicalT5 PyTorch conversion if embedding_info says so
"""

import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

# Import fix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config import (
        DEFAULTS,
        EMBEDDINGS_CSV,
        FEATURES_CSV,
        FEATURE_METADATA_JSON,
        MODELS_DIR,
        RESULTS_DIR,
        THRESHOLD_HIGH_RISK,
        THRESHOLD_MEDIUM_RISK,
    )
    from embedding_utils import get_embedding, get_model_container
except ImportError:
    from .config import (
        DEFAULTS,
        EMBEDDINGS_CSV,
        FEATURES_CSV,
        FEATURE_METADATA_JSON,
        MODELS_DIR,
        RESULTS_DIR,
        THRESHOLD_HIGH_RISK,
        THRESHOLD_MEDIUM_RISK,
    )
    from .embedding_utils import get_embedding, get_model_container

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load model once at import time
model_container = get_model_container()
_REFERENCE_MEANS_CACHE = None
_TEMPLATE_POOL_CACHE = None
FEATURE_IMPORTANCE_CSV = os.path.join(MODELS_DIR, "feature_importance_report.csv")


# Helper functions

def _load_major_features(model_features: list) -> list:
    """
    Load important model features from metadata and keep only practical,
    non-embedding, non-derived fields for user prompts.
    """
    derived = {
        "los_hours", "log_los_days", "age_los", "is_weekend", "is_night",
        "proc_per_day", "dx_per_day", "med_per_day", "icu_los_ratio",
        "prev_readmits", "readmit_age", "dx_proc", "los_transfer",
    }
    already_prompted = {"anchor_age", "gender", "los_days", "prev_admissions", "admission_type"}

    top = []
    try:
        if os.path.exists(FEATURE_IMPORTANCE_CSV):
            imp = pd.read_csv(FEATURE_IMPORTANCE_CSV, low_memory=False)
            if "feature" in imp.columns:
                score_col = "combined_score" if "combined_score" in imp.columns else None
                if score_col:
                    imp = imp.sort_values(score_col, ascending=False)
                top = imp["feature"].dropna().astype(str).tolist()
    except Exception:
        top = []

    try:
        if not top and os.path.exists(FEATURE_METADATA_JSON):
            with open(FEATURE_METADATA_JSON, "r", encoding="utf-8") as f:
                meta = json.load(f)
            top = meta.get("top_20_important", []) or []
    except Exception:
        top = []

    if not top:
        top = model_features[:30]

    filtered = []
    for feat in top:
        if feat not in model_features:
            continue
        if feat.startswith("ct5_"):
            continue
        if feat in derived or feat in already_prompted:
            continue
        filtered.append(feat)

    return filtered[:12]


FRIENDLY_MAJOR_FEATURES = {
    "bmi": {
        "label": "Body Mass Index (BMI)",
        "min": 10.0,
        "max": 60.0,
    },
    "days_since_last": {
        "label": "Days since last hospital discharge",
        "min": 0,
        "max": 3650,
    },
    "prev_readmit_rate": {
        "label": "Past readmission rate (0.0 to 1.0)",
        "min": 0.0,
        "max": 1.0,
    },
    "prev_los_mean": {
        "label": "Average past stay length in days",
        "min": 0.0,
        "max": 60.0,
    },
    "ed_time_hours": {
        "label": "ER wait time in hours",
        "min": 0.0,
        "max": 72.0,
    },
    "transfer_count": {
        "label": "Number of ward/room transfers",
        "min": 0,
        "max": 20,
    },
    "proc_count": {
        "label": "Number of procedures this stay",
        "min": 0,
        "max": 50,
    },
    "dx_count": {
        "label": "Number of diagnosed conditions",
        "min": 1,
        "max": 80,
    },
    "had_ed": {
        "label": "Came via Emergency Room? (0=no, 1=yes)",
        "min": 0,
        "max": 1,
    },
    "insurance": {
        "label": "Insurance category code",
        "min": 0,
        "max": 10,
    },
    "race_enc": {
        "label": "Race encoded value",
        "min": 0.0,
        "max": 1.0,
    },
    "language_enc": {
        "label": "Language encoded value",
        "min": 0.0,
        "max": 1.0,
    },
    "marital_enc": {
        "label": "Marital-status encoded value",
        "min": 0.0,
        "max": 1.0,
    },
    "admission_location": {
        "label": "Admission location code",
        "min": 0,
        "max": 20,
    },
    "discharge_location": {
        "label": "Discharge location code",
        "min": 0,
        "max": 20,
    },
    "admission_hour": {
        "label": "Admission hour (0-23)",
        "min": 0,
        "max": 23,
    },
    "admission_dow": {
        "label": "Admission day of week (0=Mon ... 6=Sun)",
        "min": 0,
        "max": 6,
    },
    "rx_count": {
        "label": "Medication orders count",
        "min": 0,
        "max": 500,
    },
    "med_admin_count": {
        "label": "Medication administrations count",
        "min": 0,
        "max": 2000,
    },
    "lab_abnormal_count": {
        "label": "Abnormal lab results count",
        "min": 0,
        "max": 1000,
    },
    "lab_abnormal_rate": {
        "label": "Abnormal lab ratio (0.0 to 1.0)",
        "min": 0.0,
        "max": 1.0,
    },
    "poe_count": {
        "label": "Provider order entries count",
        "min": 0,
        "max": 1000,
    },
    "primary_dx_freq": {
        "label": "Primary diagnosis frequency",
        "min": 0.0,
        "max": 1.0,
    },
}

COMMON_PROFILE_FEATURES = [
    "insurance",
    "race_enc",
    "language_enc",
    "marital_enc",
    "admission_location",
    "discharge_location",
    "admission_dow",
    "admission_hour",
]


def _default_for_feature(name: str, feature_means: dict):
    if name in feature_means:
        return float(feature_means[name])
    if name in DEFAULTS:
        return DEFAULTS[name]
    return 0.0


def _load_reference_feature_means(model_features: list) -> dict:
    """
    Build means from raw feature files (pre-SMOTE), which are better defaults
    than model-embedded means computed after class rebalancing.
    """
    global _REFERENCE_MEANS_CACHE
    if _REFERENCE_MEANS_CACHE is not None:
        return _REFERENCE_MEANS_CACHE

    means = {}
    non_ct5 = [f for f in model_features if not f.startswith("ct5_")]
    ct5 = [f for f in model_features if f.startswith("ct5_")]

    try:
        feat_path = FEATURES_CSV.replace(".csv", "_pruned.csv")
        if not os.path.exists(feat_path):
            feat_path = FEATURES_CSV
        if os.path.exists(feat_path):
            cols = list(pd.read_csv(feat_path, nrows=0).columns)
            usecols = [c for c in non_ct5 if c in cols]
            if usecols:
                df = pd.read_csv(feat_path, usecols=usecols, low_memory=False)
                means.update(df.mean(numeric_only=True).to_dict())
    except Exception:
        pass

    try:
        if os.path.exists(EMBEDDINGS_CSV) and ct5:
            cols = list(pd.read_csv(EMBEDDINGS_CSV, nrows=0).columns)
            usecols = [c for c in ct5 if c in cols]
            if usecols:
                df = pd.read_csv(EMBEDDINGS_CSV, usecols=usecols, low_memory=False)
                means.update(df.mean(numeric_only=True).to_dict())
    except Exception:
        pass

    _REFERENCE_MEANS_CACHE = {k: float(v) for k, v in means.items()}
    return _REFERENCE_MEANS_CACHE


def _load_template_pool(model_features: list, max_rows: int = 30000) -> pd.DataFrame:
    """
    Load a limited pool of real rows for nearest-neighbor baseline fill.
    This keeps manual-mode predictions closer to realistic feature combinations.
    """
    global _TEMPLATE_POOL_CACHE
    if _TEMPLATE_POOL_CACHE is not None:
        return _TEMPLATE_POOL_CACHE

    try:
        feat_path = FEATURES_CSV.replace(".csv", "_pruned.csv")
        if not os.path.exists(feat_path):
            feat_path = FEATURES_CSV
        if not os.path.exists(feat_path):
            _TEMPLATE_POOL_CACHE = pd.DataFrame()
            return _TEMPLATE_POOL_CACHE

        tab = pd.read_csv(feat_path, nrows=max_rows, low_memory=False)
        if os.path.exists(EMBEDDINGS_CSV):
            emb = pd.read_csv(EMBEDDINGS_CSV, nrows=max_rows, low_memory=False)
            if "hadm_id" in tab.columns and "hadm_id" in emb.columns:
                df = tab.merge(emb, on="hadm_id", how="left")
            else:
                df = tab
        else:
            df = tab

        keep = [c for c in model_features if c in df.columns]
        _TEMPLATE_POOL_CACHE = df[keep].fillna(0)
        return _TEMPLATE_POOL_CACHE
    except Exception:
        _TEMPLATE_POOL_CACHE = pd.DataFrame()
        return _TEMPLATE_POOL_CACHE


def _nearest_template_baseline(user_data: dict, model_features: list) -> dict:
    """Pick a realistic baseline row nearest to user-entered core fields."""
    pool = _load_template_pool(model_features)
    if pool.empty:
        return {}

    keys = [k for k in [
        "anchor_age", "los_days", "prev_admissions", "admission_type",
        "days_since_last", "prev_readmit_rate", "prev_los_mean"
    ] if k in pool.columns and k in user_data]
    if not keys:
        return {}

    scales = {
        "anchor_age": 20.0,
        "los_days": 5.0,
        "prev_admissions": 5.0,
        "admission_type": 1.0,
        "days_since_last": 365.0,
        "prev_readmit_rate": 0.2,
        "prev_los_mean": 3.0,
    }

    dist = np.zeros(len(pool), dtype=np.float32)
    for k in keys:
        s = float(scales.get(k, 1.0))
        diff = np.abs(pool[k].astype(float).to_numpy() - float(user_data[k])) / max(s, 1e-6)
        # Categorical mismatch for admission_type should cost more than small numeric shifts.
        if k == "admission_type":
            diff = (pool[k].astype(float).to_numpy() != float(user_data[k])).astype(np.float32) * 2.0
        dist += diff.astype(np.float32)

    idx = int(np.argmin(dist))
    return {c: float(pool.iloc[idx][c]) for c in pool.columns}


def _infer_cast(default):
    if isinstance(default, bool):
        return int
    if isinstance(default, int):
        return int
    return float


def _round_default(value):
    if isinstance(value, float):
        return round(value, 2)
    return value


def _coerce_range(value, min_val, max_val):
    if min_val is not None and value < min_val:
        return min_val
    if max_val is not None and value > max_val:
        return max_val
    return value


def _recompute_engineered_features(full: dict) -> None:
    """Recompute core engineered features for consistency with user inputs."""
    age = float(full.get("anchor_age", 65))
    los = float(full.get("los_days", 3.0))
    prev_adm = float(full.get("prev_admissions", 0))
    prev_rate = float(full.get("prev_readmit_rate", 0))
    proc_count = float(full.get("proc_count", 0))
    dx_count = float(full.get("dx_count", 1))
    rx_count = float(full.get("rx_count", 0))
    transfer_count = float(full.get("transfer_count", 0))
    icu_los_sum = float(full.get("icu_los_sum", 0))
    admission_hour = float(full.get("admission_hour", 12))
    admission_dow = float(full.get("admission_dow", 2))

    full["los_hours"] = los * 24
    full["log_los_days"] = float(np.log1p(max(los, 0.0)))
    full["age_los"] = age * los
    full["prev_readmits"] = max(prev_adm * prev_rate, 0.0)
    full["readmit_age"] = age * prev_rate
    full["dx_proc"] = dx_count * proc_count
    full["proc_per_day"] = proc_count / (los + 1.0)
    full["dx_per_day"] = dx_count / (los + 1.0)
    full["med_per_day"] = rx_count / (los + 1.0)
    full["los_transfer"] = los * transfer_count
    full["icu_los_ratio"] = icu_los_sum / (los + 0.01)
    full["is_first_visit"] = 1.0 if prev_adm <= 0 else 0.0
    full["high_risk"] = 1.0 if (los > 10 or age >= 80 or full.get("had_icu", 0) > 0) else 0.0
    full["very_high_risk"] = 1.0 if (los > 20 or age >= 90 or full.get("had_icu", 0) > 1) else 0.0
    full["is_weekend"] = 1.0 if int(round(admission_dow)) in (5, 6) else 0.0
    full["is_night"] = 1.0 if (admission_hour < 7 or admission_hour >= 22) else 0.0

    # Align category buckets used in extraction.
    if age < 40:
        full["age_group"] = 0
    elif age < 55:
        full["age_group"] = 1
    elif age < 65:
        full["age_group"] = 2
    elif age < 75:
        full["age_group"] = 3
    elif age < 85:
        full["age_group"] = 4
    else:
        full["age_group"] = 5

    if los <= 1:
        full["los_cat"] = 0
    elif los <= 3:
        full["los_cat"] = 1
    elif los <= 7:
        full["los_cat"] = 2
    elif los <= 14:
        full["los_cat"] = 3
    elif los <= 30:
        full["los_cat"] = 4
    else:
        full["los_cat"] = 5


def _print_payload_debug(features: list, row: dict) -> None:
    """Print exact model input payload in model order and save to JSON."""
    print("\n" + "=" * 52)
    print(f"  DEBUG PAYLOAD (exact model input): {len(features)} features")
    print("=" * 52)
    for i, feat in enumerate(features, start=1):
        print(f"  {i:03d}. {feat}: {row.get(feat, 0)}")

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "last_prediction_payload.json")
        payload = [{"index": i + 1, "feature": f, "value": row.get(f, 0)} for i, f in enumerate(features)]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n  Payload saved to: {out_path}")
    except Exception as e:
        print(f"\n  Could not save payload JSON: {e}")


# User input

def get_user_input() -> dict:
    """Interactive CLI prompt for patient data."""
    print("\n" + "=" * 52)
    print("  ACAGN RISK PREDICTION - INPUT TERMINAL")
    print("=" * 52)

    def _prompt(prompt: str, default, cast=int, min_val=None, max_val=None):
        """Prompt with default + range validation."""
        try:
            raw = input(prompt).strip()
            val = cast(raw) if raw else default
            if min_val is not None or max_val is not None:
                clipped = _coerce_range(val, min_val, max_val)
                if clipped != val:
                    print(f"  Value out of range, using nearest valid value: {clipped}")
                return clipped
            return val
        except (ValueError, TypeError):
            print(f"  Invalid input, using default: {default}")
            return default

    model_data = model_container.model_data or {}
    model_features = model_data.get("features", [])
    feature_means = model_data.get("feature_means", {})
    reference_means = _load_reference_feature_means(model_features)
    prompt_means = reference_means or feature_means
    major_features = _load_major_features(model_features)

    try:
        print("  Basic details:")
        age = _prompt("  Age in years [65]: ", 65, int, 18, 120)
        gender = _prompt("  Sex (0=female, 1=male) [0]: ", 0, int, 0, 1)
        los = _prompt("  Current hospital stay length in days [3.0]: ", 3.0, float, 0.0, 120.0)
        prev = _prompt("  Number of previous hospital admissions [0]: ", 0, int, 0, 50)
        adm_t = _prompt(
            "  Admission type (1=emergency, 2=urgent, 3=planned) [1]: ",
            1, int, 1, 3
        )

        profile_values = {}
        geo_candidates = [f for f in COMMON_PROFILE_FEATURES if f in model_features]
        if geo_candidates:
            print("\n  Common profile/geographic context (press Enter to keep suggested value):")
            for feat in geo_candidates:
                meta = FRIENDLY_MAJOR_FEATURES.get(feat, {})
                dflt = _round_default(_default_for_feature(feat, prompt_means))
                cast = _infer_cast(dflt)
                label = meta.get("label", feat.replace("_", " ").title())
                profile_values[feat] = _prompt(
                    f"  {label} [{dflt}]: ",
                    dflt,
                    cast,
                    meta.get("min"),
                    meta.get("max"),
                )

        major_values = {}
        if major_features:
            print("\n  Top contributing risk factors (press Enter to keep suggested value):")
            for feat in major_features:
                # Skip cryptic coded variables from interactive prompt.
                if (
                    feat.startswith("proc_")
                    or feat.startswith("dxcat_")
                    or feat.startswith("dx_")
                    or feat.startswith("med_")
                    or feat.startswith("ct5_")
                ):
                    continue
                if feat in profile_values:
                    continue
                meta = FRIENDLY_MAJOR_FEATURES.get(feat, {})
                dflt = _round_default(_default_for_feature(feat, prompt_means))
                cast = _infer_cast(dflt)
                label = meta.get("label", feat.replace("_", " ").title())
                major_values[feat] = _prompt(
                    f"  {label} [{dflt}]: ",
                    dflt,
                    cast,
                    meta.get("min"),
                    meta.get("max"),
                )

        note = input(
            "  Short clinical summary (optional, plain language is fine):\n"
            "  > "
        ).strip()
        debug_payload = input(
            f"  Show full {len(model_features)}-feature debug payload? (y/N): "
        ).strip().lower() == "y"
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return {}

    data = {
        "anchor_age": age,
        "gender": gender,
        "los_days": los,
        "los_hours": los * 24,
        "prev_admissions": prev,
        "admission_type": adm_t,
        "clinical_note": note or None,
        "_debug_payload": debug_payload,
    }
    data.update(profile_values)
    data.update(major_values)
    return data


# Inference

def run_inference(data: dict) -> None:
    """
    Runs the full ensemble prediction for one patient.

    Steps:
      1. Merge user input with DEFAULTS (covers all model features)
      2. Generate 128-dim text embedding via embedding_utils (matches training)
      3. Fill ct5_0 ... ct5_127 into the feature row
      4. Select model feature columns in correct order
      5. Full ensemble predict_proba with calibration
    """
    if not model_container.model_data:
        print("  Model not loaded. Run 03_train.py first.")
        return

    # Step 1: Build full feature dict.
    # Baseline strategy:
    # 1) nearest real template row (best for realistic combinations),
    # 2) raw data means fallback,
    # 3) model means fallback.
    feature_means = model_container.model_data.get("feature_means", {})
    model_features = model_container.model_data.get("features", [])
    template_base = _nearest_template_baseline(data, model_features)
    reference_means = _load_reference_feature_means(model_features)
    baseline_means = template_base or reference_means or feature_means
    if baseline_means:
        full = {k: float(v) for k, v in baseline_means.items()}
        for k, v in DEFAULTS.items():
            full.setdefault(k, v)
    else:
        full = dict(DEFAULTS)
    full.update(data)
    full["los_hours"] = data.get("los_days", 3.0) * 24
    full["log_los_days"] = np.log1p(data.get("los_days", 3.0))
    full["age_los"] = data.get("anchor_age", 65) * data.get("los_days", 3.0)
    _recompute_engineered_features(full)

    # Step 2: Ensure all remaining features are present
    if template_base:
        print("  Baseline initialized from nearest real patient template.")
    elif reference_means:
        print(f"  Baseline initialized from raw dataset means ({len(reference_means)} features).")
    elif feature_means:
        print(f"  Baseline initialized from model feature means ({len(feature_means)} features).")
    else:
        print("  Note: feature_means not in model pkl - retrain with updated 03_train.py")
        print("        for better default values on EHR lab/vital features.")

    # Step 3+4: Generate and inject embeddings
    clinical_note = data.get("clinical_note")
    if clinical_note:
        emb = get_embedding(text=clinical_note, features=full)  # 128-dim np.ndarray
        print(f"  Embedding: {len(clinical_note)} chars -> {len(emb)}-dim vector.")
        for i, val in enumerate(emb):
            full[f"ct5_{i}"] = float(val)
        full["ct5_has_note"] = 1.0
        full["ct5_note_len_chars"] = float(np.log1p(len(clinical_note)))
        full["ct5_note_len_tokens"] = float(np.log1p(len(clinical_note.split())))
    else:
        # Keep baseline ct5 means when note is missing; zero vectors can be
        # out-of-distribution and distort risk.
        print("  No clinical note - using baseline text embedding profile.")
        full["ct5_has_note"] = 0.0
        full["ct5_note_len_chars"] = 0.0
        full["ct5_note_len_tokens"] = 0.0

    # Step 5: Align feature columns with model
    features = model_container.model_data["features"]
    missing = [f for f in features if f not in full and not f.startswith("ct5_")]
    if missing:
        print(f"  {len(missing)} EHR features still missing (set to 0): {missing[:3]} ...")
    row = {f: full.get(f, 0) for f in features}
    X = pd.DataFrame([row])[features]

    # Debug: print exact payload only when requested by user.
    if data.get("_debug_payload", False):
        _print_payload_debug(features, row)

    # Step 6: Full ensemble prediction
    try:
        proba = float(model_container.predict_proba(X)[0])
    except Exception as e:
        print(f"  Prediction error: {e}")
        return

    if proba >= THRESHOLD_HIGH_RISK:
        risk = "HIGH"
        color = "!!!"
    elif proba >= THRESHOLD_MEDIUM_RISK:
        risk = "MEDIUM"
        color = "??"
    else:
        risk = "LOW"
        color = "OK"

    print("\n" + "*" * 52)
    print(f"  RESULT: {color} {risk} RISK {color}")
    print(f"  30-day readmission probability: {proba:.1%}")
    print(f"  Threshold: High>={THRESHOLD_HIGH_RISK:.0%}  Medium>={THRESHOLD_MEDIUM_RISK:.0%}")
    if data.get("_debug_payload", False):
        print("  Debug payload mode: ON")
    else:
        print("  Debug payload mode: OFF (set to 'y' at prompt to inspect all features)")
    print("*" * 52 + "\n")


# Main loop

if __name__ == "__main__":
    if not model_container.model_data:
        print("ERROR: Model not loaded. Ensure models/acagn_framework.pkl exists (legacy: models/trance_framework.pkl).")
        sys.exit(1)

    print(
        f"\nModel loaded: {len(model_container.model_data.get('features', []))} features"
        f" | {len(model_container.model_data.get('models', []))} ensemble members"
    )

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
