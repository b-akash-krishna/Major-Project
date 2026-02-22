# src/config.py
import os

# ========================================
# 1. FILE SYSTEM PATHS
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# MIMIC-IV Raw Data Sources (External)
MIMIC_IV_DIR = r"C:\INTERNSHIP_COURSES\Final-Project\physionet.org\files\mimiciv\3.1"
MIMIC_NOTE_DIR = r"C:\INTERNSHIP_COURSES\Final-Project\physionet.org\files\mimic-iv-note\2.2"
MIMIC_BHC_DIR = r"C:\INTERNSHIP_COURSES\Final-Project\physionet.org\files\mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0"

# Note: MIMIC_NOTE_PATH is typically found in MIMIC_NOTE_DIR/note/discharge.csv.gz

# Processed Data Files
FEATURES_CSV = os.path.join(DATA_DIR, "ultimate_features.csv")
# EMBEDDINGS_CSV = os.path.join(DATA_DIR, "clinical_t5_embeddings.csv")
EMBEDDINGS_CSV = os.path.join(DATA_DIR, "embeddings.csv")

# Model Files
MAIN_MODEL_PKL = os.path.join(MODELS_DIR, "trance_framework.pkl")
# EMBEDDING_INFO_PKL = os.path.join(MODELS_DIR, "clinical_t5_info.pkl")
EMBEDDING_INFO_PKL = os.path.join(MODELS_DIR, "embedding_info.pkl")
FEATURE_METADATA_JSON = os.path.join(MODELS_DIR, "feature_metadata.json")

# ========================================
# 2. MODEL CONFIGURATION
# ========================================

# Text Embedding Models (ordered by preference)
TEXT_MODEL_CANDIDATES = [
    "luqh/ClinicalT5-base",
    "emilyalsentzer/Bio_ClinicalBERT"
]

# Embedding Settings
EMBEDDING_DIM = 128
TEXT_MAX_LENGTH = 512
BATCH_SIZE_GPU = 16
BATCH_SIZE_CPU = 8

# PCA Settings
RANDOM_STATE = 42

# 30-Day Readmission Thresholds
THRESHOLD_HIGH_RISK = 0.5
THRESHOLD_MEDIUM_RISK = 0.3

# ========================================
# 3. FEATURE DEFAULTS
# ========================================

DEFAULTS = {
    'los_hours': 48, 'admission_hour': 12, 'admission_dow': 2,
    'ed_time_hours': 0, 'los_transfer': 0, 'icu_los_ratio': 0,
    'severity_score': 2, 'complexity_score': 2, 'instability_score': 0,
    'prev_los_mean': 0, 'prev_icu_rate': 0, 'lab_mean': 0, 'lab_count': 0,
    'dx_unique': 1, 'service_count': 1, 'rx_count': 0, 'med_admin_count': 0,
    'micro_count': 0, 'admission_location': 2, 'insurance': 1,
    'had_ed': 0, 'multiple_icu': 0, 'lab_abnormal': 0,
    'proc_count': 0, 'dx_count': 1, 'transfer_count': 0,
    'had_icu': 0, 'icu_los_sum': 0, 'icu_count': 0,
    'icu_los_mean': 0, 'icu_los_max': 0, 'icu_los_min': 0, 'icu_los_std': 0,
    'lab_median': 0, 'lab_min': 0, 'lab_max': 0, 'lab_sem': 0,
    'lab_q25': 0, 'lab_q75': 0, 'lab_range': 0, 'lab_iqr': 0,
    'lab_skew': 0, 'lab_kurt': 0, 'lab_cv': 0,
    'dx_seq_mean': 0, 'dx_seq_max': 0,
    'prev_readmits': 0, 'age_group': 2, 'los_cat': 2,
    'high_risk': 0, 'very_high_risk': 0,
    'icu_lab': 0, 'readmit_age': 0, 'dx_proc': 0,
    'med_per_day': 0, 'is_weekend': 0, 'is_night': 0,
    'died_during_admit': 0
}

# ========================================
# 4. API SETTINGS
# ========================================
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["*"]
