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
MIMIC_IV_DIR = "/home/csnn04/S8-MP/Major-Project/readmission-ai/data/physionet.org/files/mimiciv/3.1"
MIMIC_NOTE_DIR = "/home/csnn04/S8-MP/Major-Project/readmission-ai/data/physionet.org/files/mimic-iv-note/2.2"
MIMIC_BHC_DIR = "/home/csnn04/S8-MP/Major-Project/readmission-ai/data/physionet.org/files/mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0"

# Note: MIMIC_NOTE_PATH is typically found in MIMIC_NOTE_DIR/note/discharge.csv.gz

# Local Clinical-T5 models (PhysioNet download)
CLINICAL_T5_ROOT = os.path.join(BASE_DIR, "physionet.org", "files", "clinical-t5", "1.0.0")
CLINICAL_T5_BASE_DIR = os.path.join(CLINICAL_T5_ROOT, "Clinical-T5-Base")
CLINICAL_T5_LARGE_DIR = os.path.join(CLINICAL_T5_ROOT, "Clinical-T5-Large")
CLINICAL_T5_SCI_DIR = os.path.join(CLINICAL_T5_ROOT, "Clinical-T5-Sci")

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
    CLINICAL_T5_LARGE_DIR,
    CLINICAL_T5_BASE_DIR,
    CLINICAL_T5_SCI_DIR,
    "luqh/ClinicalT5-base",
    "emilyalsentzer/Bio_ClinicalBERT",
    "sentence-transformers/all-mpnet-base-v2",
]

# Embedding Settings
EMBEDDING_DIM = 512
TEXT_MAX_LENGTH = 512
BATCH_SIZE_GPU = 16
BATCH_SIZE_CPU = 8

# PCA Settings
RANDOM_STATE = 42

# 30-Day Readmission Thresholds
THRESHOLD_HIGH_RISK = 0.55
THRESHOLD_MEDIUM_RISK = 0.4

# Training toggles
# Set False to disable SMOTETomek oversampling in 03_train.py
ENABLE_SMOTE = False

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

# ========================================
# 5. EXTRACTION SETTINGS  (01_extract.py)
# ========================================
# Sample limit — None = full ~546k admissions; set an int for fast test runs
N_SAMPLES       = None

# How many top pivot categories to keep as binary features.
# Larger values = more features = potentially higher recall but slower training.
TOP_DX_CATS  = 50    # ICD-9/10 3-char diagnosis categories
TOP_PROC     = 80    # ICD procedure codes
TOP_MED      = 80    # EMAR medication names

# Clinical threshold flags — directly affect binary engineered features used by the model.
# Changing these shifts the decision boundary for downstream risk flags.
VITALS_HYPO_SBP     = 90     # mmHg  — hypotension flag
VITALS_HYPER_SBP    = 160    # mmHg  — hypertension flag
VITALS_HYPOXIA_SPO2 = 92     # %     — hypoxia flag
VITALS_TACHY_HR     = 100    # bpm   — tachycardia flag
VITALS_BRADY_HR     = 60     # bpm   — bradycardia flag
VITALS_TACHYPNEA_RR = 20     # /min  — tachypnea flag
VITALS_FEVER_F      = 100.4  # °F    — fever flag
VITALS_GCS_LOW      = 10     # score — low GCS flag
LAB_AKI_CREATININE  = 1.5   # mg/dL — AKI flag
LAB_HYPERLAC        = 2.0   # mmol/L — hyperlactatemia flag
LAB_ANEMIA_HGB      = 10.0  # g/dL  — anemia flag
LAB_LEUKO_HIGH      = 11.0  # 10³/µL — leukocytosis flag
LAB_LEUKO_LOW       = 4.0   # 10³/µL — leukopenia flag
LAB_THROMBO_LOW     = 100   # 10³/µL — thrombocytopenia flag
LAB_HYPONATR        = 135   # mEq/L — hyponatremia flag
LAB_HYPERNATR       = 145   # mEq/L — hypernatremia flag

# Min occurrence count for a code to be considered as a pivot column contributor.
# Higher = sparser feature matrix; lower = more features, more noise.
EXTRACT_MIN_CONTRIB_COUNT = 50

# How many top-correlated features to retain in the final CSV (finalize step).
# Reducing this cuts dimensionality before embedding fusion.
EXTRACT_FEATURE_KEEP_TOP  = 500

# ========================================
# 6. FEATURE SELECTION  (01b_select_features.py)
# ========================================
# Number of features to keep after multi-method ranking.
# Higher = more signal but slower training; lower = more aggressive pruning.
SELECT_TOP_N        = 160

# CV folds used to average SHAP and gain importances.
SELECT_N_FOLDS      = 3

# Pearson correlation above this → drop the lower-importance duplicate feature.
SELECT_CORR_THRESH  = 0.97

# Variance below this → zero-variance filter (pre-selection).
SELECT_VAR_THRESH   = 1e-8

# Aggregated ranking weights — must sum to 1.0.
# These determine how much each method drives the final feature ranking.
SELECT_WEIGHT_SHAP  = 0.5   # SHAP mean |value|
SELECT_WEIGHT_GAIN  = 0.3   # LightGBM gain importance
SELECT_WEIGHT_MI    = 0.2   # Mutual information

# Max rows sampled for mutual information (memory-controlled).
SELECT_MI_SUBSAMPLE = 50_000

# ========================================
# 7. EMBEDDING SETTINGS  (02_embed.py)
# ========================================
# Final PCA output dimension — must match EMBEDDING_DIM above.
# Increasing this captures more text variance but raises model dimensionality.
EMBED_DIM           = 512   # ← synced with EMBEDDING_DIM

# Encoding parameters — affect quality vs. speed trade-off for text embeddings.
EMBED_MAX_SEQ_LEN   = 512   # max tokens fed to the transformer
EMBED_GPU_BATCH     = 32    # batch size on GPU (RTX 4000 Ada 20GB friendly)
EMBED_CPU_BATCH     = 8

# Note preprocessing — control how much of each clinical note is used.
EMBED_MIN_TEXT_LEN  = 50    # chars; shorter notes are discarded
EMBED_MAX_CHARS     = 5_000 # chars per admission (truncation limit)
EMBED_CHUNK_WORDS   = 256   # words per chunk for long-note splitting
EMBED_CHUNK_OVERLAP = 64    # word overlap between consecutive chunks
EMBED_MAX_CHUNKS    = 10    # max chunks per note (limits compute)

# Embedding reduction strategy
EMBEDDING_REDUCTION = "pca"  # "pca" | "umap"

# Multi-model embedding fusion
EMBED_FUSION_ENABLED = True
EMBED_FUSION_MODELS = [
    "finetuned_t5",
    "emilyalsentzer/Bio_ClinicalBERT",
    "sentence-transformers/all-mpnet-base-v2",
]

# ========================================
# 8. TRAINING SETTINGS  (03_train.py)
# ========================================
# Optuna hyperparameter search — more trials → better HPO, slower run.
TRAIN_OPTUNA_TRIALS      = 50  # was 200; plateaued at 50 in previous run

# DART tree cap (DART has no early stopping, so keep bounded).
TRAIN_DART_MAX_TREES     = 800

# Patient-level GroupKFold splits for cross-validation.
TRAIN_N_FOLDS            = 5

# Temporal train/val/test split fractions (fractions of unique patients).
TRAIN_TEST_FRAC          = 0.15
TRAIN_VAL_FRAC           = 0.15

# SMOTETomek minority class oversample target ratio.
# Only applied when ENABLE_SMOTE=True. Higher = more synthetic positives.
TRAIN_SMOTE_RATIO        = 0.35

# Weighted blend optimisation — random search budget over ensemble weights.
TRAIN_BLEND_TRIALS       = 1000  # was 500

# How many of the highest-variance ct5_* embedding dimensions to keep.
# Reduces embedding noise; set to EMBED_DIM to keep all.
TRAIN_CT5_KEEP_DIMS      = 512

# Candidate feature-count subsets evaluated during auto feature selection.
# The one giving highest val AUROC is chosen.
TRAIN_FEATURE_SUBSETS    = [128, 160, 220, 259]

# Enable stacked ensemble (LightGBM + XGBoost [+ CatBoost if installed])
TRAIN_ENABLE_STACK       = True

# Auto feature subset search toggle
TRAIN_ENABLE_AUTO_FEATURE_SUBSET = True  # was False; searches best feature count

# Multi-seed ensembling
TRAIN_SEEDS = [42, 2024, 777]
TRAIN_HPO_ONCE = True

# Logistic meta-learner C regularisation search space.
TRAIN_META_C_CANDIDATES  = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]  # was [0.3, 1.0, 3.0, 10.0]

# Decision threshold strategy: "f1" | "recall80" | "j" (Youden-J) | "mcc"
TRAIN_THRESHOLD_STRATEGY = "mcc"

# HPO objective: True = maximise AUROC; False = composite(AUROC, AUPRC).
TRAIN_OPTIMIZE_AUROC     = True  # Directly optimize for AUROC to beat 0.80

# AUPRC weight in composite objective (used when TRAIN_OPTIMIZE_AUROC=False).
TRAIN_HPO_ALPHA_AUPRC    = 0.25  # was 0.35; less weight on AUPRC when chasing AUROC

# ========================================
# 9. ANALYSIS SETTINGS  (05_analyze.py)
# ========================================
# Rows sampled for SHAP TreeExplainer — larger = more accurate but slower.
SHAP_N_SAMPLES = 500

# ========================================
# 10. GATED FUSION SETTINGS
# ========================================

# Path for the new gated model
GATE_MODEL_PKL = os.path.join(MODELS_DIR, "trance_gate.pkl")

# Concat-MLP baseline (no gating; concatenate text + tabular directly)
CONCAT_MLP_MODEL_PKL = os.path.join(MODELS_DIR, "concat_mlp.pkl")

# Gate network architecture
GATE_HIDDEN_DIM = 128        # hidden layer size inside gate network
GATE_TEXT_DIM = 512          # must match EMBED_DIM from section 7
GATE_DROPOUT = 0.3           # dropout in MLP classifier head
GATE_LR = 1e-4               # learning rate for Adam optimizer
GATE_EPOCHS = 100            # max epochs, early stopping will cut this short
GATE_PATIENCE = 10           # early stopping patience on val AUROC
GATE_SEEDS = [42, 2024, 777] # seeds for multi-seed ensemble

# Gate SHAP (optional; can be slow on CPU)
GATE_ENABLE_SHAP = False
GATE_SHAP_N_BACKGROUND = 128
GATE_SHAP_N_SAMPLES = 512

# ========================================
# 11. ANALYSIS OUTPUT PATHS
# ========================================

FAIRNESS_RESULTS_CSV    = os.path.join(RESULTS_DIR, "fairness_analysis.csv")
CALIBRATION_RESULTS_CSV = os.path.join(RESULTS_DIR, "calibration_analysis.csv")
GATE_WEIGHTS_NPY        = os.path.join(RESULTS_DIR, "gate_weights.npy")
GATE_PATIENT_IDS_NPY    = os.path.join(RESULTS_DIR, "gate_patient_ids.npy")
GATE_SHAP_IMPORTANCE_CSV = os.path.join(RESULTS_DIR, "gate_shap_importance.csv")
GATE_SHAP_SUMMARY_PNG     = os.path.join(FIGURES_DIR, "gate_shap_summary.png")
EARLY_WARNING_CSV       = os.path.join(RESULTS_DIR, "early_warning_results.csv")
TEMPORAL_DRIFT_CSV      = os.path.join(RESULTS_DIR, "temporal_drift_results.csv")

CONCAT_MLP_REPORT_JSON  = os.path.join(RESULTS_DIR, "concat_mlp_training_report.json")

# ========================================
# 12. EARLY WARNING SETTINGS
# ========================================

# Day cutoffs to evaluate for early warning experiment
EARLY_WARNING_DAYS = [1, 2, 3, 5, 7]
# "full" is added automatically in the early warning script
