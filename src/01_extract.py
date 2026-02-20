# src/01_extract.py
"""
MIMIC-IV Feature Extraction Module
Refactored for modularity, efficiency, and industrial-grade quality.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from .config import DATA_DIR, MIMIC_RAW_DIR, FEATURES_CSV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MIMICExtractor:
    """
    Handles extraction and engineering of features from MIMIC-IV tables.
    """
    
    def __init__(self, n_samples=150000):
        self.n_samples = n_samples
        self.raw_dir = MIMIC_RAW_DIR
        self.df = None
        self.adm = None
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

    def _load_csv(self, filename, **kwargs):
        """Helper to load CSV with error handling"""
        path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(path):
            # Try core folder as well
            path = os.path.join(self.raw_dir, "hosp", filename)
            if not os.path.exists(path):
                path = os.path.join(self.raw_dir, "icu", filename)
        
        if not os.path.exists(path):
            # Fallback to local data dir if not in MIMIC_RAW_DIR
            path = os.path.join(DATA_DIR, filename)
            
        if not os.path.exists(path):
            logger.error(f"File not found: {filename}")
            return pd.DataFrame()
            
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()

    def extract_core_demographics(self):
        """1. Core admissions & demographics"""
        logger.info("Extracting core admissions & demographics...")
        self.adm = self._load_csv("admissions.csv.gz", nrows=self.n_samples)
        pat = self._load_csv("patients.csv.gz")
        
        if self.adm.empty or pat.empty:
            raise ValueError("Required core files (admissions or patients) are missing.")

        for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
            self.adm[col] = pd.to_datetime(self.adm[col])

        # Target variable calculation
        self.adm["los"] = (self.adm["dischtime"] - self.adm["admittime"]).dt.total_seconds() / 3600
        self.adm = self.adm.sort_values(["subject_id", "admittime"])
        self.adm["next_admit"] = self.adm.groupby("subject_id")["admittime"].shift(-1)
        self.adm["readmit_30"] = ((self.adm["next_admit"] - self.adm["dischtime"]).dt.days <= 30).fillna(0).astype(int)

        self.df = self.adm[["subject_id", "hadm_id", "readmit_30"]].copy()
        self.df = self.df.merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
        
        # Mapping and Fillna
        self.df["gender"] = self.df["gender"].map({"M": 1, "F": 0}).fillna(0)
        self.df["anchor_age"] = self.df["anchor_age"].fillna(self.df["anchor_age"].median())
        self.df["died_during_admit"] = self.adm["deathtime"].notna().astype(int)
        self.df["los_hours"] = self.adm["los"].values
        self.df["los_days"] = self.df["los_hours"] / 24
        
        # Categorical mappings
        self._map_categories()
        
        # ED features
        self.df["had_ed"] = self.adm["edregtime"].notna().astype(int)
        self.df["ed_time_hours"] = (self.adm["edouttime"] - self.adm["edregtime"]).dt.total_seconds() / 3600
        self.df["ed_time_hours"] = self.df["ed_time_hours"].fillna(0)

    def _map_categories(self):
        """Internal helper for category mappings"""
        self.df["admission_type"] = self.adm["admission_type"].map({
            "AMBULATORY OBSERVATION": 0, "DIRECT EMER.": 1, "DIRECT OBSERVATION": 2,
            "ELECTIVE": 3, "EU OBSERVATION": 4, "EW EMER.": 5, "OBSERVATION ADMIT": 6,
            "SURGICAL SAME DAY ADMISSION": 7, "URGENT": 8
        }).fillna(0)
        
        self.df["admission_location"] = self.adm["admission_location"].map({
            "AMBULATORY SURGERY TRANSFER": 0, "CLINIC REFERRAL": 1, "EMERGENCY ROOM": 2,
            "PACU": 5, "PHYSICIAN REFERRAL": 6, "PROCEDURE SITE": 7, "TRANSFER FROM HOSPITAL": 8,
            "TRANSFER FROM SKILLED NURSING FACILITY": 9, "WALK-IN/SELF REFERRAL": 10
        }).fillna(2)
        
        self.df["discharge_location"] = self.adm["discharge_location"].map({
            "DEAD": 4, "DIED": 4, "HOME": 6, "HOME HEALTH CARE": 7, "REHAB": 11, "SKILLED NURSING FACILITY": 12
        }).fillna(6)
        
        self.df["insurance"] = self.adm["insurance"].map({"Medicaid": 0, "Medicare": 1, "Other": 2}).fillna(2)

    def extract_icu_features(self):
        """2. ICU features"""
        logger.info("Extracting ICU features...")
        icu = self._load_csv("icustays.csv.gz")
        if icu.empty: return

        icu_agg = icu.groupby("hadm_id").agg({
            "stay_id": "count",
            "los": ["sum", "mean", "max"]
        }).reset_index()
        icu_agg.columns = ["hadm_id", "icu_count", "icu_los_sum", "icu_los_mean", "icu_los_max"]
        
        self.df = self.df.merge(icu_agg, on="hadm_id", how="left").fillna(0)
        self.df["had_icu"] = (self.df["icu_count"] > 0).astype(int)
        self.df["icu_los_ratio"] = self.df["icu_los_sum"] / (self.df["los_hours"] + 1)

    def extract_lab_features(self):
        """3. Lab events"""
        logger.info("Extracting Lab features...")
        labs = self._load_csv("labevents.csv.gz", nrows=2_000_000)
        if labs.empty: return

        important_labs = [50912, 50902, 50882, 50931, 50960, 50971, 50983, 51006, 51221, 51222]
        labs = labs[(labs["itemid"].isin(important_labs)) & (labs["valuenum"].notna())]

        lab_stats = labs.groupby("hadm_id")["valuenum"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        lab_stats.columns = ["hadm_id", "lab_count", "lab_mean", "lab_std", "lab_min", "lab_max"]

        self.df = self.df.merge(lab_stats, on="hadm_id", how="left").fillna(0)
        self.df["lab_cv"] = self.df["lab_std"] / (self.df["lab_mean"] + 0.1)
        self.df["lab_abnormal"] = (self.df["lab_cv"] > self.df["lab_cv"].median()).astype(int)

    def extract_codes_and_meds(self):
        """4, 5, 6, 7. Codes and Meds (Optimized)"""
        logger.info("Extracting Diagnosis, Procedure, and Medication codes...")
        
        # Diagnoses
        dx = self._load_csv("diagnoses_icd.csv.gz", nrows=500_000)
        if not dx.empty:
            dx_stats = dx.groupby("hadm_id").size().reset_index(name="dx_count")
            self.df = self.df.merge(dx_stats, on="hadm_id", how="left").fillna(0)
            
            top_dx = dx["icd_code"].value_counts().head(100).index.tolist()
            self._pivot_and_merge(dx, top_dx, "icd_code", "dx")

        # Procedures
        proc = self._load_csv("procedures_icd.csv.gz", nrows=300_000)
        if not proc.empty:
            proc_stats = proc.groupby("hadm_id").size().reset_index(name="proc_count")
            self.df = self.df.merge(proc_stats, on="hadm_id", how="left").fillna(0)
            
            top_proc = proc["icd_code"].value_counts().head(50).index.tolist()
            self._pivot_and_merge(proc, top_proc, "icd_code", "proc")

        # Medications (EMAR)
        emar = self._load_csv("emar.csv.gz", nrows=1_000_000)
        if not emar.empty:
            emar["medication"] = emar["medication"].str.lower().fillna("")
            top_meds = emar["medication"].value_counts().head(50).index.tolist()
            self._pivot_and_merge(emar, top_meds, "medication", "med")

    def _pivot_and_merge(self, source_df, items, col, prefix):
        """Helper to pivot and merge sparse categorical features"""
        filt = source_df[source_df[col].isin(items)].copy()
        filt["val"] = 1
        pivot = filt.pivot_table(index="hadm_id", columns=col, values="val", aggfunc="count", fill_value=0).reset_index()
        pivot.columns = ["hadm_id"] + [f"{prefix}_{c[:15]}" for c in pivot.columns[1:]]
        self.df = self.df.merge(pivot, on="hadm_id", how="left").fillna(0)

    def add_historical_and_engineered(self):
        """11, 12. Advanced Engineering"""
        logger.info("Calculating historical features and engineering advanced metrics...")
        
        # Historical
        self.df = self.df.merge(self.adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
        self.df["prev_admissions"] = self.df.groupby("subject_id").cumcount()
        self.df["prev_readmit_rate"] = self.df.groupby("subject_id")["readmit_30"].transform(
            lambda x: x.shift(1).expanding().mean()).fillna(0)
        self.df["days_since_last"] = self.df.groupby("subject_id")["admittime"].diff().dt.days.fillna(999)
        
        # Composite scores
        self.df["severity_score"] = self.df["icu_count"]*2 + self.df.get("proc_count", 0) + self.df.get("dx_count", 0)
        self.df["high_risk"] = ((self.df["icu_count"] > 0) | (self.df["los_days"] > 7) | (self.df["anchor_age"] > 75)).astype(int)
        
        # Interactions
        self.df["age_los"] = self.df["anchor_age"] * self.df["los_days"]
        self.df["dx_per_day"] = self.df.get("dx_count", 0) / (self.df["los_days"] + 1)
        
        # Temporal
        self.df["admission_hour"] = self.df["admittime"].dt.hour
        self.df["admission_dow"] = self.df["admittime"].dt.dayofweek
        self.df["is_night"] = ((self.df["admission_hour"] >= 22) | (self.df["admission_hour"] <= 6)).astype(int)
        
        self.df = self.df.drop(columns=["admittime"], errors="ignore")
        self.df = self.df.replace([np.inf, -np.inf], 0).fillna(0)

    def run_full_pipeline(self):
        """Execute the entire extraction workflow"""
        try:
            self.extract_core_demographics()
            self.extract_icu_features()
            self.extract_lab_features()
            self.extract_codes_and_meds()
            self.add_historical_and_engineered()
            
            logger.info(f"Extraction complete. Shape: {self.df.shape}")
            self.df.to_csv(FEATURES_CSV, index=False)
            logger.info(f"Saved to {FEATURES_CSV}")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    extractor = MIMICExtractor()
    extractor.run_full_pipeline()