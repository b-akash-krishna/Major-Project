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
from .config import (
    DATA_DIR, MIMIC_IV_DIR, MIMIC_NOTE_DIR, MIMIC_BHC_DIR, FEATURES_CSV
)

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
        self.search_dirs = [MIMIC_IV_DIR, MIMIC_NOTE_DIR, MIMIC_BHC_DIR]
        self.df = None
        self.adm = None
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

    def _load_csv(self, filename, **kwargs):
        """
        Recursively finds and loads a CSV/GZ file from the external source directories.
        Loads directly into memory for speed and efficiency.
        """
        source_path = None
        for base_dir in self.search_dirs:
            for root, _, files in os.walk(base_dir):
                if filename in files:
                    source_path = os.path.join(root, filename)
                    break
            if source_path:
                break
        
        if not source_path:
            logger.error(f"Required table {filename} not found in search directories.")
            return pd.DataFrame()
            
        try:
            logger.info(f"Loading {filename} directly from {source_path}...")
            # Compression is inferred from the extension by pandas
            return pd.read_csv(source_path, **kwargs)
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
        """3. Lab events (Expanded)"""
        logger.info("Extracting Lab features...")
        labs = self._load_csv("labevents.csv.gz", nrows=5_000_000)
        if labs.empty: return

        # Broaden lab selection
        important_labs = [
            50912, 50902, 50882, 50931, 50960, 50971, 50983, 51006, 51221, 51222,
            50813, 50820, 50821, 51265, 51301  # Adding Lactate, pH, pO2, Platelets, WBC
        ]
        labs = labs[(labs["itemid"].isin(important_labs)) & (labs["valuenum"].notna())]

        lab_stats = labs.groupby("hadm_id")["valuenum"].agg(["count", "mean", "std", "min", "max", "median"]).reset_index()
        lab_stats.columns = ["hadm_id", "lab_count", "lab_mean", "lab_std", "lab_min", "lab_max", "lab_median"]

        self.df = self.df.merge(lab_stats, on="hadm_id", how="left").fillna(0)
        self.df["lab_cv"] = self.df["lab_std"] / (self.df["lab_mean"] + 0.1)

    def extract_pharmacy_and_rx(self):
        """4. Pharmacy & Prescriptions"""
        logger.info("Extracting Pharmacy & Rx features...")
        rx = self._load_csv("prescriptions.csv.gz", nrows=1_000_000)
        if rx.empty: return
        
        rx_stats = rx.groupby("hadm_id").agg({
            "drug": "count",
            "formulary_drug_cd": "nunique"
        }).reset_index()
        rx_stats.columns = ["hadm_id", "rx_count", "unique_drugs"]
        self.df = self.df.merge(rx_stats, on="hadm_id", how="left").fillna(0)

    def extract_microbiology(self):
        """5. Microbiology (Infections)"""
        logger.info("Extracting Microbiology features...")
        micro = self._load_csv("microbiologyevents.csv.gz")
        if micro.empty: return
        
        micro_stats = micro.groupby("hadm_id").agg({
            "micro_specimen_id": "count",
            "org_name": "nunique"
        }).reset_index()
        micro_stats.columns = ["hadm_id", "micro_count", "unique_cultures"]
        
        # High risk organisms
        high_risk = micro[micro["org_name"].str.contains("STAPHYLOCOCCUS|PSEUDOMONAS|CLOSTRIDIUM", na=False)]
        hr_stats = high_risk.groupby("hadm_id").size().reset_index(name="high_risk_culture")
        
        self.df = self.df.merge(micro_stats, on="hadm_id", how="left").fillna(0)
        self.df = self.df.merge(hr_stats, on="hadm_id", how="left").fillna(0)

    def extract_severity_and_drg(self):
        """6. DRG Codes & Severity"""
        logger.info("Extracting Severity & DRG features...")
        drg = self._load_csv("drgcodes.csv.gz")
        if drg.empty: return
        
        # Map DRG types (MS or APR) to numerical intensity
        drg["drg_weight"] = drg["drg_type"].map({"MS": 1, "APR": 2}).fillna(0)
        drg_stats = drg.groupby("hadm_id")["drg_weight"].max().reset_index(name="drg_intensity")
        
        self.df = self.df.merge(drg_stats, on="hadm_id", how="left").fillna(0)

    def extract_service_history(self):
        """7. Service changes & Transfers"""
        logger.info("Extracting Service changes...")
        services = self._load_csv("services.csv.gz")
        if services.empty: return
        
        service_counts = services.groupby("hadm_id").size().reset_index(name="service_count")
        transfers = self._load_csv("transfers.csv.gz")
        transfer_counts = transfers.groupby("hadm_id").size().reset_index(name="transfer_count")
        
        self.df = self.df.merge(service_counts, on="hadm_id", how="left").fillna(0)
        self.df = self.df.merge(transfer_counts, on="hadm_id", how="left").fillna(0)

    def extract_codes_and_meds(self):
        """8. Codes and Meds (Optimized)"""
        logger.info("Extracting Diagnosis, Procedure, and EMAR codes...")
        
        # Diagnoses
        dx = self._load_csv("diagnoses_icd.csv.gz", nrows=1_000_000)
        if not dx.empty:
            dx_stats = dx.groupby("hadm_id").size().reset_index(name="dx_count")
            self.df = self.df.merge(dx_stats, on="hadm_id", how="left").fillna(0)
            
            top_dx = dx["icd_code"].value_counts().head(150).index.tolist()
            self._pivot_and_merge(dx, top_dx, "icd_code", "dx")

        # Procedures
        proc = self._load_csv("procedures_icd.csv.gz", nrows=500_000)
        if not proc.empty:
            proc_stats = proc.groupby("hadm_id").size().reset_index(name="proc_count")
            self.df = self.df.merge(proc_stats, on="hadm_id", how="left").fillna(0)
            
            top_proc = proc["icd_code"].value_counts().head(75).index.tolist()
            self._pivot_and_merge(proc, top_proc, "icd_code", "proc")

        # Medications (EMAR)
        emar = self._load_csv("emar.csv.gz", nrows=2_000_000)
        if not emar.empty:
            emar["medication"] = emar["medication"].str.lower().fillna("")
            top_meds = emar["medication"].value_counts().head(75).index.tolist()
            self._pivot_and_merge(emar, top_meds, "medication", "med")

    def _pivot_and_merge(self, source_df, items, col, prefix):
        """Helper to pivot and merge sparse categorical features"""
        filt = source_df[source_df[col].isin(items)].copy()
        filt["val"] = 1
        pivot = filt.pivot_table(index="hadm_id", columns=col, values="val", aggfunc="count", fill_value=0).reset_index()
        # Clean column names for LightGBM/SHAP
        pivot.columns = ["hadm_id"] + [f"{prefix}_{str(c)[:15].replace(' ', '_')}" for c in pivot.columns[1:]]
        self.df = self.df.merge(pivot, on="hadm_id", how="left").fillna(0)

    def add_historical_and_engineered(self):
        """Advanced Feature Engineering"""
        logger.info("Calculating historical features and engineering advanced metrics...")
        
        # Historical
        self.df = self.df.merge(self.adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
        self.df = self.df.sort_values(["subject_id", "admittime"])
        self.df["prev_admissions"] = self.df.groupby("subject_id").cumcount()
        self.df["prev_readmit_rate"] = self.df.groupby("subject_id")["readmit_30"].transform(
            lambda x: x.shift(1).expanding().mean()).fillna(0)
        self.df["days_since_last"] = self.df.groupby("subject_id")["admittime"].diff().dt.days.fillna(999)
        
        # Polypharmacy & Complexity
        self.df["polypharmacy"] = (self.df.get("rx_count", 0) > 10).astype(int)
        self.df["complexity_score"] = self.df.get("dx_count", 0) + self.df.get("proc_count", 0) + self.df.get("service_count", 0)
        
        # Composite risk
        self.df["severity_score"] = self.df["icu_count"]*3 + self.df["complexity_score"]
        self.df["high_risk"] = ((self.df["icu_count"] > 0) | (self.df["los_days"] > 10) | (self.df["anchor_age"] > 80)).astype(int)
        
        # Temporal
        self.df["admission_hour"] = self.df["admittime"].dt.hour
        self.df["admission_dow"] = self.df["admittime"].dt.dayofweek
        self.df["is_weekend"] = (self.df["admission_dow"] >= 5).astype(int)
        
        self.df = self.df.drop(columns=["admittime"], errors="ignore")
        self.df = self.df.replace([np.inf, -np.inf], 0).fillna(0)

    def run_full_pipeline(self):
        """Execute the entire holistic extraction workflow"""
        try:
            self.extract_core_demographics()
            self.extract_icu_features()
            self.extract_lab_features()
            self.extract_pharmacy_and_rx()
            self.extract_microbiology()
            self.extract_severity_and_drg()
            self.extract_service_history()
            self.extract_codes_and_meds()
            self.add_historical_and_engineered()
            
            logger.info(f"Holistic extraction complete. Shape: {self.df.shape}")
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

if __name__ == "__main__":
    extractor = MIMICExtractor()
    extractor.run_full_pipeline()