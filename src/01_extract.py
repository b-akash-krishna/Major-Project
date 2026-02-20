# src/01_extract.py
"""
MIMIC-IV feature extraction with memory-aware loading and contribution-driven
feature selection.
"""

from __future__ import annotations

import logging
import os
import re
import traceback
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from .config import DATA_DIR, FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_IV_DIR, MIMIC_NOTE_DIR
except ImportError:
    from config import DATA_DIR, FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_IV_DIR, MIMIC_NOTE_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MIMICExtractor:
    """Extract and engineer readmission-oriented features from MIMIC-IV."""

    def __init__(
        self,
        n_samples: int = 150_000,
        max_rows: Optional[Dict[str, int]] = None,
        top_sparse_features: int = 300,
    ) -> None:
        self.n_samples = int(n_samples)
        self.max_rows = max_rows or {
            "labevents.csv.gz": 5_000_000,
            "prescriptions.csv.gz": 1_000_000,
            "diagnoses_icd.csv.gz": 1_000_000,
            "procedures_icd.csv.gz": 600_000,
            "emar.csv.gz": 2_000_000,
            "microbiologyevents.csv.gz": 1_500_000,
            "discharge.csv.gz": 1_000_000,
            "radiology.csv.gz": 1_500_000,
            "mimic-iv-bhc.csv": 500_000,
        }
        self.top_sparse_features = int(top_sparse_features)
        self.search_dirs = [MIMIC_IV_DIR, MIMIC_NOTE_DIR, MIMIC_BHC_DIR]
        self.file_index: Dict[str, str] = {}
        self.df: Optional[pd.DataFrame] = None
        self.adm: Optional[pd.DataFrame] = None
        self.cohort_hadm: set[int] = set()

        os.makedirs(DATA_DIR, exist_ok=True)
        self._build_file_index()

    def _build_file_index(self) -> None:
        """Build a one-time filename -> path index to avoid repeated os.walk scans."""
        for base_dir in self.search_dirs:
            if not os.path.isdir(base_dir):
                logger.warning("Search directory not found: %s", base_dir)
                continue
            for root, _, files in os.walk(base_dir):
                for name in files:
                    self.file_index.setdefault(name, os.path.join(root, name))
        logger.info("Indexed %s files from source directories", len(self.file_index))

    def _get_path(self, filename: str) -> Optional[str]:
        path = self.file_index.get(filename)
        if path:
            return path
        logger.error("Required file not found: %s", filename)
        return None

    def _load_csv(
        self,
        filename: str,
        usecols: Optional[Sequence[str]] = None,
        dtype: Optional[Dict[str, str]] = None,
        parse_dates: Optional[Sequence[str]] = None,
        nrows: Optional[int] = None,
        low_memory: bool = True,
    ) -> pd.DataFrame:
        path = self._get_path(filename)
        if not path:
            return pd.DataFrame()
        read_nrows = nrows if nrows is not None else self.max_rows.get(filename)
        try:
            logger.info("Loading %s", filename)
            return pd.read_csv(
                path,
                usecols=usecols,
                dtype=dtype,
                parse_dates=parse_dates,
                nrows=read_nrows,
                low_memory=low_memory,
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)
            return pd.DataFrame()

    def _merge_hadm_features(self, other: pd.DataFrame, fill_value: float = 0.0) -> None:
        if other.empty:
            return
        self.df = self.df.merge(other, on="hadm_id", how="left")
        self.df = self.df.fillna(fill_value)

    def _normalize_colname(self, raw: object, prefix: str, max_len: int = 48) -> str:
        text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw).strip().lower())
        return f"{prefix}_{text[:max_len]}".strip("_")

    def _safe_numeric_fill(self) -> None:
        self.df = self.df.replace([np.inf, -np.inf], 0)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)

    def extract_core_demographics(self) -> None:
        logger.info("Step 1/10: admissions + demographics")
        admissions_cols = [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "edregtime",
            "edouttime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
        ]
        self.adm = self._load_csv(
            "admissions.csv.gz",
            usecols=admissions_cols,
            parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
            nrows=self.n_samples,
        )
        pat = self._load_csv(
            "patients.csv.gz",
            usecols=["subject_id", "gender", "anchor_age", "anchor_year_group"],
        )
        if self.adm.empty or pat.empty:
            raise ValueError("Required files missing: admissions.csv.gz or patients.csv.gz")

        self.adm["los_hours"] = (
            self.adm["dischtime"] - self.adm["admittime"]
        ).dt.total_seconds() / 3600.0
        self.adm = self.adm.sort_values(["subject_id", "admittime"])
        self.adm["next_admit"] = self.adm.groupby("subject_id")["admittime"].shift(-1)
        self.adm["readmit_30"] = (
            ((self.adm["next_admit"] - self.adm["dischtime"]).dt.days <= 30).fillna(False).astype("int8")
        )

        self.df = self.adm[["subject_id", "hadm_id", "readmit_30", "admittime"]].copy()
        self.df = self.df.merge(pat, on="subject_id", how="left")
        self.df["gender"] = self.df["gender"].map({"M": 1, "F": 0}).fillna(0).astype("int8")
        self.df["anchor_age"] = self.df["anchor_age"].fillna(self.df["anchor_age"].median()).astype("float32")
        self.df["died_during_admit"] = self.adm["deathtime"].notna().astype("int8").values
        self.df["los_hours"] = self.adm["los_hours"].astype("float32").values
        self.df["los_days"] = (self.df["los_hours"] / 24.0).astype("float32")
        self.df["had_ed"] = self.adm["edregtime"].notna().astype("int8").values
        self.df["ed_time_hours"] = (
            (self.adm["edouttime"] - self.adm["edregtime"]).dt.total_seconds() / 3600.0
        ).fillna(0).astype("float32")

        ay_map = {
            "2008 - 2010": 0,
            "2011 - 2013": 1,
            "2014 - 2016": 2,
            "2017 - 2019": 3,
            "2020 - 2022": 4,
        }
        self.df["anchor_year_group"] = self.df["anchor_year_group"].map(ay_map).fillna(-1).astype("int8")
        self._map_categories()

        self.cohort_hadm = set(self.df["hadm_id"].astype(int).tolist())
        self._safe_numeric_fill()

    def _map_categories(self) -> None:
        self.df["admission_type"] = self.adm["admission_type"].map(
            {
                "AMBULATORY OBSERVATION": 0,
                "DIRECT EMER.": 1,
                "DIRECT OBSERVATION": 2,
                "ELECTIVE": 3,
                "EU OBSERVATION": 4,
                "EW EMER.": 5,
                "OBSERVATION ADMIT": 6,
                "SURGICAL SAME DAY ADMISSION": 7,
                "URGENT": 8,
            }
        ).fillna(0).astype("int16")
        self.df["admission_location"] = self.adm["admission_location"].map(
            {
                "AMBULATORY SURGERY TRANSFER": 0,
                "CLINIC REFERRAL": 1,
                "EMERGENCY ROOM": 2,
                "PACU": 3,
                "PHYSICIAN REFERRAL": 4,
                "PROCEDURE SITE": 5,
                "TRANSFER FROM HOSPITAL": 6,
                "TRANSFER FROM SKILLED NURSING FACILITY": 7,
                "WALK-IN/SELF REFERRAL": 8,
            }
        ).fillna(2).astype("int16")
        self.df["discharge_location"] = self.adm["discharge_location"].map(
            {
                "DEAD": 0,
                "DIED": 0,
                "HOME": 1,
                "HOME HEALTH CARE": 2,
                "REHAB": 3,
                "SKILLED NURSING FACILITY": 4,
            }
        ).fillna(1).astype("int16")
        self.df["insurance"] = self.adm["insurance"].map({"Medicaid": 0, "Medicare": 1, "Other": 2}).fillna(2).astype("int16")

    def extract_icu_features(self) -> None:
        logger.info("Step 2/10: ICU")
        icu = self._load_csv("icustays.csv.gz", usecols=["hadm_id", "stay_id", "los"])
        if icu.empty:
            return
        icu = icu[icu["hadm_id"].isin(self.cohort_hadm)]
        icu_agg = icu.groupby("hadm_id", as_index=False).agg(
            icu_count=("stay_id", "count"),
            icu_los_sum=("los", "sum"),
            icu_los_mean=("los", "mean"),
            icu_los_max=("los", "max"),
        )
        self._merge_hadm_features(icu_agg)
        self.df["had_icu"] = (self.df["icu_count"] > 0).astype("int8")
        self.df["icu_los_ratio"] = (self.df["icu_los_sum"] / (self.df["los_days"] + 0.5)).astype("float32")

    def extract_lab_features(self) -> None:
        logger.info("Step 3/10: labevents")
        important_labs = {
            50912,
            50902,
            50882,
            50931,
            50960,
            50971,
            50983,
            51006,
            51221,
            51222,
            50813,
            50820,
            50821,
            51265,
            51301,
        }
        labs = self._load_csv("labevents.csv.gz", usecols=["hadm_id", "itemid", "valuenum"])
        if labs.empty:
            return
        labs = labs[
            labs["hadm_id"].isin(self.cohort_hadm)
            & labs["itemid"].isin(important_labs)
            & labs["valuenum"].notna()
        ]
        if labs.empty:
            return
        stats = labs.groupby("hadm_id", as_index=False)["valuenum"].agg(["count", "mean", "std", "min", "max", "median"])
        stats.columns = ["hadm_id", "lab_count", "lab_mean", "lab_std", "lab_min", "lab_max", "lab_median"]
        self._merge_hadm_features(stats)
        self.df["lab_cv"] = (self.df["lab_std"] / (self.df["lab_mean"].abs() + 0.1)).astype("float32")

    def extract_pharmacy_and_rx(self) -> None:
        logger.info("Step 4/10: prescriptions")
        rx = self._load_csv(
            "prescriptions.csv.gz",
            usecols=["hadm_id", "drug", "formulary_drug_cd"],
        )
        if rx.empty:
            return
        rx = rx[rx["hadm_id"].isin(self.cohort_hadm)]
        if rx.empty:
            return
        stats = rx.groupby("hadm_id", as_index=False).agg(
            rx_count=("drug", "count"),
            unique_drugs=("formulary_drug_cd", "nunique"),
        )
        self._merge_hadm_features(stats)

    def extract_microbiology(self) -> None:
        logger.info("Step 5/10: microbiology")
        micro = self._load_csv(
            "microbiologyevents.csv.gz",
            usecols=["hadm_id", "micro_specimen_id", "org_name"],
        )
        if micro.empty:
            return
        micro = micro[micro["hadm_id"].isin(self.cohort_hadm)]
        if micro.empty:
            return
        micro_stats = micro.groupby("hadm_id", as_index=False).agg(
            micro_count=("micro_specimen_id", "count"),
            unique_cultures=("org_name", "nunique"),
        )
        hr = micro["org_name"].str.contains("STAPHYLOCOCCUS|PSEUDOMONAS|CLOSTRIDIUM", case=False, na=False)
        hr_stats = micro.loc[hr].groupby("hadm_id", as_index=False).size().rename(columns={"size": "high_risk_culture"})
        self._merge_hadm_features(micro_stats)
        self._merge_hadm_features(hr_stats)

    def extract_severity_and_drg(self) -> None:
        logger.info("Step 6/10: DRG")
        drg = self._load_csv("drgcodes.csv.gz", usecols=["hadm_id", "drg_type"])
        if drg.empty:
            return
        drg = drg[drg["hadm_id"].isin(self.cohort_hadm)]
        if drg.empty:
            return
        drg["drg_weight"] = drg["drg_type"].map({"MS": 1, "APR": 2}).fillna(0)
        drg_stats = drg.groupby("hadm_id", as_index=False)["drg_weight"].max().rename(columns={"drg_weight": "drg_intensity"})
        self._merge_hadm_features(drg_stats)

    def extract_service_history(self) -> None:
        logger.info("Step 7/10: services + transfers")
        services = self._load_csv("services.csv.gz", usecols=["hadm_id"])
        if not services.empty:
            services = services[services["hadm_id"].isin(self.cohort_hadm)]
            svc_counts = services.groupby("hadm_id", as_index=False).size().rename(columns={"size": "service_count"})
            self._merge_hadm_features(svc_counts)

        transfers = self._load_csv("transfers.csv.gz", usecols=["hadm_id"])
        if not transfers.empty:
            transfers = transfers[transfers["hadm_id"].isin(self.cohort_hadm)]
            tr_counts = transfers.groupby("hadm_id", as_index=False).size().rename(columns={"size": "transfer_count"})
            self._merge_hadm_features(tr_counts)

    def _select_top_contributors(
        self,
        source_df: pd.DataFrame,
        value_col: str,
        min_count: int,
        top_k: int,
    ) -> list[str]:
        if source_df.empty or value_col not in source_df:
            return []
        base_rate = float(self.df["readmit_30"].mean())
        if np.isnan(base_rate):
            return []

        small = source_df[["hadm_id", value_col]].dropna().drop_duplicates()
        counts = small[value_col].value_counts()
        candidates = counts[counts >= min_count].head(600).index
        if len(candidates) == 0:
            return []
        small = small[small[value_col].isin(candidates)]
        small = small.merge(self.df[["hadm_id", "readmit_30"]], on="hadm_id", how="left")
        stats = small.groupby(value_col)["readmit_30"].agg(["mean", "count"]).reset_index()
        stats["score"] = (stats["mean"] - base_rate).abs() * np.log1p(stats["count"])
        return stats.sort_values("score", ascending=False)[value_col].head(top_k).astype(str).tolist()

    def _pivot_binary(self, source_df: pd.DataFrame, col: str, top_items: Iterable[str], prefix: str) -> None:
        top_items = list(top_items)
        if not top_items:
            return
        subset = source_df[source_df[col].isin(top_items)][["hadm_id", col]].copy()
        if subset.empty:
            return
        subset["value"] = 1
        pivot = (
            subset.pivot_table(index="hadm_id", columns=col, values="value", aggfunc="max", fill_value=0)
            .reset_index()
        )
        pivot.columns = ["hadm_id"] + [self._normalize_colname(name, prefix=prefix) for name in pivot.columns[1:]]
        self._merge_hadm_features(pivot)

    def extract_codes_and_meds(self) -> None:
        logger.info("Step 8/10: diagnoses + procedures + emar")
        dx = self._load_csv("diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code"])
        if not dx.empty:
            dx = dx[dx["hadm_id"].isin(self.cohort_hadm)]
            self._merge_hadm_features(dx.groupby("hadm_id", as_index=False).size().rename(columns={"size": "dx_count"}))
            top_dx = self._select_top_contributors(dx, value_col="icd_code", min_count=40, top_k=160)
            self._pivot_binary(dx, col="icd_code", top_items=top_dx, prefix="dx")

        proc = self._load_csv("procedures_icd.csv.gz", usecols=["hadm_id", "icd_code"])
        if not proc.empty:
            proc = proc[proc["hadm_id"].isin(self.cohort_hadm)]
            self._merge_hadm_features(proc.groupby("hadm_id", as_index=False).size().rename(columns={"size": "proc_count"}))
            top_proc = self._select_top_contributors(proc, value_col="icd_code", min_count=20, top_k=90)
            self._pivot_binary(proc, col="icd_code", top_items=top_proc, prefix="proc")

        emar = self._load_csv("emar.csv.gz", usecols=["hadm_id", "medication"])
        if not emar.empty:
            emar = emar[emar["hadm_id"].isin(self.cohort_hadm)]
            emar["medication"] = emar["medication"].astype(str).str.lower().str.strip()
            emar = emar[emar["medication"].ne("")]
            top_meds = self._select_top_contributors(emar, value_col="medication", min_count=80, top_k=90)
            self._pivot_binary(emar, col="medication", top_items=top_meds, prefix="med")

    def extract_note_and_bhc_features(self) -> None:
        logger.info("Step 9/10: notes + bhc")
        discharge = self._load_csv("discharge.csv.gz", usecols=["note_id", "hadm_id", "text"])
        if not discharge.empty and "hadm_id" in discharge.columns:
            d = discharge[discharge["hadm_id"].isin(self.cohort_hadm)].copy()
            text_col = "text" if "text" in d.columns else None
            if text_col:
                d["note_len"] = d[text_col].astype(str).str.len().clip(0, 30_000)
                note_agg = d.groupby("hadm_id", as_index=False).agg(
                    discharge_note_count=("hadm_id", "size"),
                    discharge_note_len_mean=("note_len", "mean"),
                    discharge_note_len_max=("note_len", "max"),
                )
                self._merge_hadm_features(note_agg)

        radiology = self._load_csv("radiology.csv.gz", usecols=["hadm_id", "text"])
        if not radiology.empty and "hadm_id" in radiology.columns:
            r = radiology[radiology["hadm_id"].isin(self.cohort_hadm)].copy()
            text_col = "text" if "text" in r.columns else None
            if text_col:
                r["rad_len"] = r[text_col].astype(str).str.len().clip(0, 30_000)
                rad_agg = r.groupby("hadm_id", as_index=False).agg(
                    radiology_note_count=("hadm_id", "size"),
                    radiology_note_len_mean=("rad_len", "mean"),
                )
                self._merge_hadm_features(rad_agg)

        bhc = self._load_csv("mimic-iv-bhc.csv", usecols=["note_id", "input_tokens", "target_tokens"])
        if bhc.empty:
            return
        needed = {"note_id", "input_tokens", "target_tokens"}
        if not needed.issubset(set(bhc.columns)):
            return
        if "note_id" not in discharge.columns or "hadm_id" not in discharge.columns:
            return

        note_to_hadm = discharge[["note_id", "hadm_id"]].dropna().drop_duplicates()
        bhc = bhc[list(needed)].dropna(subset=["note_id"]).merge(note_to_hadm, on="note_id", how="inner")
        bhc = bhc[bhc["hadm_id"].isin(self.cohort_hadm)]
        if bhc.empty:
            return
        bhc_agg = bhc.groupby("hadm_id", as_index=False).agg(
            bhc_input_tokens_mean=("input_tokens", "mean"),
            bhc_target_tokens_mean=("target_tokens", "mean"),
            bhc_target_tokens_max=("target_tokens", "max"),
        )
        self._merge_hadm_features(bhc_agg)

    def add_historical_and_engineered(self) -> None:
        logger.info("Step 10/10: historical + engineered")
        self.df = self.df.sort_values(["subject_id", "admittime"])
        self.df["prev_admissions"] = self.df.groupby("subject_id").cumcount().astype("int16")
        self.df["prev_readmit_rate"] = (
            self.df.groupby("subject_id")["readmit_30"]
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0)
            .astype("float32")
        )
        self.df["days_since_last"] = (
            self.df.groupby("subject_id")["admittime"].diff().dt.days.fillna(999).astype("float32")
        )
        self.df["polypharmacy"] = (self.df.get("rx_count", 0) > 10).astype("int8")
        self.df["complexity_score"] = (
            self.df.get("dx_count", 0) + self.df.get("proc_count", 0) + self.df.get("service_count", 0)
        ).astype("float32")
        self.df["severity_score"] = (self.df.get("icu_count", 0) * 3 + self.df["complexity_score"]).astype("float32")
        self.df["high_risk"] = (
            (self.df.get("icu_count", 0) > 0) | (self.df["los_days"] > 10) | (self.df["anchor_age"] >= 80)
        ).astype("int8")
        self.df["admission_hour"] = self.df["admittime"].dt.hour.fillna(0).astype("int8")
        self.df["admission_dow"] = self.df["admittime"].dt.dayofweek.fillna(0).astype("int8")
        self.df["is_weekend"] = (self.df["admission_dow"] >= 5).astype("int8")
        self.df = self.df.drop(columns=["admittime"], errors="ignore")
        self._safe_numeric_fill()

    def select_top_features(self, max_features: int = 220) -> None:
        """Keep strongest numeric contributors to readmit_30 (target retained)."""
        target = "readmit_30"
        protected = {"subject_id", "hadm_id", target}
        numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in protected]
        if not numeric_cols:
            return
        contrib = self.df[numeric_cols].corrwith(self.df[target]).abs().fillna(0).sort_values(ascending=False)
        keep = list(contrib.head(max_features).index)
        self.df = self.df[["subject_id", "hadm_id", target] + keep]

    def run_full_pipeline(self) -> bool:
        try:
            self.extract_core_demographics()
            self.extract_icu_features()
            self.extract_lab_features()
            self.extract_pharmacy_and_rx()
            self.extract_microbiology()
            self.extract_severity_and_drg()
            self.extract_service_history()
            self.extract_codes_and_meds()
            self.extract_note_and_bhc_features()
            self.add_historical_and_engineered()
            self.select_top_features(max_features=self.top_sparse_features)

            logger.info("Extraction complete. Final shape: %s", self.df.shape)
            self.df.to_csv(FEATURES_CSV, index=False)
            logger.info("Saved features to %s", FEATURES_CSV)
            return True
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc)
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    MIMICExtractor().run_full_pipeline()
