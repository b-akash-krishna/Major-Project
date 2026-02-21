# src/01_extract.py
"""
TRANCE Framework - Full Dataset MIMIC-IV Feature Extraction v2
Key additions over v1:
  - Full dataset (N_SAMPLES=None) with chunked loading for RAM safety
  - chartevents vitals (last 24h: HR, BP, SpO2, Temp, RR, GCS)
  - Per-item lab stats for 15 key labs
  - Proper cohort exclusion: planned readmissions filtered out
  - ICD 3-char category pivots (less sparse, better signal)
  - Discharge-hour / weekend features
  - First-visit vs repeat-visitor features
  - Clinical threshold flags (hypotension, AKI, anemia, etc.)
Optimized for 8GB RAM with chunked I/O.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import traceback
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from .config import DATA_DIR, FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_IV_DIR, MIMIC_NOTE_DIR
except ImportError:
    from config import DATA_DIR, FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_IV_DIR, MIMIC_NOTE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_SAMPLES    = None        # None = full ~546k admissions; set int for quick test
LAB_CHUNK    = 3_000_000
CHART_CHUNK  = 2_000_000
EMAR_CHUNK   = 1_000_000
TOP_DX_CATS  = 50          # ICD 3-char categories
TOP_PROC     = 80
TOP_MED      = 80

VITAL_ITEMS = {
    220045: "hr",    220179: "sbp",   220180: "dbp",
    220277: "spo2",  223761: "temp_f", 220210: "rr",
    220739: "gcs_e", 223900: "gcs_v", 223901: "gcs_m",
    220621: "glucose",
}

KEY_LAB_ITEMS = {
    50912: "creatinine", 50902: "chloride",  50882: "bicarb",
    50931: "glucose",    50971: "potassium", 50983: "sodium",
    51006: "bun",        51221: "hematocrit",51222: "hemoglobin",
    51265: "platelets",  51301: "wbc",       50813: "lactate",
    50820: "ph",         50821: "pao2",      50818: "paco2",
}

HIGH_RISK_ORGS = r"STAPHYLOCOCCUS AUREUS|PSEUDOMONAS|CLOSTRIDIUM|CANDIDA|KLEBSIELLA|ACINETOBACTER"


class MIMICExtractor:

    def __init__(self, n_samples=N_SAMPLES):
        self.n_samples = n_samples
        self.file_index: Dict[str, str] = {}
        self.df: Optional[pd.DataFrame] = None
        self.adm: Optional[pd.DataFrame] = None
        self.cohort_hadm: set = set()
        self.cohort_subject: set = set()
        os.makedirs(DATA_DIR, exist_ok=True)
        self._build_index()

    def _build_index(self):
        for base in [MIMIC_IV_DIR, MIMIC_NOTE_DIR, MIMIC_BHC_DIR]:
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for f in files:
                    self.file_index.setdefault(f, os.path.join(root, f))
        logger.info("Indexed %d files.", len(self.file_index))

    def _path(self, fn):
        p = self.file_index.get(fn)
        if not p:
            logger.warning("Not found: %s", fn)
        return p

    def _load(self, fn, usecols=None, parse_dates=None, nrows=None):
        p = self._path(fn)
        if not p:
            return pd.DataFrame()
        try:
            logger.info("  Loading %s ...", fn)
            return pd.read_csv(p, usecols=usecols, parse_dates=parse_dates,
                               nrows=nrows, low_memory=True)
        except Exception as e:
            logger.error("  Failed %s: %s", fn, e)
            return pd.DataFrame()

    def _merge(self, other, on="hadm_id", fill=0.0):
        if other.empty:
            return
        self.df = self.df.merge(other, on=on, how="left")
        for c in other.columns:
            if c != on and c in self.df.columns and np.issubdtype(self.df[c].dtype, np.number):
                self.df[c] = self.df[c].fillna(fill)

    def _col(self, raw, prefix, ml=36):
        return f"{prefix}_{re.sub(r'[^a-zA-Z0-9_]+','_',str(raw).strip().lower())[:ml]}"

    def _fill(self):
        self.df = self.df.replace([np.inf, -np.inf], 0)
        num = self.df.select_dtypes(include=[np.number]).columns
        self.df[num] = self.df[num].fillna(0)

    def _pivot_binary(self, src, col, top_items, prefix):
        if not top_items or src.empty:
            return
        src = src[src[col].isin(top_items)][["hadm_id", col]].drop_duplicates()
        if src.empty:
            return
        src = src.copy()
        src["_v"] = 1
        piv = src.pivot_table("_v", "hadm_id", col, aggfunc="max", fill_value=0).reset_index()
        piv.columns = ["hadm_id"] + [self._col(c, prefix) for c in piv.columns[1:]]
        self._merge(piv)

    def _top_contributors(self, src, val_col, min_count=50, top_k=100):
        if src.empty or val_col not in src.columns:
            return []
        base = float(self.df["readmit_30"].mean())
        counts = src[val_col].value_counts()
        cands = counts[counts >= min_count].head(2000).index
        sub = src[src[val_col].isin(cands)].merge(
            self.df[["hadm_id", "readmit_30"]], on="hadm_id", how="left")
        s = sub.groupby(val_col)["readmit_30"].agg(["mean", "count"]).reset_index()
        s["score"] = (s["mean"] - base).abs() * np.log1p(s["count"])
        return s.sort_values("score", ascending=False)[val_col].head(top_k).astype(str).tolist()

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    def extract_core(self):
        logger.info("=== STEP 1: Admissions + Patients ===")
        adm_cols = ["subject_id","hadm_id","admittime","dischtime","deathtime",
                    "edregtime","edouttime","admission_type","admission_location",
                    "discharge_location","insurance","language","marital_status","race"]
        self.adm = self._load("admissions.csv.gz", usecols=adm_cols,
                              parse_dates=["admittime","dischtime","deathtime","edregtime","edouttime"],
                              nrows=self.n_samples)
        pat = self._load("patients.csv.gz",
                         usecols=["subject_id","gender","anchor_age","anchor_year","anchor_year_group","dod"])
        if self.adm.empty or pat.empty:
            raise ValueError("admissions/patients missing")

        self.adm = self.adm.sort_values(["subject_id","admittime"]).reset_index(drop=True)
        self.adm["next_admittime"] = self.adm.groupby("subject_id")["admittime"].shift(-1)
        self.adm["next_adm_type"]  = self.adm.groupby("subject_id")["admission_type"].shift(-1)
        self.adm["days_to_next"]   = (
            (self.adm["next_admittime"] - self.adm["dischtime"]).dt.total_seconds() / 86400
        )
        self.adm["died_hospital"] = self.adm["deathtime"].notna()
        # Exclude planned next admissions (elective/surgical same day)
        planned = {"ELECTIVE","SURGICAL SAME DAY ADMISSION"}
        self.adm["next_planned"] = self.adm["next_adm_type"].isin(planned)
        self.adm["readmit_30"] = (
            (self.adm["days_to_next"] >= 0) & (self.adm["days_to_next"] <= 30)
            & (~self.adm["died_hospital"]) & (~self.adm["next_planned"])
        ).fillna(False).astype("int8")

        self.adm = self.adm.merge(pat, on="subject_id", how="left")

        self.adm["los_hours"]   = ((self.adm["dischtime"]-self.adm["admittime"]).dt.total_seconds()/3600).clip(lower=0)
        self.adm["los_days"]    = self.adm["los_hours"]/24
        self.adm["ed_time_hours"] = ((self.adm["edouttime"]-self.adm["edregtime"]).dt.total_seconds()/3600).fillna(0).clip(lower=0)
        self.adm["had_ed"]      = self.adm["edregtime"].notna().astype("int8")
        self.adm["died_during_admit"] = self.adm["deathtime"].notna().astype("int8")
        self.adm["admission_hour"]  = self.adm["admittime"].dt.hour.astype("int8")
        self.adm["admission_dow"]   = self.adm["admittime"].dt.dayofweek.astype("int8")
        self.adm["admission_month"] = self.adm["admittime"].dt.month.astype("int8")
        self.adm["is_weekend"]      = (self.adm["admission_dow"]>=5).astype("int8")
        self.adm["is_night"]        = ((self.adm["admission_hour"]<7)|(self.adm["admission_hour"]>=22)).astype("int8")
        self.adm["discharge_hour"]  = self.adm["dischtime"].dt.hour.astype("int8")
        self.adm["discharge_dow"]   = self.adm["dischtime"].dt.dayofweek.astype("int8")
        self.adm["discharge_weekend"]= (self.adm["discharge_dow"]>=5).astype("int8")

        ay_map = {"2008 - 2010":0,"2011 - 2013":1,"2014 - 2016":2,"2017 - 2019":3,"2020 - 2022":4}
        self.adm["anchor_year_group"] = self.adm["anchor_year_group"].map(ay_map).fillna(-1).astype("int8")
        self.adm["gender"] = self.adm["gender"].map({"M":1,"F":0}).fillna(0).astype("int8")
        self.adm["anchor_age"] = pd.to_numeric(self.adm["anchor_age"],errors="coerce").fillna(60).astype("float32")
        self.adm["age_group"] = pd.cut(self.adm["anchor_age"],bins=[0,40,55,65,75,85,200],
                                       labels=[0,1,2,3,4,5],right=False).astype("float32")

        # Categorical encodings
        adm_type_map = {"AMBULATORY OBSERVATION":0,"DIRECT EMER.":1,"DIRECT OBSERVATION":2,
                        "ELECTIVE":3,"EU OBSERVATION":4,"EW EMER.":5,
                        "OBSERVATION ADMIT":6,"SURGICAL SAME DAY ADMISSION":7,"URGENT":8}
        adm_loc_map  = {"AMBULATORY SURGERY TRANSFER":0,"CLINIC REFERRAL":1,"EMERGENCY ROOM":2,
                        "PACU":3,"PHYSICIAN REFERRAL":4,"PROCEDURE SITE":5,
                        "TRANSFER FROM HOSPITAL":6,"TRANSFER FROM SKILLED NURSING FACILITY":7,"WALK-IN/SELF REFERRAL":8}
        disc_loc_map = {"DEAD":0,"DIED":0,"HOME":1,"HOME HEALTH CARE":2,"REHAB":3,
                        "SKILLED NURSING FACILITY":4,"ASSISTED LIVING":5,
                        "CHRONIC/LONG TERM ACUTE CARE":6,"HOSPICE":7,"ACUTE HOSPITAL":8,"OTHER FACILITY":9}
        ins_map = {"Medicaid":0,"Medicare":1,"Other":2}
        self.adm["admission_type"]     = self.adm["admission_type"].map(adm_type_map).fillna(0).astype("int8")
        self.adm["admission_location"] = self.adm["admission_location"].map(adm_loc_map).fillna(2).astype("int8")
        self.adm["discharge_location"] = self.adm["discharge_location"].map(disc_loc_map).fillna(1).astype("int8")
        self.adm["insurance"]          = self.adm["insurance"].map(ins_map).fillna(2).astype("int8")
        for col,enc in [("marital_status","marital_enc"),("race","race_enc"),("language","language_enc")]:
            if col in self.adm.columns:
                freq = self.adm[col].value_counts(normalize=True)
                self.adm[enc] = self.adm[col].map(freq).fillna(0).astype("float32")

        keep = ["subject_id","hadm_id","readmit_30","admittime",
                "gender","anchor_age","anchor_year_group","age_group",
                "los_hours","los_days","ed_time_hours","had_ed","died_during_admit",
                "admission_type","admission_location","discharge_location","insurance",
                "marital_enc","race_enc","language_enc",
                "admission_hour","admission_dow","admission_month","is_weekend","is_night",
                "discharge_hour","discharge_dow","discharge_weekend"]
        keep = [c for c in keep if c in self.adm.columns]
        self.df = self.adm[keep].copy()
        self.cohort_hadm    = set(self.df["hadm_id"].astype(int))
        self.cohort_subject = set(self.df["subject_id"].astype(int))
        logger.info("Cohort: %d admissions | readmit=%.2f%%",
                    len(self.df), self.df["readmit_30"].mean()*100)

    # ── STEP 2 : ICU ──────────────────────────────────────────────────────────
    def extract_icu(self):
        logger.info("=== STEP 2: ICU ===")
        icu = self._load("icustays.csv.gz",
                         usecols=["hadm_id","stay_id","first_careunit","last_careunit","los"])
        if icu.empty: return
        icu = icu[icu["hadm_id"].isin(self.cohort_hadm)]
        high_acuity = {"MICU","SICU","CSRU","CCU","TSICU","NICU",
                       "Neuro Surgical ICU","Medical/Surgical (Neuro) ICU","Trauma SICU"}
        icu["high_acuity"] = icu["first_careunit"].isin(high_acuity).astype("int8")
        agg = icu.groupby("hadm_id",as_index=False).agg(
            icu_count=("stay_id","count"), icu_los_sum=("los","sum"),
            icu_los_mean=("los","mean"),   icu_los_max=("los","max"),
            icu_los_std=("los","std"),     icu_high_acuity=("high_acuity","max"),
            icu_unique_units=("first_careunit","nunique"))
        agg["icu_los_std"] = agg["icu_los_std"].fillna(0)
        self._merge(agg)
        self.df["had_icu"]       = (self.df["icu_count"]>0).astype("int8")
        self.df["multiple_icu"]  = (self.df["icu_count"]>1).astype("int8")
        self.df["icu_los_ratio"] = (self.df["icu_los_sum"]/(self.df["los_days"]+0.01)).astype("float32")

    # ── STEP 3 : chartevents vitals ───────────────────────────────────────────
    def extract_vitals(self):
        logger.info("=== STEP 3: Chartevents Vitals (chunked) ===")
        path = self._path("chartevents.csv.gz")
        if not path:
            logger.warning("chartevents.csv.gz not found — skipping (reduces AUROC ~2-3%).")
            return

        target_items = set(VITAL_ITEMS.keys())
        item_to_name = VITAL_ITEMS
        disch_map    = dict(zip(self.adm["hadm_id"], self.adm["dischtime"]))
        agg_dict: Dict[int, Dict[str, list]] = {}

        try:
            reader = pd.read_csv(path, usecols=["hadm_id","itemid","valuenum","charttime"],
                                 chunksize=CHART_CHUNK, low_memory=True, parse_dates=["charttime"])
            for ci, chunk in enumerate(reader):
                chunk = chunk[chunk["hadm_id"].isin(self.cohort_hadm)]
                chunk = chunk[chunk["itemid"].isin(target_items)]
                chunk = chunk[chunk["valuenum"].notna()]
                if chunk.empty: continue
                chunk["dischtime_"] = chunk["hadm_id"].map(disch_map)
                chunk["hbdc"] = (chunk["dischtime_"]-chunk["charttime"]).dt.total_seconds()/3600
                chunk = chunk[(chunk["hbdc"]>=0)&(chunk["hbdc"]<=24)]
                chunk["vname"] = chunk["itemid"].map(item_to_name)
                for (h,v), grp in chunk.groupby(["hadm_id","vname"]):
                    agg_dict.setdefault(int(h),{}).setdefault(v,[]).extend(grp["valuenum"].tolist())
                del chunk
                if ci%5==0: gc.collect()
                if ci%20==0: logger.info("  chart chunk %d",ci)
        except Exception as e:
            logger.error("Chartevents failed: %s", e); return

        if not agg_dict: return
        rows = []
        vnames = list(VITAL_ITEMS.values())
        for hadm_id, vitals in agg_dict.items():
            row = {"hadm_id": hadm_id}
            for v in vnames:
                vals = vitals.get(v,[])
                if vals:
                    arr=np.array(vals)
                    row[f"v_{v}_mean"]=float(np.mean(arr)); row[f"v_{v}_std"]=float(np.std(arr)) if len(arr)>1 else 0.
                    row[f"v_{v}_min"]=float(np.min(arr));  row[f"v_{v}_max"]=float(np.max(arr))
                    row[f"v_{v}_n"]=len(arr)
                else:
                    for s in ["mean","std","min","max","n"]: row[f"v_{v}_{s}"]=0.
            rows.append(row)
        vdf = pd.DataFrame(rows)
        # Clinical threshold flags
        if "v_sbp_mean" in vdf.columns:
            vdf["hypotension"]=(vdf["v_sbp_mean"]<90).astype("int8")
            vdf["hypertension"]=(vdf["v_sbp_mean"]>160).astype("int8")
        if "v_spo2_mean" in vdf.columns:
            vdf["hypoxia"]=(vdf["v_spo2_mean"]<92).astype("int8")
        if "v_hr_mean" in vdf.columns:
            vdf["tachycardia"]=(vdf["v_hr_mean"]>100).astype("int8")
            vdf["bradycardia"]=(vdf["v_hr_mean"]<60).astype("int8")
        if "v_rr_mean" in vdf.columns:
            vdf["tachypnea"]=(vdf["v_rr_mean"]>20).astype("int8")
        if "v_temp_f_mean" in vdf.columns:
            vdf["fever"]=(vdf["v_temp_f_mean"]>100.4).astype("int8")
        if "v_gcs_e_mean" in vdf.columns:
            vdf["gcs_total"]=vdf.get("v_gcs_e_mean",0)+vdf.get("v_gcs_v_mean",0)+vdf.get("v_gcs_m_mean",0)
            vdf["gcs_low"]=(vdf["gcs_total"]<10).astype("int8")
        self._merge(vdf)
        del agg_dict, rows, vdf; gc.collect()

    # ── STEP 4 : labs ─────────────────────────────────────────────────────────
    def extract_labs(self):
        logger.info("=== STEP 4: Lab Events (chunked) ===")
        path = self._path("labevents.csv.gz")
        if not path: return
        per_item: Dict[int, Dict[str,list]] = {}
        abn_dict: Dict[int,int] = {}
        cnt_dict: Dict[int,int] = {}
        target_items = set(KEY_LAB_ITEMS.keys())
        try:
            reader = pd.read_csv(path, usecols=["hadm_id","itemid","valuenum","flag"],
                                 chunksize=LAB_CHUNK, low_memory=True)
            for ci, chunk in enumerate(reader):
                chunk = chunk[chunk["hadm_id"].isin(self.cohort_hadm)]
                chunk = chunk[chunk["valuenum"].notna()]
                if chunk.empty: continue
                for h,n in chunk.groupby("hadm_id")["valuenum"].count().items():
                    cnt_dict[int(h)] = cnt_dict.get(int(h),0)+int(n)
                for h,fl in chunk.groupby("hadm_id")["flag"].apply(
                    lambda x:(x.fillna("")=="abnormal").sum()).items():
                    abn_dict[int(h)] = abn_dict.get(int(h),0)+int(fl)
                key = chunk[chunk["itemid"].isin(target_items)].copy()
                key["lname"] = key["itemid"].map(KEY_LAB_ITEMS)
                for (h,ln),grp in key.groupby(["hadm_id","lname"]):
                    per_item.setdefault(int(h),{}).setdefault(ln,[]).extend(grp["valuenum"].tolist())
                del chunk, key
                if ci%5==0: gc.collect()
                if ci%10==0: logger.info("  lab chunk %d",ci)
        except Exception as e:
            logger.error("Labs failed: %s", e); return

        rows = []
        lnames = list(KEY_LAB_ITEMS.values())
        for hadm_id in self.cohort_hadm:
            row = {"hadm_id":hadm_id,
                   "lab_count":cnt_dict.get(hadm_id,0),
                   "lab_abnormal_count":abn_dict.get(hadm_id,0)}
            row["lab_abnormal_rate"] = row["lab_abnormal_count"]/max(row["lab_count"],1)
            labs = per_item.get(hadm_id,{})
            for n in lnames:
                vals=labs.get(n,[])
                if vals:
                    arr=np.array(vals)
                    row[f"lab_{n}_mean"]=float(np.mean(arr)); row[f"lab_{n}_last"]=float(arr[-1])
                    row[f"lab_{n}_min"]=float(np.min(arr));   row[f"lab_{n}_max"]=float(np.max(arr))
                    row[f"lab_{n}_n"]=len(arr);               row[f"lab_{n}_range"]=float(np.ptp(arr))
                else:
                    for s in ["mean","last","min","max","n","range"]: row[f"lab_{n}_{s}"]=0.
            rows.append(row)
        ldf = pd.DataFrame(rows)
        # Clinical thresholds
        if "lab_creatinine_last" in ldf.columns: ldf["aki"]=(ldf["lab_creatinine_last"]>1.5).astype("int8")
        if "lab_lactate_last"    in ldf.columns: ldf["hyperlactatemia"]=(ldf["lab_lactate_last"]>2.0).astype("int8")
        if "lab_hemoglobin_last" in ldf.columns: ldf["anemia"]=(ldf["lab_hemoglobin_last"]<10.0).astype("int8")
        if "lab_wbc_last"        in ldf.columns:
            ldf["leukocytosis"]=(ldf["lab_wbc_last"]>11.0).astype("int8")
            ldf["leukopenia"]=(ldf["lab_wbc_last"]<4.0).astype("int8")
        if "lab_platelets_last"  in ldf.columns: ldf["thrombocytopenia"]=(ldf["lab_platelets_last"]<100).astype("int8")
        if "lab_sodium_last"     in ldf.columns:
            ldf["hyponatremia"]=(ldf["lab_sodium_last"]<135).astype("int8")
            ldf["hypernatremia"]=(ldf["lab_sodium_last"]>145).astype("int8")
        self._merge(ldf)
        del per_item, abn_dict, cnt_dict, rows, ldf; gc.collect()

    # ── STEP 5–11 : same structure as before ──────────────────────────────────
    def extract_pharmacy(self):
        logger.info("=== STEP 5: Prescriptions ===")
        rx = self._load("prescriptions.csv.gz", usecols=["hadm_id","drug","formulary_drug_cd","route"])
        if rx.empty: return
        rx = rx[rx["hadm_id"].isin(self.cohort_hadm)]
        hr = r"WARFARIN|HEPARIN|INSULIN|DIGOXIN|LITHIUM|PHENYTOIN|METHOTREXATE|VANCOMYCIN|AMIODARONE|TACROLIMUS"
        rx["hrm"] = rx["drug"].str.upper().str.contains(hr,na=False).astype("int8")
        agg = rx.groupby("hadm_id",as_index=False).agg(
            rx_count=("drug","count"), unique_drugs=("formulary_drug_cd","nunique"),
            unique_routes=("route","nunique"), high_risk_med_count=("hrm","sum"))
        agg["polypharmacy"]=(agg["rx_count"]>10).astype("int8")
        agg["high_polypharmacy"]=(agg["rx_count"]>20).astype("int8")
        self._merge(agg)

    def extract_microbiology(self):
        logger.info("=== STEP 6: Microbiology ===")
        micro = self._load("microbiologyevents.csv.gz",
                           usecols=["hadm_id","micro_specimen_id","org_name","spec_type_desc","interpretation"])
        if micro.empty: return
        micro = micro[micro["hadm_id"].isin(self.cohort_hadm)]
        micro["is_pos"]  = (micro["org_name"].notna()&(micro["org_name"].str.strip()!="")).astype("int8")
        micro["is_hrisk"]= micro["org_name"].str.upper().str.contains(HIGH_RISK_ORGS,na=False).astype("int8")
        micro["is_res"]  = (micro["interpretation"].fillna("").str.upper()=="R").astype("int8")
        agg = micro.groupby("hadm_id",as_index=False).agg(
            micro_count=("micro_specimen_id","count"), unique_cultures=("org_name","nunique"),
            positive_cultures=("is_pos","sum"), high_risk_org=("is_hrisk","max"), resistant_org=("is_res","max"))
        agg["culture_pos_rate"]=(agg["positive_cultures"]/(agg["micro_count"]+1)).astype("float32")
        self._merge(agg)

    def extract_drg(self):
        logger.info("=== STEP 7: DRG ===")
        drg = self._load("drgcodes.csv.gz", usecols=["hadm_id","drg_type","drg_code","drg_severity","drg_mortality"])
        if drg.empty: return
        drg = drg[drg["hadm_id"].isin(self.cohort_hadm)]
        drg["drg_weight"] = drg["drg_type"].map({"MS":1,"APR":2,"HCFA":1}).fillna(0)
        agg = drg.groupby("hadm_id",as_index=False).agg(
            drg_count=("drg_code","count"), drg_intensity=("drg_weight","max"),
            drg_severity_max=("drg_severity","max"), drg_mortality_max=("drg_mortality","max")).fillna(0)
        self._merge(agg)

    def extract_services_transfers(self):
        logger.info("=== STEP 8: Services + Transfers ===")
        svc = self._load("services.csv.gz", usecols=["hadm_id","curr_service"])
        if not svc.empty:
            svc = svc[svc["hadm_id"].isin(self.cohort_hadm)]
            hr_svc = {"CMED","MED","NMED","NSURG","OMED","SURG","TRAUM","ORTHO"}
            svc["hr_svc"] = svc["curr_service"].isin(hr_svc).astype("int8")
            self._merge(svc.groupby("hadm_id",as_index=False).agg(
                service_count=("curr_service","count"), unique_services=("curr_service","nunique"),
                high_risk_service=("hr_svc","max")))
        tr = self._load("transfers.csv.gz", usecols=["hadm_id","careunit"])
        if not tr.empty:
            tr = tr[tr["hadm_id"].isin(self.cohort_hadm)]
            self._merge(tr.groupby("hadm_id",as_index=False).agg(
                transfer_count=("careunit","count"), unique_careunits=("careunit","nunique")))

    def extract_codes(self):
        logger.info("=== STEP 9: Diagnoses + Procedures ===")
        dx = self._load("diagnoses_icd.csv.gz", usecols=["hadm_id","icd_code","icd_version","seq_num"])
        if not dx.empty:
            dx = dx[dx["hadm_id"].isin(self.cohort_hadm)]
            self._merge(dx.groupby("hadm_id",as_index=False).agg(
                dx_count=("icd_code","count"), dx_unique=("icd_code","nunique"), dx_seq_mean=("seq_num","mean")))
            prim = dx[dx["seq_num"]==1][["hadm_id","icd_code"]].copy()
            prim["dx_cat3"] = prim["icd_code"].str[:3]
            freq = prim["dx_cat3"].value_counts(normalize=True)
            prim["primary_dx_freq"] = prim["dx_cat3"].map(freq).fillna(0).astype("float32")
            self._merge(prim[["hadm_id","primary_dx_freq"]])
            dx["dx_cat3"] = dx["icd_code"].str[:3]
            top_cats = self._top_contributors(
                dx[["hadm_id","dx_cat3"]].rename(columns={"dx_cat3":"val"}), "val", min_count=100, top_k=TOP_DX_CATS)
            self._pivot_binary(dx[["hadm_id","dx_cat3"]].rename(columns={"dx_cat3":"val"}), "val", top_cats, "dxcat")
            self._comorbidities(dx)
            del dx; gc.collect()
        proc = self._load("procedures_icd.csv.gz", usecols=["hadm_id","icd_code","seq_num"])
        if not proc.empty:
            proc = proc[proc["hadm_id"].isin(self.cohort_hadm)]
            self._merge(proc.groupby("hadm_id",as_index=False).agg(
                proc_count=("icd_code","count"), proc_unique=("icd_code","nunique")))
            top_p = self._top_contributors(proc[["hadm_id","icd_code"]].rename(columns={"icd_code":"val"}),
                                           "val", min_count=50, top_k=TOP_PROC)
            self._pivot_binary(proc[["hadm_id","icd_code"]].rename(columns={"icd_code":"val"}), "val", top_p, "proc")
            del proc; gc.collect()

    def _comorbidities(self, dx):
        cm_map = {
            "chf":r"^(428|I50)", "arrhythmia":r"^(427|I4[7-9])", "diabetes":r"^(250|E1[0-4])",
            "hypertension":r"^(40[1-5]|I1[0-6])", "renal_fail":r"^(585|586|N1[89])",
            "copd":r"^(49[0-6]|J4[1-6])", "liver":r"^(57[0-3]|K7[0-4])",
            "cancer":r"^(1[4-9][0-9]|C[0-9])", "depression":r"^(296|311|F3[2-4])",
            "psychosis":r"^(295|297|298|F2[0-9])", "obesity":r"^(278|E66)",
            "sepsis":r"^(99591|99592|A41|R65)", "pneumonia":r"^(48[0-6]|J1[2-8])",
            "stroke":r"^(43[0-8]|I6[0-9])", "mi":r"^(410|41[0-2]|I2[1-2])",
            "dementia":r"^(290|331|F0[0-3])",
        }
        dx["icd_s"] = dx["icd_code"].astype(str)
        frames = []
        for name, pat in cm_map.items():
            has = dx[dx["icd_s"].str.contains(pat,na=False,regex=True)][["hadm_id"]].drop_duplicates()
            has = has.copy(); has[f"cm_{name}"] = np.int8(1); frames.append(has)
        if frames:
            cm = frames[0]
            for f in frames[1:]: cm = cm.merge(f, on="hadm_id", how="outer")
            cm = cm.fillna(0)
            for c in cm.columns:
                if c!="hadm_id": cm[c]=cm[c].astype("int8")
            self._merge(cm)

    def extract_emar(self):
        logger.info("=== STEP 10: EMAR (chunked) ===")
        path = self._path("emar.csv.gz")
        if not path: return
        chunks = []
        try:
            reader = pd.read_csv(path, usecols=["hadm_id","medication"], chunksize=EMAR_CHUNK, low_memory=True)
            for chunk in reader:
                c = chunk[chunk["hadm_id"].isin(self.cohort_hadm)].copy()
                c["medication"] = c["medication"].astype(str).str.lower().str.strip()
                chunks.append(c); del c, chunk; gc.collect()
        except Exception as e:
            logger.error("EMAR: %s", e); return
        if not chunks: return
        emar = pd.concat(chunks, ignore_index=True); del chunks; gc.collect()
        self._merge(emar.groupby("hadm_id",as_index=False).agg(
            med_admin_count=("medication","count"), med_unique=("medication","nunique")))
        top_m = self._top_contributors(emar[["hadm_id","medication"]].rename(columns={"medication":"val"}),
                                       "val", min_count=200, top_k=TOP_MED)
        self._pivot_binary(emar[["hadm_id","medication"]].rename(columns={"medication":"val"}), "val", top_m, "med")
        del emar; gc.collect()

    def extract_misc(self):
        logger.info("=== STEP 11: POE + OMR ===")
        poe = self._load("poe.csv.gz", usecols=["hadm_id","order_type"])
        if not poe.empty:
            poe = poe[poe["hadm_id"].isin(self.cohort_hadm)]
            self._merge(poe.groupby("hadm_id",as_index=False).agg(
                poe_count=("order_type","count"), poe_types=("order_type","nunique")))
        omr = self._load("omr.csv.gz", usecols=["subject_id","result_name","result_value"])
        if not omr.empty:
            omr = omr[omr["subject_id"].isin(self.cohort_subject)]
            bmi = omr[omr["result_name"].str.lower().str.contains("bmi",na=False)].copy()
            bmi["bmi_val"] = pd.to_numeric(bmi["result_value"],errors="coerce")
            bmi_agg = bmi.groupby("subject_id",as_index=False)["bmi_val"].mean().rename(columns={"bmi_val":"bmi"})
            self.df = self.df.merge(bmi_agg, on="subject_id", how="left")
            self.df["bmi"] = self.df["bmi"].fillna(0).astype("float32")
            self.df["obese"] = (self.df["bmi"]>30).astype("int8")

    # ── STEP 12 : historical + engineered ────────────────────────────────────
    def add_historical(self):
        logger.info("=== STEP 12: Historical + Engineered ===")
        self.df = self.df.sort_values(["subject_id","admittime"]).reset_index(drop=True)

        self.df["prev_admissions"]   = self.df.groupby("subject_id").cumcount().astype("int16")
        self.df["is_first_visit"]    = (self.df["prev_admissions"]==0).astype("int8")
        self.df["prev_readmits"]     = self.df.groupby("subject_id")["readmit_30"].transform(
            lambda x: x.shift(1).expanding().sum()).fillna(0).astype("float32")
        self.df["prev_readmit_rate"] = self.df.groupby("subject_id")["readmit_30"].transform(
            lambda x: x.shift(1).expanding().mean()).fillna(0).astype("float32")
        self.df["days_since_last"]   = (
            self.df.groupby("subject_id")["admittime"].diff().dt.total_seconds()/86400
        ).fillna(999).astype("float32")
        self.df["prev_los_mean"] = self.df.groupby("subject_id")["los_days"].transform(
            lambda x: x.shift(1).expanding().mean()).fillna(0).astype("float32")
        self.df["prev_los_max"]  = self.df.groupby("subject_id")["los_days"].transform(
            lambda x: x.shift(1).expanding().max()).fillna(0).astype("float32")
        if "had_icu" in self.df.columns:
            self.df["prev_icu_rate"] = self.df.groupby("subject_id")["had_icu"].transform(
                lambda x: x.shift(1).expanding().mean()).fillna(0).astype("float32")

        self.df["log_los_days"]  = np.log1p(self.df["los_days"]).astype("float32")
        self.df["log_los_hours"] = np.log1p(self.df["los_hours"]).astype("float32")
        self.df["los_cat"]       = pd.cut(self.df["los_days"],bins=[0,1,3,7,14,30,9999],
                                          labels=[0,1,2,3,4,5],right=False).astype("float32")

        dx_c  = self.df.get("dx_count",  pd.Series(0,index=self.df.index))
        proc_c= self.df.get("proc_count",pd.Series(0,index=self.df.index))
        icu_c = self.df.get("icu_count", pd.Series(0,index=self.df.index))
        tr_c  = self.df.get("transfer_count",pd.Series(0,index=self.df.index))
        rx_c  = self.df.get("rx_count",  pd.Series(0,index=self.df.index))
        lab_c = self.df.get("lab_count", pd.Series(0,index=self.df.index))
        lab_a = self.df.get("lab_abnormal_count",pd.Series(0,index=self.df.index))
        drg_s = self.df.get("drg_severity_max",pd.Series(0,index=self.df.index))
        poe_c = self.df.get("poe_count", pd.Series(0,index=self.df.index))
        icu_hrs=self.df.get("icu_los_sum",pd.Series(0,index=self.df.index))
        micro_c=self.df.get("micro_count",pd.Series(0,index=self.df.index))
        age   = self.df.get("anchor_age",pd.Series(60,index=self.df.index))

        # Charlson score
        self.df["charlson_score"] = (
            self.df.get("cm_chf",0) + self.df.get("cm_diabetes",0) +
            self.df.get("cm_renal_fail",0)*2 + self.df.get("cm_cancer",0)*2 +
            self.df.get("cm_liver",0) + self.df.get("cm_copd",0) +
            self.df.get("cm_mi",0) + self.df.get("cm_stroke",0)*2 +
            self.df.get("cm_dementia",0)*2
        ).astype("float32")

        self.df["complexity_score"]  = (dx_c+proc_c+self.df.get("service_count",0)).astype("float32")
        self.df["severity_score"]    = (icu_c*4+proc_c*2+dx_c+drg_s*2+self.df["charlson_score"]*2).astype("float32")
        self.df["instability_score"] = (tr_c+icu_c*2+lab_a+micro_c).astype("float32")

        # LACE
        L = pd.cut(self.df["los_days"].clip(upper=14),bins=[0,1,3,7,14,9999],
                   labels=[1,2,3,4,5],right=False).astype(float).fillna(1)
        A = (self.df.get("admission_type",0)>=4).astype(int)*3
        C = self.df["charlson_score"].clip(upper=4)
        E = self.df.get("had_ed",0)
        self.df["lace_score"] = (L+A+C+E).astype("float32")

        self.df["age_los"]        = (age*self.df["los_days"]).astype("float32")
        self.df["readmit_age"]    = (self.df["prev_readmit_rate"]*age).astype("float32")
        self.df["severity_age"]   = (self.df["severity_score"]*age/100).astype("float32")
        self.df["icu_lab"]        = (icu_hrs*lab_c).astype("float32")
        self.df["dx_proc"]        = (dx_c*proc_c).astype("float32")
        self.df["los_transfer"]   = (self.df["los_days"]*tr_c).astype("float32")
        self.df["icu_los_pct"]    = (icu_hrs/(self.df["los_hours"]+1)).astype("float32")
        self.df["dx_per_day"]     = (dx_c/(self.df["los_days"]+1)).astype("float32")
        self.df["proc_per_day"]   = (proc_c/(self.df["los_days"]+1)).astype("float32")
        self.df["med_per_day"]    = (rx_c/(self.df["los_days"]+1)).astype("float32")
        self.df["lab_per_day"]    = (lab_c/(self.df["los_days"]+1)).astype("float32")
        self.df["poe_per_day"]    = (poe_c/(self.df["los_days"]+1)).astype("float32")
        self.df["complexity_day"] = (self.df["complexity_score"]/(self.df["los_days"]+1)).astype("float32")
        self.df["high_risk"]      = ((icu_c>0)|(self.df["los_days"]>10)|(age>=80)).astype("int8")
        self.df["very_high_risk"] = ((icu_c>1)|(self.df["los_days"]>20)|(age>=90)).astype("int8")

        for col in ["dx_count","proc_count","rx_count","lab_count","poe_count","micro_count"]:
            if col in self.df.columns:
                self.df[f"log_{col}"] = np.log1p(self.df[col]).astype("float32")

        self.df = self.df.drop(columns=["admittime","died_hospital","next_planned",
                                         "next_adm_type","next_admittime","days_to_next"], errors="ignore")
        self._fill()

    # ── finalize ──────────────────────────────────────────────────────────────
    def finalize(self):
        logger.info("=== FINAL ===")
        target = "readmit_30"
        protected = {"subject_id","hadm_id",target}
        num_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in protected]
        corr = self.df[num_cols].corrwith(self.df[target]).abs().fillna(0).sort_values(ascending=False)
        keep = list(corr.head(500).index)
        self.df = self.df[["subject_id","hadm_id",target]+keep]
        logger.info("Shape: %s | readmit=%.2f%%", self.df.shape, self.df[target].mean()*100)
        self.df.to_csv(FEATURES_CSV, index=False)
        logger.info("Saved → %s", FEATURES_CSV)

    def run(self):
        try:
            self.extract_core()
            self.extract_icu()
            self.extract_vitals()
            self.extract_labs()
            self.extract_pharmacy()
            self.extract_microbiology()
            self.extract_drg()
            self.extract_services_transfers()
            self.extract_codes()
            self.extract_emar()
            self.extract_misc()
            self.add_historical()
            self.finalize()
            return True
        except Exception as e:
            logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
            return False

if __name__ == "__main__":
    MIMICExtractor().run()