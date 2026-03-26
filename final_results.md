# An Adaptive Context-Aware Gating Network for 30-Day Hospital Readmission Prediction — Detailed Final Results

This is the **detailed** and **fully traceable** final-results report for our project:
**30-day hospital readmission prediction** using **structured EHR** + **clinical note embeddings**, with:

1) a **calibrated stacked tree ensemble** (Base Ensemble; implemented as the base framework in this repo),  
2) a **text-guided neural gating model** (ACAGN-Gate), and  
3) a **Concat-MLP baseline** (no gating; direct concatenation), and  
4) a **hybrid ensemble** of the two (ACAGN-Hybrid).

Naming note:
- The paper/system name is **ACAGN** (*An Adaptive Context-Aware Gating Network*).
- Some saved artifacts and code identifiers still use the legacy `trance_*` prefix (for example `models/trance_framework.pkl`, `models/trance_gate.pkl`). Those correspond to the **ACAGN** Base Ensemble and ACAGN-Gate implementations.

All metrics and tables below are taken from artifacts under `results/`:
- `2026-03-24T04:46:17` (Base framework + Gate + Hybrid final run), plus follow-on analyses generated the same day.
- `2026-03-26T03:39:37` (Concat-MLP baseline run; same split + hyperparameters as ACAGN-Gate).

---

## Abstract (What We Built + What We Achieved)

We built an imaging-free readmission-risk modeling pipeline for MIMIC-IV that fuses (i) structured EHR features
with (ii) dense clinical-text embeddings (`ct5_*`). We trained and evaluated three model families:
1) a calibrated stacked ensemble of gradient-boosted decision trees (LightGBM + XGBoost + CatBoost) with a
logistic-regression meta-learner, 2) a neural “text-guided gating” model where the note embedding controls
per-feature gates applied to tabular features, 3) a no-gating Concat-MLP baseline (direct concatenation),
and 4) a probability-level hybrid (50/50 blend) of the tree ensemble and gated model.

On a held-out patient-level test set of **82,241 admissions** (readmission rate **18.43%**), our final calibrated
metrics were:

| Model | AUROC (cal) | AUPRC | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: |
| Base Ensemble | 0.7705 | 0.4708 | 0.0036 | 0.1245 |
| ACAGN-Gate | 0.7683 | 0.4679 | 0.0058 | 0.1249 |
| Concat-MLP (no gating) | 0.7545 | 0.4511 | 0.0033 | 0.1272 |
| **ACAGN-Hybrid** | **0.7738** | **0.4838** | 0.0061 | **0.1239** |

We also performed early-warning evaluation (Day 1–7), temporal drift checks (2008–2022),
fairness subgroup reporting, and interpretability analyses (SHAP for the tree model; gate-weight analysis for ACAGN-Gate).

---

## Contents (What’s Included in This Report)

1. Problem statement and outputs  
2. Data, cohort, and patient-level splits  
3. Pipeline overview (scripts + outputs)  
4. Inputs and feature engineering (structured + text embeddings)  
5. Models trained (Base, Gate, Hybrid)  
6. Metrics definitions (discrimination + calibration + thresholds)  
7. Final test-set results (headline table)  
8. Threshold / operating points (TP/TN/FP/FN)  
9. Additional analyses (early warning, drift, fairness, gating interpretability)  
10. Interpretability (SHAP + what “no extra training” means)  
11. Artifacts and figures saved  
12. Conclusions  

---

## 1. Problem Statement and Outputs

**Goal:** Predict whether an inpatient admission will result in **30-day readmission** (`readmit_30`).

**Outputs produced by the pipeline:**
- Calibrated probability of readmission (0–1)
- Threshold-based operating points for clinical trade-offs (precision/recall)
- Interpretability artifacts:
  - Tree-model SHAP summary (global)
  - Gate-weight interpretability tests (condition-aware gating behavior)

---

## 2. Data, Cohort, and Splits (What Was Evaluated)

### 2.1 Unit of Prediction
- One row per admission identified by `hadm_id`
- Patient identifier: `subject_id`

### 2.2 Patient-level Split
We perform a **patient-level split** so admissions from the same patient do not leak across train/val/test.

Saved split sizes (admissions) from `results/training_report.json`:
- Train: `382,566`
- Validation: `81,251`
- Test: `82,241`

### 2.3 Test-set Class Balance
From `results/fairness_analysis.csv` (overall row):
- Test readmission rate: `0.1843` (18.43%)

---

## 3. End-to-End Pipeline (What We Built and Where It Lives)

This repo is organized as a script pipeline under `src/`:

| Step | Script | Purpose | Main Outputs |
| :--- | :--- | :--- | :--- |
| 1 | `src/01_extract.py` | Extract structured EHR features | `data/ultimate_features.csv` |
| 2 | `src/01b_select_features.py` | Feature ranking + pruning utilities | selection artifacts |
| 3 | `src/02_embed.py` | Generate note embeddings (`ct5_*`) | `data/embeddings.csv` |
| 4 | `src/03_train.py` | Train + calibrate stacked tree ensemble | `models/trance_framework.pkl`, `results/training_report.json` |
| 5 | `src/gated_fusion_model.py` | Train + calibrate ACAGN-Gate | `models/trance_gate.pkl`, `results/gate_training_report.json` |
| 5b | `src/concat_mlp_baseline.py` | Train + calibrate Concat-MLP baseline (no gating) | `models/concat_mlp.pkl`, `results/concat_mlp_training_report.json` |
| 6 | `src/10_gate_interpretability.py` | Gate interpretability testing | `results/gate_interpretability.csv`, `figures/gate_heatmap.png` |
| 7 | `src/11_fairness_calibration.py` | Fairness + calibration across subgroups | `results/fairness_analysis.csv` |
| 8 | `src/12_early_warning.py` | Early warning (Day-limited EHR) | `results/early_warning_results.csv`, `figures/early_warning_curve.png` |
| 9 | `src/13_temporal_drift.py` | Temporal drift (by year group) | `results/temporal_drift_results.csv`, `figures/temporal_drift.png` |
| 10 | `src/14_hybrid_ensemble.py` | Hybrid blend of Base + Gate | `results/hybrid_report.json`, `results/hybrid_predictions.csv` |

For deployment, we also implemented:
- `src/07_api.py` (FastAPI inference service)
- `src/08_predict.py` (local prediction utility)

Note on artifacts:
- Large generated data/model files are typically **not committed to git** (see `.gitignore`), so your workspace may contain
  only the pruned feature file (`data/ultimate_features_pruned.csv`) and the embedding file (`data/embeddings.csv`).

---

## 4. Inputs and Feature Engineering (What the Models See)

### 4.1 Structured EHR Features (Base Framework)

The base framework run reports:
- **675 candidate features** before selection (`results/training_report.json → feature_subset.n_features_full`)

These structured features include categories such as:
- Prior utilization history (previous admissions/readmissions, days since last admission)
- Lab/vitals summary statistics (min/max/mean/last/range, abnormal counts)
- Medications/procedures (counts, “high-risk” indicators)
- Demographics/administrative variables (insurance, language, discharge location, etc.)
- Derived clinical flags (e.g., anemia, hyponatremia)

### 4.2 Clinical Text Embeddings (`ct5_*`)

The embedding file `data/embeddings.csv` contains:
- `ct5_0` … `ct5_511` (**512 embedding dimensions**)
- `ct5_has_note`, `ct5_note_len_chars`, `ct5_note_len_tokens` (**3 metadata features**)

So the embeddings file has **515 `ct5_*` columns**, plus `hadm_id`.

### 4.3 Final Feature Subset Used by the Base Ensemble

The Base Ensemble uses automatic feature subset selection:
- Candidate subsets evaluated on validation AUROC: top-128, 160, 220, 259, and full 675
- Selected: **k = 128** (best validation AUROC)

From the saved model bundle (`models/trance_framework.pkl`):
- Final features used: `128`
- Among them, **58** are `ct5_*` dimensions and **70** are structured features.

### 4.4 Feature Set Used by ACAGN-Gate

ACAGN-Gate is trained on:
- Structured features from `data/ultimate_features_pruned.csv` (**160 structured features**)
- Full text vector from `data/embeddings.csv` (**515 `ct5_*` columns**)

Evidence:
- `results/gate_training_report.json`: `tab_features` length = **160**
- `models/trance_gate.pkl`: `text_dim = 515`, `tabular_dim = 160`

---

## 5. Models (What We Trained)

## 5.1 Base Ensemble (ACAGN Base Ensemble)

**Architecture:**
- Base learners: LightGBM + XGBoost + CatBoost
- Meta-learner: Logistic Regression (stacking)
- Calibration: Isotonic Regression fitted on validation → applied to test

Saved in `models/trance_framework.pkl`:
- `models`: `['lgbm', 'xgb', 'catboost']`
- `meta`: `LogisticRegression`
- `calibrator`: isotonic regressor

**Seeds / ensembling:**
- Seeds: `[42, 2024, 777]` (from `results/training_report.json`)

**Cross-validation summary (training-time sanity check):**
- CV AUROC mean: `0.7746`
- CV AUROC std: `0.0021`

**Ablation (AUROC):** (`results/training_report.json → ablation`)
- Fused (tabular + text): `0.7717`
- Tabular-only: `0.7714`
- Text-only: `0.6280`

<details>
<summary>Base Ensemble best LightGBM hyperparameters (from final run)</summary>

From `results/training_report.json` (`best_params`):

```json
{
  "boosting_type": "gbdt",
  "num_leaves": 186,
  "max_depth": 11,
  "learning_rate": 0.006253352854024568,
  "n_estimators": 4207,
  "min_child_samples": 30,
  "colsample_bytree": 0.8355030127394723,
  "colsample_bynode": 0.7682369329609422,
  "reg_alpha": 3.372809834648114,
  "reg_lambda": 4.073552940345179,
  "min_split_gain": 0.14799140225639096,
  "path_smooth": 0.6038223897701793,
  "max_bin": 362,
  "subsample": 0.8052721759516217,
  "subsample_freq": 2,
  "is_unbalance": true
}
```
</details>

---

## 5.2 ACAGN-Gate (Text-Guided Feature Gating)

**Core idea:** Use the clinical text embedding to produce **patient-specific per-feature gates** over structured inputs.

**Conceptual flow:**
1) Input: text embedding `t` and tabular vector `x`
2) Gate network: `g = sigmoid(MLP(t))` producing one weight per tabular feature
3) Gating: `x_gated = g ⊙ x`
4) Classifier: `p = MLP([t || x_gated])`

**Training strategy:**
- Train multiple seeds and average predictions across seeds
- Calibrate with isotonic regression (val-fitted → test-applied)

Saved gate metrics in `results/gate_training_report.json`:
- AUROC raw: `0.7684`
- AUROC calibrated: `0.7683`
- AUPRC: `0.4679`
- Brier: `0.1249`
- ECE before calibration: `0.0032`
- ECE after calibration: `0.0058`

---

## 5.3 ACAGN-Hybrid (Base + Gate)

**Definition:** a probability-level ensemble:

`p_hybrid = 0.5 * p_base_calibrated + 0.5 * p_gate_calibrated`

Saved in:
- `results/hybrid_report.json`
- `results/hybrid_predictions.csv`

---

## 5.4 Concat-MLP Baseline (No Gating; Direct Concatenation)

**Goal:** isolate whether ACAGN-Gate performance gains come from the *gating mechanism itself*, rather than simply
combining text + structured features.

**Architecture:** identical to ACAGN-Gate’s classifier head, but removes the gate network entirely:
- Fusion: `concat(text_embedding, tabular_features)`
- Classifier head: same MLP layout and dropout as ACAGN-Gate
- Training: same patient-level split strategy, seeds, optimizer, epochs, scheduler, early stopping, and isotonic calibration

**Artifacts:**
- `src/concat_mlp_baseline.py`
- `models/concat_mlp.pkl`
- `results/concat_mlp_training_report.json`

**Final test-set metrics (calibrated):**
- AUROC (cal): `0.7545`
- AUPRC: `0.4511`
- ECE: `0.0033`
- Brier: `0.1272`

---

## 6. Metrics (What We Focused On and Why)

We reported both **discrimination** and **calibration**, because clinical risk stratification needs reliable probabilities.

### Discrimination
- **AUROC**: ranking quality across thresholds
- **AUPRC**: precision/recall quality for the positive class under imbalance

### Calibration
- **Brier score**: mean squared error of probabilities (lower is better)
- **ECE**: expected calibration error over 10 probability bins (lower is better)

### Threshold-based performance (Base Ensemble)
We also report operating points to support clinical trade-offs:
- MCC-selected threshold
- F1-optimal threshold
- Recall≥0.80 threshold
- Youden-J threshold

---

## 7. Final Test-Set Results (Headline Table)

All results below are on the held-out **patient-level** test set (`n = 82,241` admissions).

From `results/training_report.json`, `results/gate_training_report.json`, and `results/hybrid_report.json`:

| Model | AUROC (raw) | AUROC (cal) | AUPRC | ECE | Brier | Log loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base Ensemble | 0.7708 | 0.7705 | 0.4708 | 0.0036 | 0.1245 | 0.3984 |
| ACAGN-Gate | 0.7684 | 0.7683 | 0.4679 | 0.0058 | 0.1249 | — |
| **ACAGN-Hybrid** | — | **0.7738** | **0.4838** | 0.0061 | **0.1239** | — |

Key observations:
- Base Ensemble is the strongest single model in calibration (lowest ECE/Brier among single models).
- ACAGN-Gate is close in AUROC/AUPRC and supports gating-based interpretability.
- ACAGN-Hybrid provides the best AUROC and AUPRC in the saved run.

---

## 8. Base Ensemble Threshold / Operating Point Details

This section is directly from `results/training_report.json → operating_points`.

| Strategy | Threshold | Acc | Prec | Recall | Spec | F1 | MCC | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| MCC (selected) | 0.295 | 0.8075 | 0.4751 | 0.4227 | 0.8944 | 0.4474 | 0.3321 | 6,409 | 59,999 | 7,081 | 8,752 |
| F1-optimal | 0.235 | 0.7679 | 0.4057 | 0.5572 | 0.8155 | 0.4695 | 0.3324 | 8,448 | 54,704 | 12,376 | 6,713 |
| Recall≥0.80 | 0.130 | 0.6155 | 0.2981 | 0.8019 | 0.5734 | 0.4347 | 0.2910 | 12,157 | 38,462 | 28,618 | 3,004 |
| Youden-J | 0.1897 | 0.7190 | 0.3588 | 0.6661 | 0.7309 | 0.4664 | 0.3245 | 10,099 | 49,030 | 18,050 | 5,062 |

---

## 9. Additional Analyses (Everything We Evaluated Beyond the Headline Metrics)

## 9.1 Early Warning (Day-limited EHR)

Script: `src/12_early_warning.py`

This retrains a LightGBM model using only features available within the first *N* days of the admission.

From `results/early_warning_results.csv`:

| Cutoff | AUROC |
| :--- | :---: |
| Full | 0.7707 |
| Day 1 | 0.7708 |
| Day 2 | 0.7710 |
| Day 3 | 0.7706 |
| Day 5 | 0.7707 |
| Day 7 | 0.7709 |

Note:
- The file stores `n_train` / `n_test` as the **count of positive readmissions** in each split (not total rows).

## 9.2 Temporal Drift (2008–2022)

Script: `src/13_temporal_drift.py`

From `results/temporal_drift_results.csv`:

| Year group | Admissions (n) | AUROC (Base Ensemble) | AUROC (ACAGN-Gate) |
| :--- | :---: | :---: | :---: |
| 2008–2010 | 34,549 | 0.7650 | 0.7641 |
| 2011–2013 | 17,556 | 0.7674 | 0.7677 |
| 2014–2016 | 13,478 | 0.7717 | 0.7699 |
| 2017–2019 | 10,287 | 0.7772 | 0.7726 |
| 2020–2022 | 6,371 | 0.7772 | 0.7591 |

## 9.3 Fairness / Calibration by Subgroup

Script: `src/11_fairness_calibration.py`

From `results/fairness_analysis.csv` (highlights):

**Overall**
- Base Ensemble: AUROC `0.7705`, ECE `0.0036`, Brier `0.1245`
- ACAGN-Gate: AUROC `0.7683`, ECE `0.0058`, Brier `0.1249`

**Gender (AUROC)**
- Base: Female `0.7747`, Male `0.7654`
- Gate: Female `0.7739`, Male `0.7616`

**Age (AUROC extremes)**
- Best (both): `<40` (Base `0.8417`, Gate `0.8418`)
- Most challenging: `75–84` (Base `0.6951`, Gate `0.6828`)

Important note:
- The saved output contains race quartiles `Q1` and `Q2` (not `Q3/Q4`) for both models.

## 9.4 Gating Interpretability (Condition-aware Gate Behavior)

Script: `src/10_gate_interpretability.py`

From `results/gate_interpretability.csv` (Mann–Whitney U tests + Bonferroni correction):
- Chronic anemia: gate weight on `lab_hemoglobin_min` is lower when anemia is mentioned  
  (mean difference `-0.0508`, p `1.68e-90`)
- Chronic kidney disease: multiple creatinine/BUN-related gate weights shift when CKD is mentioned (many p-values reported as ~`0.0`)

---

## 10. Interpretability: SHAP and “Do We Need Extra Training?”

### 10.1 Base Ensemble SHAP (Already Produced)

No extra training is required.
- SHAP is computed for the trained **LightGBM** component using `shap.TreeExplainer`.
- Output already present: `figures/shap_summary.png`.

### 10.2 ACAGN-Gate SHAP (Optional)

Gate SHAP also does **not** require training *if* the saved gate bundle contains model weights/state.
If an older bundle does not contain saved weights, you must train once with an updated pipeline to persist them,
then SHAP-only runs can be executed without training.

Practical commands:
- Train (once, to save weights): `python3 src/gated_fusion_model.py`
- SHAP-only (no training): `python3 src/gated_fusion_model.py --shap-only`

### 10.3 ACAGN-Hybrid “SHAP”

Hybrid is a probability blend, not a feature-level model. The explainability we provide is an **importance aggregation**
from Base + Gate SHAP importances, once both are available.

Practical command (after the SHAP importance CSVs exist):
- `python3 src/14_hybrid_ensemble.py`

---

## 11. Artifacts and Figures (Exactly What Was Saved)

### 11.0 Metric comparison diagrams (generated for the paper)

To make access easier, we generate a dedicated metrics bundle:
- Generator: `python3 src/15_generate_metric_diagrams.py`
- Index (categorized list of outputs): `results/metrics/INDEX.json`

### 11.1 Primary reports (present)
- `results/training_report.json`
- `results/gate_training_report.json`
- `results/hybrid_report.json`
- `results/hybrid_predictions.csv`
- `results/test_predictions.csv`

### 11.2 Analysis outputs (present)
- `results/fairness_analysis.csv`
- `results/early_warning_results.csv`
- `results/temporal_drift_results.csv`
- `results/gate_interpretability.csv`

### 11.3 Figures (present)
- `figures/roc_pr_curve.png`
- `figures/calibration_curve.png`
- `figures/threshold_analysis.png`
- `figures/shap_summary.png`
- `figures/reliability_diagram.png`
- `figures/early_warning_curve.png`
- `figures/temporal_drift.png`
- `figures/gate_heatmap.png`

### 11.3.1 Paper-ready metric diagrams (generated)
- `figures/metrics/roc_curves.png`
- `figures/metrics/pr_curves.png`
- `figures/metrics/discrimination_metrics.png`
- `figures/metrics/calibration_metrics.png`
- `figures/metrics/shared_threshold_operating_metrics.png`
- `figures/metrics/confusion_matrices_test_opt_mcc.png`
- `figures/metrics/confusion_matrices_test_opt_recall80.png`
- `figures/metrics/gate_mannwhitney_top.png`

### 11.3.2 Paper-ready metric tables (generated)
- `results/metrics/threshold_free_metrics.csv` (AUROC, AUPRC, ECE, Brier, Log loss)
- `results/metrics/shared_threshold_operating_metrics.csv` (Accuracy, F1, MCC, Precision/Recall/Specificity, TP/TN/FP/FN, Youden-J at shared threshold)
- `results/metrics/operating_points.csv` (test-optimized thresholds by strategy; labeled `test_opt_*`)
- `results/metrics/delong_auroc_pvalues.csv` (DeLong p-values for AUROC comparison)
- `results/metrics/gate_mannwhitney_all.csv` (Mann–Whitney U p-values for gate analysis)

### 11.4 Optional SHAP artifacts (supported; generate by running SHAP pipelines)
- `results/base_shap_importance.csv`
- `results/gate_shap_importance.csv`
- `results/hybrid_shap_importance.csv`
- `figures/gate_shap_summary.png`
- `figures/hybrid_shap_importance.png`

---

## 12. Conclusion (Final Takeaways)

On a large held-out patient-level test set, the Base Ensemble is strong and tightly calibrated, ACAGN-Gate matches it
closely while adding interpretable feature gating, and the simple ACAGN-Hybrid blend delivers the best AUROC and AUPRC
in this saved evaluation. The early-warning, temporal drift, and subgroup reports suggest stable performance across time
and clinically relevant cohorts, with expected degradation in more complex elderly readmission groups.
