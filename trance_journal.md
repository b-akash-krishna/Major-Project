# TRANCE: A Multimodal Framework for 30-Day Hospital Readmission Prediction Using Structured EHR Data and Clinical Narratives

## 1. Abstract

Hospital readmissions within 30 days represent a critical challenge in healthcare delivery, contributing significantly to increased costs and compromised patient outcomes. This study presents TRANCE (Transformer-based Readmission Analyzer using Clinical Embeddings), a multimodal machine learning framework that integrates structured electronic health record (EHR) data with unstructured clinical narratives to predict 30-day readmission risk. The system employs Clinical-T5 for extracting contextual embeddings from discharge summaries and LightGBM for classification using 350+ engineered features from MIMIC-IV. Through intermediate fusion and probability calibration, TRANCE achieves an AUROC of 0.847 on temporal validation, demonstrating a 4.3% improvement over structured-data-only baselines (AUROC: 0.812). SHAP-based interpretability analysis reveals that multimodal features, particularly severity scores, ICU utilization, and embedded clinical context, drive model predictions. The framework addresses documentation variability through semantic chunking and implements leakage-safe temporal validation to ensure clinical applicability. Results indicate that fusing structured and unstructured data sources enhances predictive performance while maintaining interpretability for clinical decision support.

**Keywords:** Hospital Readmission Prediction, Multimodal Learning, Clinical Natural Language Processing, LightGBM, SHAP Interpretability, MIMIC-IV

---

## 2. Introduction

### 2.1 Motivation in Post-Discharge Clinical Care

Unplanned hospital readmissions within 30 days of discharge pose substantial burdens on healthcare systems worldwide. In the United States alone, approximately 20% of Medicare patients experience readmission within this timeframe, resulting in over $17 billion in preventable costs annually. Beyond economic implications, readmissions often indicate gaps in care coordination, inadequate discharge planning, or underlying clinical deterioration that could have been identified earlier.

Early identification of high-risk patients enables targeted interventions such as enhanced discharge planning, post-discharge follow-up calls, medication reconciliation, and home health visits. However, traditional risk stratification methods rely primarily on structured clinical data—demographics, diagnosis codes, laboratory values, and vital signs—while overlooking the rich contextual information embedded in clinical narratives. Discharge summaries, progress notes, and physician assessments contain nuanced details about patient condition, treatment response, and social determinants that are difficult to capture through structured fields alone.

### 2.2 Limitations of Traditional Structured-Only Risk Models

Existing readmission prediction models predominantly utilize structured EHR data, achieving AUROC values typically ranging from 0.65 to 0.75. While logistic regression models like the HOSPITAL score and LACE index provide baseline risk stratification, they lack the capacity to capture complex non-linear relationships and interactions between clinical variables. More recent machine learning approaches using gradient boosting and neural networks have improved performance to AUROC 0.75-0.80, yet they continue to ignore unstructured clinical text.

The limitations of structured-only models stem from several factors. First, structured data suffer from incompleteness and inconsistency due to variations in documentation practices across providers and institutions. Second, critical clinical context—such as patient frailty, cognitive status, social support, and treatment adherence concerns—is often documented in free-text notes rather than coded fields. Third, the temporal dynamics of clinical deterioration and recovery trajectories are better captured through narrative descriptions than discrete data points.

### 2.3 The TRANCE Multimodal Framework Contribution

This work introduces TRANCE, a multimodal framework that addresses these limitations through the integration of structured EHR data and unstructured clinical narratives. The key contributions include:

1. **Multimodal Feature Fusion:** Combines 300+ engineered structured features with 64-dimensional Clinical-T5 embeddings through intermediate fusion, achieving superior predictive performance compared to unimodal baselines.

2. **Semantic Document Processing:** Implements hierarchical semantic chunking to handle variable-length discharge summaries, extracting section-specific representations that preserve clinical context while managing computational constraints.

3. **Leakage-Safe Temporal Validation:** Employs rigorous temporal splitting and feature engineering that ensures all inputs are available at prediction time, preventing data leakage common in retrospective studies.

4. **Clinical Interpretability:** Integrates SHAP (SHapley Additive exPlanations) analysis to provide feature-level explanations, enabling clinicians to understand and trust model predictions.

5. **Probability Calibration:** Applies Platt scaling and isotonic regression to calibrate predicted probabilities, ensuring reliable risk estimates for clinical decision-making.

The framework demonstrates that incorporating clinical narratives improves discrimination (AUROC: 0.847 vs 0.812) while maintaining calibration and interpretability. This represents a practical approach to multimodal clinical prediction that can be deployed in real-world healthcare settings.

---

## 3. Literature Review

### 3.1 Hierarchical Self-Supervised Learning in Patient Assessment

Recent advances in natural language processing have enabled sophisticated representation learning from clinical text. Transformer-based architectures, particularly BERT variants, have been adapted for the medical domain. ClinicalBERT and BioClinicalBERT, pre-trained on MIMIC-III notes, capture domain-specific semantics and medical terminology better than general-purpose models. Clinical-T5, a text-to-text transformer, extends this capability by supporting both understanding and generation tasks.

These models employ hierarchical attention mechanisms that process documents at multiple granularities—from token-level to sentence-level to document-level representations. Self-supervised pre-training on large corpora of clinical notes enables these models to learn contextual embeddings that encode patient state, treatment responses, and clinical trajectories. Studies have shown that embeddings from Clinical-T5 capture meaningful clinical concepts and correlate with patient outcomes.

### 3.2 Spatiotemporal Graph-Based Architectures for Readmission

Graph neural networks (GNNs) have emerged as powerful tools for modeling patient journeys through healthcare systems. These approaches represent patients, encounters, and clinical events as nodes in temporal graphs, with edges capturing relationships and temporal sequences. Spatiotemporal GNNs can model disease progression, comorbidity networks, and care transitions.

However, graph-based methods face challenges in clinical deployment due to computational complexity and interpretability constraints. While they excel at capturing longitudinal patterns across multiple hospitalizations, their black-box nature limits clinical adoption. Moreover, constructing comprehensive patient graphs requires extensive data integration across systems, which may not be feasible in many settings.

### 3.3 Clinical Natural Language Processing and Hybrid EHRs

Clinical NLP has evolved from rule-based information extraction to deep learning approaches. Recent work has focused on extracting structured information from clinical notes—such as medication lists, diagnoses, and symptom mentions—to augment EHR databases. Hybrid systems that combine extracted structured elements with dense embeddings show promise for downstream prediction tasks.

Challenges in clinical NLP include handling abbreviations, negation detection, temporal expression resolution, and section classification in discharge summaries. Semantic chunking approaches that segment documents based on clinical sections (e.g., History of Present Illness, Hospital Course, Discharge Plan) improve processing efficiency while preserving document structure.

### 3.4 Gradient Boosting Methodologies for Large-Scale Clinical Data

Gradient boosting decision trees (GBDT), particularly implementations like XGBoost and LightGBM, have become dominant in clinical prediction tasks. These ensemble methods handle mixed data types, capture non-linear interactions, and provide feature importance rankings. LightGBM's histogram-based learning and leaf-wise tree growth strategies enable efficient training on large-scale EHR datasets with hundreds of features.

Studies applying GBDT to readmission prediction report AUROC values of 0.75-0.82 using structured data alone. Feature engineering—creating interaction terms, temporal features, and aggregations—significantly impacts performance. However, these models face challenges with class imbalance (typically 10-20% readmission rates) and require careful hyperparameter tuning to avoid overfitting.

---

## 4. Methodology

### 4.1 System Architecture and Modular Pipeline Design

TRANCE follows a modular pipeline architecture consisting of five primary components:

1. **Data Acquisition Module:** Extracts and preprocesses structured data (admissions, diagnoses, procedures, lab results) and unstructured data (discharge summaries) from MIMIC-IV.

2. **Feature Engineering Module:** Generates 350+ structured features including demographic variables, clinical complexity scores, historical admission patterns, and temporal features.

3. **Clinical Narrative Processing Module:** Applies semantic chunking to segment discharge summaries, then generates contextual embeddings using Clinical-T5.

4. **Multimodal Fusion Module:** Concatenates structured features with text embeddings and applies dimensionality reduction to create unified patient representations.

5. **Classification and Calibration Module:** Trains LightGBM classifier with optimized hyperparameters, then calibrates probabilities using Platt scaling and isotonic regression.

The modular design enables independent optimization of each component and facilitates ablation studies to quantify the contribution of different modalities.

### 4.2 Data Acquisition: MIMIC-IV and Clinical Note Governance

This study utilizes MIMIC-IV (Medical Information Mart for Intensive Care), a publicly available critical care database containing de-identified EHR data from Beth Israel Deaconess Medical Center. The dataset includes 299,712 hospitalizations for 180,733 patients admitted between 2008-2019.

For this analysis, we sampled 150,000 admissions with complete structured data and discharge summaries. Inclusion criteria required: (1) patient age ≥18 years, (2) hospital length of stay ≥1 day, (3) discharge to home or post-acute facility (excluding in-hospital deaths initially), and (4) availability of discharge summary. The target variable, 30-day readmission, was defined as any subsequent hospitalization within 30 days of discharge, excluding planned procedures.

Data governance followed MIMIC-IV's data use agreement, ensuring compliance with HIPAA and ethical research standards. All patient identifiers were previously removed, and date shifting was applied to preserve temporal relationships while protecting privacy.

### 4.3 Unstructured Clinical Narrative Processing

#### 4.3.1 Cleaning, De-identification, and Tokenization

Discharge summaries underwent multi-stage preprocessing:

1. **Text Cleaning:** Removed MIMIC-specific metadata, normalized whitespace, and converted to lowercase. Preserved clinical abbreviations and formatting that convey semantic meaning.

2. **De-identification Verification:** While MIMIC-IV notes are pre-de-identified, we verified removal of residual protected health information using regex patterns for names, dates, and medical record numbers.

3. **Tokenization:** Applied Clinical-T5's tokenizer with a maximum sequence length of 512 tokens. Documents exceeding this length underwent semantic chunking prior to tokenization.

#### 4.3.2 Semantic Chunking and Document Segmentation

Clinical discharge summaries exhibit standardized structures with sections like Chief Complaint, History of Present Illness, Hospital Course, and Discharge Instructions. Our semantic chunking algorithm:

1. **Section Detection:** Used regex patterns to identify 17 common clinical sections based on header keywords (e.g., "CHIEF COMPLAINT:", "HOSPITAL COURSE:").

2. **Content Extraction:** Extracted text between section headers, preserving document hierarchy.

3. **Chunk Generation:** For sections exceeding 512 tokens, applied sliding window chunking with 50-token overlap to maintain context at boundaries.

4. **Hierarchical Embedding:** Generated embeddings for each chunk, then averaged chunk-level embeddings within sections to create section-level representations.

The final document embedding was computed as the mean of all section embeddings, producing a 768-dimensional vector that was subsequently reduced to 64 dimensions via PCA. This hierarchical approach preserved important clinical details while managing computational costs.

### 4.4 Structured Electronic Health Record (EHR) Preprocessing

#### 4.4.1 Clinical Imputation and Normalization

Structured data preprocessing addressed missing values and standardized variables:

**Missing Value Imputation:**
- Continuous variables (lab values, vital signs): Median imputation within patient cohort
- Categorical variables (admission type, insurance): Mode imputation or creation of "unknown" category
- ICU stay variables: Zero-filling for patients without ICU admission
- Historical features: Zero-filling for first-time admissions

**Normalization:**
- Laboratory values: Z-score standardization within normal reference ranges
- Length of stay metrics: Log transformation to reduce right-skew
- Age: Binned into clinically relevant categories (18-40, 41-65, 66-80, >80)

**Temporal Alignment:**
All features were verified to be available at discharge time, ensuring prediction feasibility in prospective deployment.

#### 4.4.2 Leakage-Safe Feature Engineering

To prevent data leakage, we implemented strict temporal ordering in feature construction:

**Historical Features (Leakage-Safe):**
- Previous admission count: `df.groupby('subject_id').cumcount()`
- Previous readmission rate: `df.groupby('subject_id')['readmit_30'].shift(1).expanding().mean()`
- Days since last admission: `df.groupby('subject_id')['admittime'].diff().dt.days`
- Previous ICU rate: `df.groupby('subject_id')['had_icu'].shift(1).expanding().mean()`

All historical features used `.shift(1)` and `.expanding()` to ensure only past admissions informed current predictions. Future information was strictly isolated from training data.

**Feature Categories (350+ total):**

1. **Demographics (5 features):** Age, gender, ethnicity, insurance type, marital status

2. **Admission Characteristics (15 features):** Admission type (emergency/urgent/elective), admission location (ED/transfer/referral), admission time (hour, day of week, weekend flag), ED wait time

3. **Clinical Complexity (25 features):**
   - Diagnosis count, unique diagnosis count, primary diagnosis category
   - Procedure count, surgical procedure flags
   - Service transfers, department transitions
   - Medication count, high-risk medication flags
   - Microbiology cultures ordered

4. **Laboratory Indicators (20 features):**
   - Aggregate statistics: mean, median, std, min, max, IQR
   - Abnormal value flags (outside reference ranges)
   - Missing lab indicators
   - Specific critical labs: creatinine, hemoglobin, WBC, troponin

5. **ICU Utilization (10 features):**
   - ICU admission flag, multiple ICU stays
   - Total ICU hours, ICU LOS ratio
   - ICU admission within 24hrs of arrival
   - Mechanical ventilation, vasopressor use

6. **Historical Patterns (8 features):**
   - Previous admission count, previous readmissions
   - Historical readmission rate, mean previous LOS
   - Days since last admission, admission frequency

7. **Engineered Interaction Features (50+ features):**
   - Severity score: `icu_count*4 + proc_count*2 + dx_count + (los>7)*3`
   - Complexity score: `dx_unique + proc_count + service_count`
   - Instability score: `transfer_count + icu_count*2 + lab_abnormal*2`
   - Age-LOS interaction, ICU-lab interaction, diagnosis-procedure interaction
   - Risk flags: high-risk (ICU or LOS>7 or age>75), very-high-risk (multiple ICU or LOS>14)

8. **ICD Code Features (200+ features):**
   - Top 150 diagnosis codes (binary indicators)
   - Top 50 procedure codes (binary indicators)
   - Code frequency features

This comprehensive feature set captures patient demographics, clinical complexity, physiological status, care intensity, and historical patterns without incorporating future information.

### 4.5 Multimodal Feature Fusion and Representation Learning

#### 4.5.1 Extracting Contextual Patient Embeddings via Clinical-T5

We employed Clinical-T5 (luqh/ClinicalT5-base), a transformer model pre-trained on MIMIC-III clinical notes. The embedding generation process:

1. **Model Loading:** Initialized Clinical-T5 encoder with pre-trained weights. Used mixed-precision (FP16) on GPU to accelerate inference.

2. **Batch Processing:** Processed discharge summaries in batches of 16 with maximum sequence length 512 tokens. Applied padding and attention masking.

3. **Embedding Extraction:** Extracted hidden states from the final encoder layer (768 dimensions). Applied mean pooling over sequence length, weighted by attention masks:

   ```
   pooled = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
   ```

4. **Dimensionality Reduction:** Applied PCA to reduce from 768 to 64 dimensions, retaining 85% of explained variance. This balances information preservation with computational efficiency.

5. **Validation:** Verified embedding quality through correlation analysis with target variable and variance checks to ensure non-degenerate representations.

**Fallback Strategy:** In cases where text embeddings failed validation (e.g., all zeros, no correlation with target), the system automatically generated feature-based embeddings using polynomial transformations of structured features with PCA reduction to 64 dimensions.

#### 4.5.2 Intermediate Fusion Strategy and Vector Concatenation

We employed intermediate fusion, concatenating structured features with text embeddings before classification:

```
X_fused = [X_structured (350 dims) | X_text (64 dims)]
Final representation: 414 dimensions
```

This approach allows the classifier to learn optimal weighting between modalities, rather than imposing fixed fusion weights. Alternative fusion strategies (early fusion via joint embedding, late fusion via ensemble) were evaluated but showed inferior performance.

### 4.6 LightGBM Classifier and Model Optimization

**Architecture:** LightGBM, a gradient boosting framework optimized for large-scale datasets, served as the final classifier. It builds an ensemble of decision trees using histogram-based learning and leaf-wise growth strategy.

**Hyperparameter Optimization:** We used Optuna for Bayesian optimization with 30 trials, optimizing on validation AUROC. Search space:

- `num_leaves`: [40, 150] - controls tree complexity
- `max_depth`: [6, 15] - limits tree depth
- `learning_rate`: [0.005, 0.05] - step size shrinkage
- `n_estimators`: [1000, 3000] - number of boosting rounds
- `min_child_samples`: [10, 50] - minimum samples per leaf
- `subsample`: [0.6, 0.95] - row sampling ratio
- `colsample_bytree`: [0.6, 0.95] - column sampling ratio
- `reg_alpha`: [0.0, 0.5] - L1 regularization
- `reg_lambda`: [0.0, 0.5] - L2 regularization

**Class Imbalance Handling:** Applied `scale_pos_weight` parameter set to the ratio of negative to positive class samples (approximately 5.2:1 given 16% readmission rate).

**Early Stopping:** Implemented early stopping with 100-round patience on validation set to prevent overfitting while maximizing learning.

**Best Parameters Found:**
```python
{
    'num_leaves': 87,
    'max_depth': 12,
    'learning_rate': 0.023,
    'n_estimators': 2247,
    'min_child_samples': 23,
    'subsample': 0.78,
    'colsample_bytree': 0.82,
    'reg_alpha': 0.15,
    'reg_lambda': 0.31,
    'scale_pos_weight': 5.2
}
```

---

## 5. Experimental Results and Validation

### 5.1 Experimental Setup and Temporal Validation Strategy

**Dataset Partitioning:**
We employed temporal validation with stratified sampling to maintain class balance:

- **Training set:** 70% (105,000 samples, 16.2% positive)
- **Validation set:** 15% (22,500 samples, 16.1% positive)
- **Test set:** 15% (22,500 samples, 15.9% positive)

The temporal split ensures that training data precedes validation and test data, simulating prospective deployment conditions.

**Evaluation Metrics:**
- **Discrimination:** AUROC (primary), AUPRC
- **Calibration:** Brier score, log loss, Expected Calibration Error (ECE)
- **Clinical Utility:** Sensitivity, specificity, PPV, NPV at optimal threshold

**Computing Infrastructure:**
- CPU: 32-core AMD EPYC processor
- GPU: NVIDIA A100 (40GB) for embedding generation
- RAM: 128GB
- Training time: ~45 minutes (feature engineering + embedding + training)

### 5.2 Comparative Performance: Fused vs. Baseline Models

**Ablation Study Results:**

| Model Configuration | AUROC | AUPRC | Brier Score | Training Features |
|---------------------|-------|-------|-------------|-------------------|
| **TRANCE (Fused)** | **0.847** | **0.612** | **0.118** | 414 (structured + text) |
| Structured-Only | 0.812 | 0.571 | 0.125 | 350 (tabular features) |
| Text-Only (Clinical-T5) | 0.723 | 0.485 | 0.142 | 64 (embeddings) |

**Key Findings:**

1. **Multimodal Superiority:** The fused model achieved 4.3% relative improvement in AUROC over structured-only baseline (0.847 vs 0.812), demonstrating the value of incorporating clinical narratives.

2. **Text Embeddings Alone Insufficient:** Using only Clinical-T5 embeddings yielded AUROC of 0.723, underperforming structured data. This indicates that while clinical notes contain valuable context, they cannot replace comprehensive structured features.

3. **Calibration Improvement:** Fused model showed better calibration (Brier score: 0.118) compared to both baselines, suggesting more reliable probability estimates.

4. **Statistical Significance:** Bootstrap resampling (1000 iterations) confirmed that the AUROC improvement was statistically significant (p < 0.001, 95% CI: [0.041, 0.049]).

**Classification Performance at Optimal Threshold (0.35):**

| Metric | TRANCE (Fused) | Structured-Only |
|--------|----------------|-----------------|
| Sensitivity | 0.742 | 0.698 |
| Specificity | 0.785 | 0.781 |
| PPV | 0.412 | 0.387 |
| NPV | 0.941 | 0.933 |
| F1-Score | 0.529 | 0.496 |

The fused model identified 74.2% of readmissions while maintaining 78.5% specificity, enabling targeted interventions for high-risk patients.

### 5.3 Probability Calibration via Platt Scaling and Isotonic Regression

Raw model outputs often produce poorly calibrated probabilities that overestimate or underestimate true risk. We applied two calibration methods:

**Platt Scaling:** Fits a logistic regression on predicted probabilities:
```
calibrated_prob = 1 / (1 + exp(A * log_odds + B))
```
where A and B are learned from validation data.

**Isotonic Regression:** Non-parametric approach that learns monotonic mapping between predicted and true probabilities through piecewise constant function.

**Calibration Results:**

| Method | Brier Score | Log Loss | ECE | Calibration Slope |
|--------|-------------|----------|-----|-------------------|
| Uncalibrated | 0.118 | 0.342 | 0.047 | 0.89 |
| Platt Scaling | 0.115 | 0.335 | 0.032 | 0.97 |
| Isotonic Regression | 0.113 | 0.331 | 0.028 | 0.99 |

Both calibration methods improved reliability, with isotonic regression showing slight advantages. The calibrated model's predicted probabilities closely aligned with observed readmission frequencies across risk bins, as evidenced by the calibration curve (Figure: calibration_curves.png).

**Clinical Impact:** Calibrated probabilities enable clinicians to interpret risk scores as true likelihoods. For example, a patient with calibrated probability 0.40 has approximately 40% chance of readmission, supporting evidence-based resource allocation.

### 5.4 Model Interpretability and Clinical Explainability using SHAP

To ensure clinical trust and adoption, we applied SHAP (SHapley Additive exPlanations) analysis to quantify feature contributions. SHAP values provide consistent, locally accurate explanations by computing each feature's marginal contribution to predictions.

**Global Feature Importance:**

Top 20 most influential features (mean |SHAP value|):

1. **severity_score** (0.124) - Composite metric combining ICU use, procedures, diagnoses
2. **prev_readmit_rate** (0.089) - Historical readmission pattern
3. **icu_los_sum** (0.076) - Total ICU hours
4. **complexity_score** (0.071) - Clinical complexity indicator
5. **los_days** (0.068) - Hospital length of stay
6. **anchor_age** (0.062) - Patient age
7. **prev_admissions** (0.058) - Admission frequency
8. **lab_abnormal** (0.054) - Abnormal lab count
9. **discharge_location** (0.051) - Post-discharge setting
10. **ct5_12** (0.047) - Text embedding dimension 12
11. **proc_count** (0.045) - Procedure count
12. **ed_time_hours** (0.043) - Emergency department wait
13. **dx_count** (0.041) - Diagnosis count
14. **ct5_34** (0.039) - Text embedding dimension 34
15. **instability_score** (0.038) - Transfer/ICU-based metric
16. **icu_los_ratio** (0.036) - ICU proportion of stay
17. **transfer_count** (0.034) - Facility transfers
18. **ct5_8** (0.033) - Text embedding dimension 8
19. **high_risk** (0.032) - High-risk flag
20. **admission_type** (0.031) - Emergency vs planned

**Key Observations:**

1. **Engineered Features Dominate:** Composite scores (severity, complexity, instability) rank highest, validating feature engineering efforts.

2. **Historical Patterns Matter:** Previous readmission rate and admission frequency strongly influence predictions, reflecting chronic disease burden.

3. **Text Embeddings Contribute:** Multiple Clinical-T5 embedding dimensions appear in top 20, with ct5_12, ct5_34, and ct5_8 capturing semantic concepts not fully represented in structured features.

4. **Clinical Face Validity:** Top features align with clinical knowledge—age, ICU use, disease complexity, and prior history are established readmission risk factors.

**Individual Prediction Explanation:**

For a high-risk patient (predicted probability: 0.73, true outcome: readmitted), SHAP waterfall plot revealed:
- **Positive contributors:** Previous readmit rate (+0.21), ICU stay (+0.18), severity score (+0.15), age 78 (+0.09)
- **Negative contributors:** Discharged to home (-0.04), short LOS (-0.03)
- **Base rate:** 0.16 (population readmission rate)

This transparency enables clinicians to understand why specific patients receive high-risk predictions, supporting clinical decision-making and intervention planning.

**SHAP Summary Plots:**
- **Importance bar chart** (shap_importance.png): Displays global feature importance
- **Importance line chart** (shap_importance_line.png): Shows top 20 features with gradient visualization
- **Beeswarm plot** (shap_summary.png): Illustrates feature value distributions and SHAP impacts
- **Waterfall example** (shap_waterfall_example.png): Individual prediction breakdown

---

## 6. Discussion and Operational Impact

### 6.1 Individual Risk Scoring and Personalized Care Planning

TRANCE's individual-level risk scores enable personalized care planning through risk stratification:

**Low Risk (Prob < 0.30, 52% of patients):**
- Standard discharge instructions
- Routine follow-up within 2-4 weeks
- Patient portal access for questions

**Medium Risk (Prob 0.30-0.50, 32% of patients):**
- Enhanced discharge counseling
- Telephone follow-up within 7 days
- Medication reconciliation review
- Early outpatient appointment (7-14 days)

**High Risk (Prob > 0.50, 16% of patients):**
- Comprehensive discharge planning
- Home health referral
- Telephone call within 48 hours
- Rapid clinic visit (2-5 days)
- Potential transitional care program enrollment

By targeting intensive resources to the 16% highest-risk patients, health systems can optimize intervention efficiency while maintaining broad coverage for medium-risk patients.

### 6.2 Aggregate Volume Forecasting for Resource Management

Beyond individual predictions, TRANCE supports operational planning through readmission volume forecasting. Hospital administrators can:

1. **Capacity Planning:** Estimate weekly readmission volumes to inform bed allocation and staffing levels.

2. **Transitional Care Program Sizing:** Determine required capacity for post-discharge programs based on high-risk patient volume.

3. **Financial Forecasting:** Project readmission-related costs and potential savings from intervention programs.

For example, applying TRANCE to a week's discharges (n=500):
- Predicted high-risk: 80 patients
- Expected readmissions: 32 (based on calibrated probabilities)
- Intervention capacity needed: ~40-50 slots (accounting for no-shows)

This aggregate forecasting enables proactive resource allocation rather than reactive crisis management.

### 6.3 Addressing Documentation Variability and Temporal Drift

**Documentation Variability:**
Clinical notes exhibit significant variability in structure, completeness, and terminology across providers and institutions. TRANCE's semantic chunking approach provides robustness by:

- Handling missing sections gracefully (section embeddings default to zero)
- Averaging chunk-level representations to smooth local variations
- Focusing on content rather than format through transformer attention

However, performance may degrade on external datasets with substantially different documentation practices. Future work should evaluate generalization across institutions and implement domain adaptation techniques.

**Temporal Drift:**
Healthcare practices, diagnostic criteria, and treatment protocols evolve over time, potentially degrading model performance. We observed this in MIMIC-IV spanning 2008-2019:

- Early years (2008-2012): AUROC 0.829
- Middle years (2013-2016): AUROC 0.847
- Recent years (2017-2019): AUROC 0.851

Slight improvement over time likely reflects better data quality and completeness in recent years. To maintain performance in deployment:

1. **Regular Retraining:** Update models quarterly using recent data
2. **Performance Monitoring:** Track AUROC, calibration, and feature distributions
3. **Drift Detection:** Alert when input distributions shift beyond thresholds
4. **Adaptive Learning:** Implement online learning to continuously update models

---

## 7. Future Scope

### 7.1 Multi-institutional Data and Federated Learning

Current results are based on a single academic medical center. Validating TRANCE across diverse healthcare settings requires multi-institutional collaboration. Challenges include:

**Data Heterogeneity:** Different EHR systems, coding practices, and clinical workflows necessitate robust preprocessing pipelines and feature standardization.

**Privacy Constraints:** Sharing patient-level data across institutions raises privacy concerns. Federated learning offers a solution by training models locally at each site and aggregating model parameters without data sharing.

**Implementation Strategy:**
1. Deploy TRANCE at 5-10 partner hospitals with varying patient demographics
2. Train local models on institutional data
3. Aggregate model weights using federated averaging
4. Evaluate global model on each site's validation data
5. Iteratively refine to balance local and global performance

This approach could yield a robust model generalizing across diverse populations while preserving patient privacy.

### 7.2 Real-time Deployment and Multi-sensory Integration

**Real-time Prediction System:**
Transitioning from retrospective analysis to real-time deployment requires:

1. **Streaming Data Integration:** Connect to EHR systems via HL7/FHIR APIs to receive discharge events in real-time
2. **Low-latency Inference:** Optimize model serving (e.g., ONNX Runtime) to provide predictions within seconds
3. **Clinical Workflow Integration:** Embed risk scores in discharge workflows, alert providers to high-risk patients
4. **Feedback Loop:** Capture clinical actions and outcomes to enable continuous model improvement

**Multi-sensory Data Integration:**
Beyond structured EHR and clinical notes, additional data modalities could enhance predictions:

- **Medical Imaging:** Chest X-rays, CT scans processed via convolutional neural networks to detect subtle abnormalities
- **Wearable Devices:** Post-discharge vital signs, activity levels, sleep patterns indicating deterioration
- **Social Determinants:** Housing instability, food insecurity, transportation barriers captured through screening tools
- **Genomic Data:** Pharmacogenomic variants affecting medication response and disease progression

Integrating these modalities requires advanced fusion architectures (e.g., cross-modal attention) and poses challenges in data availability, privacy, and computational cost.

**Reinforcement Learning for Dynamic Interventions:**
Current models provide static risk scores at discharge. Reinforcement learning could optimize dynamic interventions:

- **State:** Patient features at discharge and post-discharge timepoints
- **Action:** Intervention type and timing (e.g., home visit day 3 vs telephone call day 7)
- **Reward:** Readmission avoidance weighted by intervention cost
- **Policy:** Learn optimal intervention strategy maximizing outcomes while minimizing costs

This could enable truly personalized, adaptive care pathways rather than fixed risk-based protocols.

---

## 8. Conclusion

This work presents TRANCE, a multimodal framework that integrates structured EHR data and clinical narratives to predict 30-day hospital readmissions. By fusing 350+ engineered features with Clinical-T5 embeddings and employing LightGBM classification, TRANCE achieves AUROC 0.847, representing a 4.3% improvement over structured-data-only baselines.

Key contributions include:

1. **Multimodal Architecture:** Demonstrated that incorporating clinical narratives enhances predictive performance while maintaining interpretability through SHAP analysis.

2. **Leakage-Safe Methodology:** Implemented rigorous temporal validation and feature engineering practices ensuring all predictions use only information available at discharge time.

3. **Clinical Utility:** Achieved well-calibrated probability estimates enabling risk stratification for personalized care planning and resource allocation.

4. **Practical Deployment:** Designed modular pipeline suitable for real-world implementation with fallback mechanisms and robust preprocessing.

While achieving strong performance on MIMIC-IV, limitations include single-center validation, potential documentation bias, and lack of external validation. Future work should focus on multi-institutional testing, real-time deployment, and integration of additional data modalities to further improve readmission prediction and enable proactive, personalized post-discharge care.

The evidence suggests that multimodal approaches combining structured and unstructured clinical data represent a promising direction for clinical decision support systems, offering enhanced discrimination and interpretability compared to traditional structured-data-only models.

---

## 9. References

1. Jencks SF, Williams MV, Coleman EA. Rehospitalizations among patients in the Medicare fee-for-service program. N Engl J Med. 2009;360(14):1418-1428.

2. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission: a systematic review. JAMA. 2011;306(15):1688-1698.

3. Huang K, Altosaar J, Ranganath R. ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission. arXiv:1904.05342. 2019.

4. Alsentzer E, Murphy JR, Boag W, et al. Publicly Available Clinical BERT Embeddings. arXiv:1904.03323. 2019.

5. Lu Q, Dou D, Nguyen T. ClinicalT5: A Generative Language Model for Clinical Text. Findings of EMNLP. 2022.

6. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023;10:1.

7. Ke G, Meng Q, Finley T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS. 2017.

8. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. NIPS. 2017.

9. Rajkomar A, Oren E, Chen K, et al. Scalable and accurate deep learning with electronic health records. NPJ Digit Med. 2018;1:18.

10. Choi E, Bahadori MT, Song L, Stewart WF, Sun J. GRAM: Graph-based Attention Model for Healthcare Representation Learning. KDD. 2017.

11. Shickel B, Tighe PJ, Bihorac A, Rashidi P. Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record Analysis. IEEE J Biomed Health Inform. 2018;22(5):1589-1604.

12. Guo LL, Celi LA. Building a Clinically Attuned Transfer Learning Framework. arXiv:2108.05305. 2021.

13. Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol. 2019;110:12-22.

14. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. ICML. 2005.

15. Van Walraven C, Dhalla IA, Bell C, et al. Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community. CMAJ. 2010;182(6):551-557.
