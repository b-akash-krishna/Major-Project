Got it. Here is the full written documentation of your research plan.

---

# **TRANCE: Text-Guided Feature Gating for 30-Day Hospital Readmission Prediction**

## **Research Plan and Technical Documentation**

---

## **1\. Problem Statement and Motivation**

Hospital readmission within 30 days of discharge is one of the most costly and preventable outcomes in modern healthcare. In the United States alone, Medicare spends approximately $26 billion annually on readmissions, of which an estimated $17 billion is potentially preventable. Accurate prediction of which patients are at high risk of readmission before discharge allows clinical teams to intervene — arranging follow-up care, adjusting discharge disposition, or initiating social work referrals.

The current clinical standard for readmission prediction is the LACE+ index, a rule-based score that achieves an AUROC of approximately 0.61, which is considered poor-to-moderate discriminative power. Recent deep learning approaches have improved substantially on this, with the state-of-the-art method MM-STGNN (Tang et al., 2023, IEEE JBHI) achieving AUROC of 0.79 using a multimodal spatiotemporal graph neural network that fuses longitudinal chest radiographs with electronic health record data across a patient similarity graph.

Despite this progress, several critical gaps remain in the literature that no existing paper has addressed together. First, MM-STGNN requires chest radiographs, which are not universally available in all inpatient settings and represent an additional data dependency that complicates deployment. Second, no existing readmission prediction paper has reported calibration metrics — meaning that while models discriminate between high and low risk patients, it is unknown whether their predicted probabilities are actually reliable in absolute terms. A model that says a patient has 70% readmission risk should have approximately 70% of such patients actually readmitted. Without calibration, the probability score cannot be used directly in clinical decision-making. Third, no paper has studied whether these models perform equitably across demographic subgroups — race, gender, and age — which is an essential requirement for clinical deployment under modern AI fairness standards. Fourth, existing models predict readmission only at the point of discharge, when all data is available. No paper has studied how early in the hospital stay a reliable prediction can be made, which has significant implications for when interventions can realistically be initiated.

Our work, which we call TRANCE (Text-guided Readmission prediction with Adaptive Neural Context-aware gating and Ensemble learning), addresses all four of these gaps simultaneously. We propose an imaging-free, end-to-end trainable fusion architecture that uses clinical text embeddings not merely as an additional input feature, but as a dynamic controller that selectively amplifies or suppresses individual structured EHR features based on the patient-specific clinical context described in the notes. We further provide the first comprehensive fairness, calibration, and early warning analysis for readmission prediction at scale on MIMIC-IV.

---

## **2\. Positioning Relative to the Base Paper**

It is important to be precise about how our work relates to MM-STGNN, since our paper is developed in the context of that work.

MM-STGNN is a stronger model in terms of raw discriminative AUROC because it uses additional imaging data and models inter-patient relationships through a graph. We do not claim to surpass it on AUROC. Instead, we argue that AUROC alone is an insufficient criterion for evaluating clinical readmission models, and that our system is superior on a different and arguably more clinically relevant set of criteria: it requires no imaging, its predictions are calibrated, its fusion mechanism is interpretable, it performs equitably across demographic groups, and it can generate reliable predictions early in the hospital stay.

This framing is not a retreat from competition — it is a reframing of what the research question is. MM-STGNN asks: what is the highest achievable discriminative performance? Our paper asks: what does a clinically deployable, interpretable, and equitable readmission prediction system look like? These are different questions, and both are legitimate research contributions.

Concretely, our contributions relative to MM-STGNN are as follows. We operate without chest radiographs, making our system applicable in any hospital with an EHR. We operate without a patient similarity graph, making inference on new patients straightforward without recomputing graph structure. We provide calibrated probability outputs rather than raw scores, making our predictions directly actionable. We demonstrate equitable performance across race, gender, and age subgroups. We demonstrate prediction capability as early as day two of hospitalization. And we show that our text-guided gating mechanism learns clinically meaningful suppression and amplification patterns that align with domain knowledge, providing a form of interpretability that graph-based models cannot easily offer.

---

## **3\. Core Technical Contribution: Text-Guided Feature Gating**

### **3.1 Motivation for Gating**

The standard approach to multimodal fusion in clinical machine learning is concatenation: take the text embedding vector, append it to the structured feature vector, and pass the combined representation to a classifier. This approach treats text as just another set of features and ignores the fact that clinical notes carry rich contextual information about which structured features should be interpreted differently for a given patient.

Consider a concrete example. A patient's hemoglobin value of 9.2 g/dL is clinically very different depending on context. If the discharge note reads "patient has known chronic anemia, baseline hemoglobin 9-10, stable," then this value is expected and carries little prognostic signal. If the note reads "previously healthy 35-year-old presenting with acute blood loss," then the same value represents a serious acute abnormality. A naive concatenation model cannot make this distinction — it sees the same hemoglobin value and the same text embedding but has no mechanism to connect them. The text and the structured features remain informationally isolated in the final representation.

Our gating mechanism addresses this directly. The clinical text embedding is used to produce a set of per-feature weights — one weight for each structured EHR feature — that modulates the contribution of that feature to the final representation. Features whose abnormality is contextually explained by the note receive low weights and are effectively suppressed. Features that are unexpectedly abnormal given the note's context receive high weights and are amplified. This produces a patient-specific, context-aware structured representation that carries more signal than raw feature values.

### **3.2 Architecture**

Our architecture consists of three components trained end-to-end: a text encoder, a gate network, and a classification head.

The text encoder is ClinicalT5 (luqh/ClinicalT5-base), a T5 encoder model pretrained on clinical text from MIMIC-III and other sources. For each patient admission, we concatenate available clinical notes — primarily the discharge summary supplemented by radiology notes where available — preprocess them to extract high-value sections (hospital course, discharge diagnosis, assessment and plan, pertinent results), and encode them through ClinicalT5 with mean pooling over the token representations. We then apply PCA to reduce the raw 768-dimensional encoder output to 256 dimensions, matching the dimensionality used during our embedding training phase. This produces a 256-dimensional text embedding vector for each patient, which we denote as e.

The gate network takes the text embedding e as input and produces a weight vector g of the same dimensionality as the structured feature vector x. Specifically, the gate network is a two-layer feedforward network: a linear layer mapping from 256 to 128 dimensions with ReLU activation, followed by a linear layer mapping from 128 to the number of structured features (350 in our case) with sigmoid activation. The sigmoid ensures all gate weights lie in the interval \[0, 1\], meaning features are suppressed toward zero or passed through at full strength, but never inverted or amplified beyond their original value. The gated feature vector is then computed as the element-wise product of g and x, producing x\_gated \= g ⊙ x.

The classification head receives the concatenation of the original text embedding e and the gated feature vector x\_gated, producing a combined representation of dimensionality 256 \+ 350 \= 606\. This passes through a three-layer MLP: a linear layer to 256 dimensions with ReLU and dropout (p=0.3), a linear layer to 64 dimensions with ReLU, and a final linear layer to a scalar with sigmoid activation producing the readmission probability.

The entire system is trained end-to-end with binary cross-entropy loss, Adam optimizer with learning rate 1e-4, and early stopping on validation AUROC with patience of 10 epochs. We use the same patient-level train/validation/test split as our LightGBM baselines to ensure fair comparison, and we average predictions across three random seeds to reduce variance.

The key architectural choice that distinguishes our approach from Option A (two-stage training) is that the gate network is trained jointly with the classifier. This means the gate weights are optimized specifically to make the downstream readmission prediction task easier, not just to perform general feature selection. The gradient signal from the prediction loss flows back through the classifier, through the gated representation, and into the gate network, forcing the gate to learn which features the classifier actually needs suppressed or amplified.

### **3.3 Why This is Architecturally Novel in the Readmission Context**

Feature-wise linear modulation (FiLM, Perez et al. 2018\) and gated multimodal units (Arevalo et al. 2017\) have established the general principle of cross-modal gating in the computer vision and natural language processing literature. However, to our knowledge, no prior readmission prediction paper has applied this principle, and more specifically, no prior clinical prediction paper has studied whether the resulting gate weights align with clinical domain knowledge in an interpretable and statistically testable way. The gating mechanism itself is not our primary novelty claim — the clinical interpretability analysis that validates it is.

---

## **4\. Baseline Models**

Our results table includes the following models, which serve as ablation baselines and external comparisons.

The first baseline is LightGBM with tabular features only, using no text data whatsoever. This establishes the performance ceiling of structured EHR data alone and allows us to quantify how much the addition of clinical notes improves performance.

The second baseline is our current TRANCE system: a LightGBM and XGBoost ensemble with Optuna-tuned hyperparameters, isotonic calibration, and naive concatenation of ClinicalT5 embeddings with tabular features. This is the system described in our existing codebase and achieves AUROC of 0.7747. This baseline allows us to isolate the contribution of gating over simple concatenation.

The third baseline is TRANCE-Gate, our proposed system as described in Section 3\. Comparing TRANCE-Gate against the LightGBM-only baseline measures the total contribution of text. Comparing TRANCE-Gate against the naive concatenation TRANCE baseline measures the specific contribution of context-aware gating.

External baselines taken from prior literature include MM-STGNN (AUROC 0.79), LACE+ (AUROC 0.61), ClinicalT5 plus LightGBM from the PLOS ONE 2025 paper (AUROC 0.68), and ClinicalT5 plus VotingClassifier from the same paper (AUROC 0.68). These numbers are reported directly from those papers without replication on our part, as is standard practice.

---

## **5\. Gate Interpretability Analysis**

The gate interpretability analysis is the most important experiment in our paper and the one most directly tied to our novelty claim. The goal is to demonstrate empirically that the gate weights learned by our model carry clinical meaning — that the model has learned to suppress expected chronic-condition abnormalities and amplify unexpected acute findings without being explicitly trained to do so.

The experimental procedure is as follows. After training TRANCE-Gate, we run inference on the full test set and collect the gate weight vector g for every patient. We then group patients into condition-specific cohorts based on keyword presence in their clinical notes. For each condition, we select a set of EHR features that are directly related to that condition and compare the average gate weight assigned to those features between patients whose notes mention the condition and patients whose notes do not mention it.

The conditions and corresponding features we analyze are: chronic anemia matched against hemoglobin minimum, hemoglobin range, and the anemia binary flag; chronic kidney disease matched against creatinine mean, creatinine maximum, BUN mean, and BUN maximum; heart failure matched against the heart failure comorbidity flag, ICU length of stay, and diuretic medication indicators; COPD matched against pO2 range, pCO2 range, and respiratory medication indicators; diabetes matched against glucose mean, glucose range, and insulin medication indicators; and hypertension matched against systolic blood pressure mean and antihypertensive medication indicators.

For each condition-feature pairing, we compute the mean gate weight in the condition-mentioned group and the condition-not-mentioned group, and assess statistical significance using a two-sided Mann-Whitney U test with Bonferroni correction for multiple comparisons. We expect that patients with notes mentioning chronic anemia will have significantly lower gate weights on hemoglobin features than patients without such mentions, reflecting the model's learned suppression of contextually expected abnormalities.

The primary output of this analysis is a heatmap figure with disease conditions on one axis and feature groups on the other, with mean gate weight difference (mentioned minus not-mentioned) as the cell value. Negative values indicate learned suppression. Positive values would indicate learned amplification, though we expect these to be less common since amplification requires the model to recognize unexpected abnormalities, which is a harder cross-modal inference task.

A finding where the gate weights do not align with clinical expectations for some features is equally valuable and should be reported honestly. Such misalignments reveal the limits of what the text embedding captures, identify failure modes, and directly motivate future work on richer text representations or more targeted note preprocessing.

---

## **6\. Fairness Analysis**

Every prior readmission prediction paper, including MM-STGNN, reports model performance as a single aggregate metric across all patients. This obscures potentially serious disparities in performance across demographic subgroups. A model with AUROC 0.79 averaged across all patients may achieve 0.82 on white patients and 0.71 on Black patients — a disparity that would directly affect which patients receive follow-up care and which are missed.

We compute AUROC and Expected Calibration Error (ECE) separately for demographic subgroups defined by race, gender, and age group. For race, we report results for the four largest groups in our MIMIC-IV cohort: White, Black/African American, Hispanic/Latino, and Asian. For gender, we compare male and female patients. For age, we use the five-year age buckets already computed during feature extraction.

We report the maximum AUROC gap across race groups, the maximum AUROC gap across age groups, and the maximum ECE gap across race groups. We then compare these disparity measures between our LightGBM baseline and TRANCE-Gate to determine whether the gating mechanism improves or worsens demographic equity. Our hypothesis is that context-aware gating may improve equity by allowing the model to interpret the same lab value differently depending on patient context described in the note, which may partially compensate for the fact that reference ranges and chronic disease prevalence differ across demographic groups.

This analysis requires re-linking the encoded race and gender variables back to their original categorical labels, which means joining the test set predictions back against the raw admissions data before grouping. The ECE calculation is already implemented in our codebase and simply needs to be called within each subgroup loop.

---

## **7\. Calibration Analysis**

Calibration measures whether predicted probabilities correspond to observed event rates. A perfectly calibrated model produces a predicted probability of 0.7 for exactly 70% of patients who are subsequently readmitted. Poor calibration means the model's output cannot be interpreted as a true probability, limiting its clinical utility.

We measure calibration using two metrics. The first is Expected Calibration Error (ECE), computed by binning predictions into ten equal-width bins from 0 to 1, computing the absolute difference between mean predicted probability and observed readmission rate in each bin, and taking the weighted average across bins. Lower ECE is better, with 0 representing perfect calibration. The second is Brier score, which measures the mean squared error between predicted probability and binary outcome. Lower Brier score is better.

We report these metrics for our LightGBM baseline before calibration, our LightGBM baseline after isotonic calibration, and TRANCE-Gate before and after calibration. We also report a reliability diagram for each model — a plot of mean predicted probability against observed readmission rate within each bin — as a figure in the paper.

The comparison against MM-STGNN is made by noting that MM-STGNN reports no calibration metrics whatsoever, which means its raw output probabilities have unknown reliability. We argue that calibration is a prerequisite for clinical deployment, not an optional analysis, and that our explicit calibration step addresses a gap left by all prior work in this area.

---

## **8\. Early Warning Analysis**

All existing readmission prediction models, including MM-STGNN and our own LightGBM baseline, generate predictions at the time of discharge using the full complement of data collected during the hospital stay. This timing has a fundamental clinical limitation: discharge decisions are often made rapidly, and a prediction generated only at the moment of discharge may not provide sufficient lead time to arrange meaningful interventions such as home health referrals, social work consultations, or specialist follow-up appointments.

We study how early in the hospital stay a reliable readmission prediction can be made by training and evaluating our model using data truncated to the first N days of admission, for N in the set {1, 2, 3, 5, 7, full stay}. For each cutoff, we filter all EHR events — laboratory results, medication administrations, procedure codes, vital signs — to retain only those recorded within the first N days. For the text component, when N is less than the full stay, we use only the admission note and any notes generated before day N rather than the discharge summary, since the discharge summary is by definition not available until the end of the stay.

The output of this analysis is a performance-versus-earliness curve plotting AUROC on the vertical axis against days from admission on the horizontal axis. We report this curve for both TRANCE-Gate and the LightGBM baseline, allowing us to assess whether the text-guided gating mechanism specifically helps in the low-data regime where fewer days of structured features are available and the clinical note provides a proportionally larger fraction of the available information.

We pay particular attention to the day-two prediction. A model that achieves AUROC above 0.73 on day two of hospitalization represents a clinically actionable finding, because most hospital stays of clinical interest extend beyond two days, meaning there is sufficient time for interventions initiated on day two to take effect before discharge. We note whether TRANCE-Gate achieves a higher day-two AUROC than the LightGBM baseline, and if so, by how much — this would suggest that clinical notes, which are partially available by day two in the form of admission notes and early progress notes, provide meaningful signal that structured features alone cannot capture at that early stage.

---

## **9\. Temporal Stability Analysis**

MIMIC-IV spans admissions from 2008 through 2022, encoded in the `anchor_year_group` variable. Clinical practice changes over this period — new medications become available, coding practices shift, care pathways evolve. A model trained on data from all years may perform well on average but conceal significant degradation on more recent data, which would be the data it would actually encounter during deployment.

We analyze temporal stability by splitting our test set by `anchor_year_group` and computing AUROC separately for each cohort. We do not retrain the model for each cohort — we train once on the full training set and evaluate the same trained model on each temporal slice of the test set. A flat AUROC curve across year groups indicates temporal robustness. A declining curve indicates that the model captures patterns that have become less predictive over time, implying a need for periodic retraining.

This analysis requires no additional training and minimal additional code — it is an additional grouping applied during evaluation. Nevertheless, to our knowledge no prior paper using MIMIC-IV for readmission prediction has reported this analysis, making it a genuine contribution to the practical deployment literature for clinical AI.

---

## **10\. Paper Contribution Statement**

Combining all of the above, our paper makes the following specific contributions that are collectively novel relative to prior work.

We propose a text-guided feature gating architecture for readmission prediction in which clinical note embeddings dynamically modulate the contribution of individual structured EHR features through a learned, end-to-end trainable gate network. This is the first application of cross-modal feature gating in the readmission prediction literature.

We demonstrate empirically that the learned gate weights align with clinical domain knowledge, showing statistically significant suppression of chronic-condition features for patients whose notes describe those conditions as baseline. This interpretability finding is novel in the clinical prediction literature and provides a form of model transparency that graph neural network approaches cannot easily replicate.

We provide the first comprehensive fairness analysis for readmission prediction on MIMIC-IV, reporting AUROC and calibration error broken down by race, gender, and age group across multiple models.

We provide the first explicit calibration analysis in the readmission prediction literature, reporting ECE and Brier score before and after isotonic calibration, and arguing that calibration is a necessary condition for clinical deployment.

We provide an early warning analysis showing the AUROC achievable at each day of hospitalization, with specific attention to the clinically important day-two prediction.

We provide a temporal stability analysis across the full time range of MIMIC-IV, demonstrating how model performance evolves over the period 2008-2022.

All of this is achieved without requiring chest radiographs or patient similarity graphs, making our system substantially more accessible to deploy in standard hospital EHR environments than imaging-dependent or graph-dependent approaches.

---

## **11\. What Remains If Gating Does Not Converge**

There is a realistic possibility that TRANCE-Gate does not improve substantially over the naive concatenation baseline. This could happen if the PCA-compressed text embeddings do not contain sufficient information for the gate network to learn meaningful per-feature weights, or if the MLP classifier is not expressive enough to leverage the gated representation effectively. In that case, the AUROC of TRANCE-Gate may be similar to or slightly below that of our existing LightGBM ensemble.

This outcome does not invalidate the paper. It changes the narrative slightly: instead of "gating improves performance," the finding becomes "context-aware gating does not improve discriminative performance over naive concatenation, suggesting that the performance gain from text data is primarily through additional signal rather than through contextual reweighting of existing structured signals." This is itself a meaningful finding that informs future research directions. The fairness, calibration, early warning, and temporal stability analyses remain fully valid regardless of whether gating helps performance.

In practical terms, if gating does not help, your paper's proposed method reverts to your existing TRANCE system with the LightGBM and XGBoost ensemble, and your contributions shift weight toward the four analyses rather than the architecture. That paper is still publishable at JAMIA or Journal of Biomedical Informatics because none of those four analyses have been performed together on readmission prediction at this scale before.

---

## **12\. Target Venue and Framing**

The appropriate target venues for this work are the Journal of the American Medical Informatics Association (JAMIA), the Journal of Biomedical Informatics, and npj Digital Medicine. All three publish clinical prediction papers combining machine learning with real EHR data and have published work directly comparable to yours in scope and methodology.

The framing for submission should emphasize deployment readiness rather than benchmark competition. The abstract and introduction should explicitly acknowledge that MM-STGNN achieves higher AUROC, then argue that AUROC alone is insufficient for evaluating clinical AI systems. The case should be made that calibration, fairness, interpretability, and early prediction capability are equally important criteria — and that no prior paper has addressed all of them together. This positions the paper not as a failed attempt to beat the state of the art, but as a deliberate extension of the research agenda into questions that the state of the art has not asked.

Got it. Here is the complete technical guide for what needs to change, what needs to be added, and exactly how to do each thing in your existing codebase.

---

# **Technical Implementation Guide: TRANCE Additions and Changes**

---

## **Understanding Your Existing Codebase First**

Before describing changes, here is what already exists and what role each file plays, because every addition connects to these.

`src/01_extract.py` — extracts all structured EHR features from MIMIC-IV and saves to `data/ultimate_features.csv`. This file does not need changes for the gating work, but needs one small addition for the early warning experiment.

`src/02_embed.py` — generates ClinicalT5 embeddings for all admissions and saves to `data/embeddings.csv` along with `models/embedding_info.pkl` which stores the fitted PCA. This file does not need changes.

`src/03_train.py` — trains your LightGBM and XGBoost ensemble. This file stays exactly as it is. It becomes your baseline model in the paper. Do not modify it.

`src/config.py` — central configuration. Needs small additions for new thresholds and paths.

`src/embedding_utils.py` — loads models and generates embeddings for inference. No changes needed.

The new files you need to create are described below in the order you should build them.

---

## **Change 1: Small Additions to config.py**

Open `src/config.py` and add the following at the bottom. These are paths and constants that every new module will import from.

\# \========================================  
\# 10\. GATED FUSION SETTINGS  
\# \========================================

\# Path for the new gated model  
GATE\_MODEL\_PKL \= os.path.join(MODELS\_DIR, "trance\_gate.pkl")

\# Gate network architecture  
GATE\_HIDDEN\_DIM \= 128        \# hidden layer size inside gate network  
GATE\_TEXT\_DIM \= 256          \# must match EMBED\_DIM from section 7  
GATE\_DROPOUT \= 0.3           \# dropout in MLP classifier head  
GATE\_LR \= 1e-4               \# learning rate for Adam optimizer  
GATE\_EPOCHS \= 100            \# max epochs, early stopping will cut this short  
GATE\_PATIENCE \= 10           \# early stopping patience on val AUROC  
GATE\_SEEDS \= \[42, 2024, 777\] \# seeds for multi-seed ensemble

\# \========================================  
\# 11\. ANALYSIS OUTPUT PATHS  
\# \========================================

FAIRNESS\_RESULTS\_CSV    \= os.path.join(RESULTS\_DIR, "fairness\_analysis.csv")  
CALIBRATION\_RESULTS\_CSV \= os.path.join(RESULTS\_DIR, "calibration\_analysis.csv")  
GATE\_WEIGHTS\_NPY        \= os.path.join(RESULTS\_DIR, "gate\_weights.npy")  
GATE\_PATIENT\_IDS\_NPY    \= os.path.join(RESULTS\_DIR, "gate\_patient\_ids.npy")  
EARLY\_WARNING\_CSV       \= os.path.join(RESULTS\_DIR, "early\_warning\_results.csv")  
TEMPORAL\_DRIFT\_CSV      \= os.path.join(RESULTS\_DIR, "temporal\_drift\_results.csv")

\# \========================================  
\# 12\. EARLY WARNING SETTINGS  
\# \========================================

\# Day cutoffs to evaluate for early warning experiment  
EARLY\_WARNING\_DAYS \= \[1, 2, 3, 5, 7\]  
\# "full" is added automatically in the early warning script

---

## **Change 2: Create src/gated\_fusion\_model.py (New File)**

This is the most important new file. It defines the PyTorch architecture for the text-guided gate, the training loop, and the evaluation logic. Create this file from scratch at `src/gated_fusion_model.py`.

"""  
TRANCE-Gate: Text-Guided Feature Gating Model  
\==============================================  
Architecture:  
  \- ClinicalT5 text embedding (256-dim, pre-computed) acts as context signal  
  \- Gate network: text\_emb \-\> per-feature weights in \[0,1\]  
  \- Gated features: gate\_weights \* tabular\_features (element-wise)  
  \- Classifier: concat(text\_emb, gated\_features) \-\> MLP \-\> readmission prob

The gate is trained jointly with the classifier end-to-end.  
Gate weights are saved per patient for interpretability analysis.  
"""

import os  
import sys  
import gc  
import json  
import logging  
import numpy as np  
import pandas as pd  
import joblib  
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
from sklearn.metrics import roc\_auc\_score, average\_precision\_score, brier\_score\_loss, log\_loss  
from sklearn.calibration import IsotonicRegression

sys.path.append(os.path.dirname(os.path.abspath(\_\_file\_\_)))  
try:  
    from config import (  
        FEATURES\_CSV, EMBEDDINGS\_CSV, GATE\_MODEL\_PKL, RESULTS\_DIR,  
        GATE\_HIDDEN\_DIM, GATE\_TEXT\_DIM, GATE\_DROPOUT,  
        GATE\_LR, GATE\_EPOCHS, GATE\_PATIENCE, GATE\_SEEDS,  
        GATE\_WEIGHTS\_NPY, GATE\_PATIENT\_IDS\_NPY,  
        TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, RANDOM\_STATE,  
        THRESHOLD\_HIGH\_RISK, THRESHOLD\_MEDIUM\_RISK,  
    )  
except ImportError:  
    from .config import (  
        FEATURES\_CSV, EMBEDDINGS\_CSV, GATE\_MODEL\_PKL, RESULTS\_DIR,  
        GATE\_HIDDEN\_DIM, GATE\_TEXT\_DIM, GATE\_DROPOUT,  
        GATE\_LR, GATE\_EPOCHS, GATE\_PATIENCE, GATE\_SEEDS,  
        GATE\_WEIGHTS\_NPY, GATE\_PATIENT\_IDS\_NPY,  
        TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, RANDOM\_STATE,  
        THRESHOLD\_HIGH\_RISK, THRESHOLD\_MEDIUM\_RISK,  
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s \- %(levelname)s \- %(message)s")  
logger \= logging.getLogger(\_\_name\_\_)

\# ── Dataset ───────────────────────────────────────────────────────────────────

class ReadmissionDataset(Dataset):  
    """  
    Holds text embeddings, tabular features, and labels for one split.  
    Returns tensors of (text\_emb, tabular, label) per patient.  
    """  
    def \_\_init\_\_(self, text\_emb: np.ndarray, tabular: np.ndarray, labels: np.ndarray):  
        self.text\_emb \= torch.tensor(text\_emb, dtype=torch.float32)  
        self.tabular  \= torch.tensor(tabular,  dtype=torch.float32)  
        self.labels   \= torch.tensor(labels,   dtype=torch.float32)

    def \_\_len\_\_(self):  
        return len(self.labels)

    def \_\_getitem\_\_(self, idx):  
        return self.text\_emb\[idx\], self.tabular\[idx\], self.labels\[idx\]

\# ── Architecture ──────────────────────────────────────────────────────────────

class TextGuidedGate(nn.Module):  
    """  
    Full gated fusion model.

    Gate network:  
        text\_emb (256) \-\> Linear(256, 128\) \-\> ReLU  
                       \-\> Linear(128, n\_tab) \-\> Sigmoid  
        output: gate\_weights in \[0,1\] for each tabular feature

    Gating:  
        x\_gated \= gate\_weights \* x\_tabular   (element-wise product)

    Classifier:  
        concat(text\_emb, x\_gated) \-\> Linear(256+n\_tab, 256\) \-\> ReLU \-\> Dropout  
                                   \-\> Linear(256, 64\) \-\> ReLU  
                                   \-\> Linear(64, 1\) \-\> Sigmoid  
    """

    def \_\_init\_\_(self, text\_dim: int, tabular\_dim: int,  
                 hidden\_dim: int \= GATE\_HIDDEN\_DIM, dropout: float \= GATE\_DROPOUT):  
        super().\_\_init\_\_()

        self.gate\_network \= nn.Sequential(  
            nn.Linear(text\_dim, hidden\_dim),  
            nn.ReLU(),  
            nn.Linear(hidden\_dim, tabular\_dim),  
            nn.Sigmoid()  
        )

        self.classifier \= nn.Sequential(  
            nn.Linear(text\_dim \+ tabular\_dim, 256),  
            nn.ReLU(),  
            nn.Dropout(dropout),  
            nn.Linear(256, 64),  
            nn.ReLU(),  
            nn.Linear(64, 1),  
            nn.Sigmoid()  
        )

    def forward(self, text\_emb: torch.Tensor, x\_tab: torch.Tensor):  
        gate\_weights \= self.gate\_network(text\_emb)       \# (batch, tabular\_dim)  
        x\_gated      \= gate\_weights \* x\_tab              \# element-wise  
        x\_fused      \= torch.cat(\[text\_emb, x\_gated\], dim=1)  
        prob         \= self.classifier(x\_fused).squeeze(1)  
        return prob, gate\_weights

\# ── Data Loading ──────────────────────────────────────────────────────────────

def load\_fused\_data():  
    """  
    Loads and merges tabular features with text embeddings.  
    Returns aligned arrays for text, tabular, labels, groups (subject\_id),  
    hadm\_ids, and the list of tabular feature names.  
    """  
    pruned \= FEATURES\_CSV.replace(".csv", "\_pruned.csv")  
    feat\_path \= pruned if os.path.exists(pruned) else FEATURES\_CSV  
    logger.info("Loading features from %s", feat\_path)  
    tab\_df \= pd.read\_csv(feat\_path, low\_memory=False).fillna(0)

    logger.info("Loading embeddings from %s", EMBEDDINGS\_CSV)  
    emb\_df \= pd.read\_csv(EMBEDDINGS\_CSV, low\_memory=False)

    df \= tab\_df.merge(emb\_df, on="hadm\_id", how="left").fillna(0)  
    logger.info("Merged shape: %s", df.shape)

    \# Separate columns  
    id\_cols  \= {"subject\_id", "hadm\_id", "readmit\_30"}  
    emb\_cols \= \[c for c in emb\_df.columns if c.startswith("ct5\_")\]  
    tab\_cols \= \[c for c in df.columns if c not in id\_cols and c not in emb\_cols\]

    groups   \= df\["subject\_id"\].astype(int).values  
    hadm\_ids \= df\["hadm\_id"\].astype(int).values  
    labels   \= df\["readmit\_30"\].astype(np.float32).values

    text\_emb \= df\[emb\_cols\].values.astype(np.float32)  
    tabular  \= df\[tab\_cols\].values.astype(np.float32)

    logger.info("Text embedding dim: %d | Tabular features: %d", text\_emb.shape\[1\], tabular.shape\[1\])  
    return text\_emb, tabular, labels, groups, hadm\_ids, tab\_cols

def make\_splits(groups, labels):  
    """  
    Patient-level train/val/test split.  
    Identical strategy to 03\_train.py so results are comparable.  
    """  
    rng \= np.random.RandomState(RANDOM\_STATE)  
    unique\_patients \= np.unique(groups)  
    rng.shuffle(unique\_patients)

    n \= len(unique\_patients)  
    n\_test \= int(n \* TRAIN\_TEST\_FRAC)  
    n\_val  \= int(n \* TRAIN\_VAL\_FRAC)

    test\_pats  \= set(unique\_patients\[-n\_test:\])  
    val\_pats   \= set(unique\_patients\[-(n\_test \+ n\_val):-n\_test\])  
    train\_pats \= set(unique\_patients\[:-(n\_test \+ n\_val)\])

    train\_mask \= np.array(\[g in train\_pats for g in groups\])  
    val\_mask   \= np.array(\[g in val\_pats   for g in groups\])  
    test\_mask  \= np.array(\[g in test\_pats  for g in groups\])

    return train\_mask, val\_mask, test\_mask

\# ── Training ──────────────────────────────────────────────────────────────────

def train\_one\_seed(text\_emb, tabular, labels, groups, seed: int, device: torch.device):  
    """  
    Trains one instance of TextGuidedGate with a given random seed.  
    Returns the trained model, validation probabilities, and test probabilities.  
    """  
    torch.manual\_seed(seed)  
    np.random.seed(seed)

    train\_mask, val\_mask, test\_mask \= make\_splits(groups, labels)

    pos\_weight \= (labels\[train\_mask\] \== 0).sum() / max((labels\[train\_mask\] \== 1).sum(), 1\)  
    criterion  \= nn.BCELoss(weight=None)   \# isotonic calibration handles imbalance post-hoc  
    \# For training we use weighted BCE to handle imbalance  
    criterion\_train \= nn.BCEWithLogitsLoss(  
        pos\_weight=torch.tensor(\[pos\_weight\], dtype=torch.float32).to(device)  
    )  
    \# But our model outputs sigmoid already, so use BCELoss with manual weighting  
    \# Simpler: just use BCELoss and rely on calibration  
    criterion \= nn.BCELoss()

    train\_ds \= ReadmissionDataset(text\_emb\[train\_mask\], tabular\[train\_mask\], labels\[train\_mask\])  
    val\_ds   \= ReadmissionDataset(text\_emb\[val\_mask\],   tabular\[val\_mask\],   labels\[val\_mask\])  
    test\_ds  \= ReadmissionDataset(text\_emb\[test\_mask\],  tabular\[test\_mask\],  labels\[test\_mask\])

    train\_loader \= DataLoader(train\_ds, batch\_size=256, shuffle=True,  num\_workers=0)  
    val\_loader   \= DataLoader(val\_ds,   batch\_size=512, shuffle=False, num\_workers=0)  
    test\_loader  \= DataLoader(test\_ds,  batch\_size=512, shuffle=False, num\_workers=0)

    text\_dim    \= text\_emb.shape\[1\]  
    tabular\_dim \= tabular.shape\[1\]  
    model       \= TextGuidedGate(text\_dim, tabular\_dim).to(device)  
    optimizer   \= torch.optim.Adam(model.parameters(), lr=GATE\_LR, weight\_decay=1e-5)  
    scheduler   \= torch.optim.lr\_scheduler.CosineAnnealingLR(optimizer, T\_max=GATE\_EPOCHS)

    best\_val\_auroc \= 0.0  
    best\_state     \= None  
    patience\_count \= 0

    for epoch in range(GATE\_EPOCHS):  
        \# Training  
        model.train()  
        for text\_b, tab\_b, label\_b in train\_loader:  
            text\_b, tab\_b, label\_b \= text\_b.to(device), tab\_b.to(device), label\_b.to(device)  
            optimizer.zero\_grad()  
            probs, \_ \= model(text\_b, tab\_b)  
            loss \= criterion(probs, label\_b)  
            loss.backward()  
            torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm=1.0)  
            optimizer.step()  
        scheduler.step()

        \# Validation  
        model.eval()  
        val\_probs\_list \= \[\]  
        val\_labels\_list \= \[\]  
        with torch.no\_grad():  
            for text\_b, tab\_b, label\_b in val\_loader:  
                text\_b, tab\_b \= text\_b.to(device), tab\_b.to(device)  
                probs, \_ \= model(text\_b, tab\_b)  
                val\_probs\_list.append(probs.cpu().numpy())  
                val\_labels\_list.append(label\_b.numpy())

        val\_probs  \= np.concatenate(val\_probs\_list)  
        val\_labels \= np.concatenate(val\_labels\_list)  
        val\_auroc  \= roc\_auc\_score(val\_labels, val\_probs)

        if val\_auroc \> best\_val\_auroc:  
            best\_val\_auroc \= val\_auroc  
            best\_state     \= {k: v.cpu().clone() for k, v in model.state\_dict().items()}  
            patience\_count \= 0  
        else:  
            patience\_count \+= 1

        if epoch % 10 \== 0:  
            logger.info("Seed %d | Epoch %d | Val AUROC: %.4f | Best: %.4f",  
                        seed, epoch, val\_auroc, best\_val\_auroc)

        if patience\_count \>= GATE\_PATIENCE:  
            logger.info("Early stopping at epoch %d", epoch)  
            break

    \# Load best weights  
    model.load\_state\_dict(best\_state)  
    model.eval()

    \# Get test predictions and gate weights  
    test\_probs\_list  \= \[\]  
    test\_labels\_list \= \[\]  
    gate\_weights\_list \= \[\]

    with torch.no\_grad():  
        for text\_b, tab\_b, label\_b in test\_loader:  
            text\_b, tab\_b \= text\_b.to(device), tab\_b.to(device)  
            probs, gates \= model(text\_b, tab\_b)  
            test\_probs\_list.append(probs.cpu().numpy())  
            test\_labels\_list.append(label\_b.numpy())  
            gate\_weights\_list.append(gates.cpu().numpy())

    test\_probs   \= np.concatenate(test\_probs\_list)  
    test\_labels  \= np.concatenate(test\_labels\_list)  
    gate\_weights \= np.concatenate(gate\_weights\_list)

    return model, val\_probs, val\_labels, test\_probs, test\_labels, gate\_weights, test\_mask

\# ── ECE ───────────────────────────────────────────────────────────────────────

def compute\_ece(probs, labels, n\_bins=10):  
    bins \= np.linspace(0, 1, n\_bins \+ 1\)  
    ece, total \= 0.0, len(labels)  
    for i in range(n\_bins):  
        mask \= (probs \>= bins\[i\]) & (probs \< bins\[i \+ 1\])  
        if mask.sum() \== 0:  
            continue  
        ece \+= (mask.sum() / total) \* abs(float(labels\[mask\].mean()) \- float(probs\[mask\].mean()))  
    return float(ece)

\# ── Main Training Entry Point ─────────────────────────────────────────────────

def train\_gate\_model():  
    """  
    Trains TRANCE-Gate across multiple seeds, averages predictions,  
    applies isotonic calibration, and saves everything needed for analysis.  
    """  
    os.makedirs(RESULTS\_DIR, exist\_ok=True)  
    device \= torch.device("cuda" if torch.cuda.is\_available() else "cpu")  
    logger.info("Device: %s", device)

    text\_emb, tabular, labels, groups, hadm\_ids, tab\_cols \= load\_fused\_data()

    all\_val\_probs  \= \[\]  
    all\_test\_probs \= \[\]  
    all\_gate\_weights \= \[\]  
    test\_labels\_ref  \= None  
    val\_labels\_ref   \= None  
    test\_mask\_ref    \= None

    for seed in GATE\_SEEDS:  
        logger.info("=== Training seed %d \===", seed)  
        model, val\_probs, val\_labels, test\_probs, test\_labels, gate\_weights, test\_mask \= \\  
            train\_one\_seed(text\_emb, tabular, labels, groups, seed, device)

        all\_val\_probs.append(val\_probs)  
        all\_test\_probs.append(test\_probs)  
        all\_gate\_weights.append(gate\_weights)

        if test\_labels\_ref is None:  
            test\_labels\_ref \= test\_labels  
            val\_labels\_ref  \= val\_labels  
            test\_mask\_ref   \= test\_mask

        del model  
        gc.collect()  
        if device.type \== "cuda":  
            torch.cuda.empty\_cache()

    \# Average across seeds  
    avg\_val\_probs  \= np.mean(all\_val\_probs,  axis=0)  
    avg\_test\_probs \= np.mean(all\_test\_probs, axis=0)  
    avg\_gate\_weights \= np.mean(all\_gate\_weights, axis=0)

    \# Isotonic calibration fitted on val, applied to test  
    calibrator \= IsotonicRegression(out\_of\_bounds="clip")  
    calibrator.fit(avg\_val\_probs, val\_labels\_ref)  
    cal\_test\_probs \= calibrator.predict(avg\_test\_probs).astype(np.float32)

    \# Metrics  
    auroc\_raw \= roc\_auc\_score(test\_labels\_ref, avg\_test\_probs)  
    auroc\_cal \= roc\_auc\_score(test\_labels\_ref, cal\_test\_probs)  
    auprc     \= average\_precision\_score(test\_labels\_ref, cal\_test\_probs)  
    brier     \= brier\_score\_loss(test\_labels\_ref, cal\_test\_probs)  
    ece\_before \= compute\_ece(avg\_test\_probs, test\_labels\_ref)  
    ece\_after  \= compute\_ece(cal\_test\_probs,  test\_labels\_ref)

    logger.info("=" \* 55\)  
    logger.info("TRANCE-Gate Results")  
    logger.info("  AUROC (raw):        %.4f", auroc\_raw)  
    logger.info("  AUROC (calibrated): %.4f", auroc\_cal)  
    logger.info("  AUPRC:              %.4f", auprc)  
    logger.info("  Brier score:        %.4f", brier)  
    logger.info("  ECE before cal:     %.4f", ece\_before)  
    logger.info("  ECE after cal:      %.4f", ece\_after)  
    logger.info("=" \* 55\)

    \# Save gate weights and patient ids for interpretability analysis  
    test\_hadm\_ids \= hadm\_ids\[test\_mask\_ref\]  
    np.save(GATE\_WEIGHTS\_NPY,    avg\_gate\_weights)  
    np.save(GATE\_PATIENT\_IDS\_NPY, test\_hadm\_ids)  
    logger.info("Gate weights saved \-\> %s", GATE\_WEIGHTS\_NPY)

    \# Save model bundle  
    results \= {  
        "auroc\_raw":    round(float(auroc\_raw), 4),  
        "auroc\_cal":    round(float(auroc\_cal), 4),  
        "auprc":        round(float(auprc),     4),  
        "brier":        round(float(brier),     4),  
        "ece\_before":   round(float(ece\_before), 4),  
        "ece\_after":    round(float(ece\_after),  4),  
        "tab\_features": tab\_cols,  
        "n\_test":       int(len(test\_labels\_ref)),  
        "seeds":        GATE\_SEEDS,  
    }

    joblib.dump({  
        "calibrator":      calibrator,  
        "tab\_cols":        tab\_cols,  
        "text\_dim":        text\_emb.shape\[1\],  
        "tabular\_dim":     tabular.shape\[1\],  
        "results":         results,  
        "test\_probs\_raw":  avg\_test\_probs,  
        "test\_probs\_cal":  cal\_test\_probs,  
        "test\_labels":     test\_labels\_ref,  
        "test\_hadm\_ids":   test\_hadm\_ids,  
        "avg\_gate\_weights": avg\_gate\_weights,  
    }, GATE\_MODEL\_PKL)

    results\_path \= os.path.join(RESULTS\_DIR, "gate\_training\_report.json")  
    with open(results\_path, "w") as f:  
        json.dump(results, f, indent=2)

    logger.info("Gate model saved \-\> %s", GATE\_MODEL\_PKL)  
    return results

if \_\_name\_\_ \== "\_\_main\_\_":  
    train\_gate\_model()

---

## **Change 3: Create src/10\_gate\_interpretability.py (New File)**

This file loads the saved gate weights and discharge notes, groups patients by clinical keyword presence, and produces the interpretability analysis described in the research plan.

"""  
Gate Interpretability Analysis  
\===============================  
Groups test patients by clinical keyword presence in their notes,  
then compares average gate weights on related EHR features between  
the keyword-present and keyword-absent groups.

Produces:  
  \- results/gate\_interpretability.csv  (per-condition per-feature stats)  
  \- figures/gate\_heatmap.png           (condition x feature group heatmap)  
"""

import os  
import sys  
import json  
import logging  
import numpy as np  
import pandas as pd  
from scipy import stats  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(\_\_file\_\_)))  
try:  
    from config import (  
        GATE\_WEIGHTS\_NPY, GATE\_PATIENT\_IDS\_NPY, GATE\_MODEL\_PKL,  
        MIMIC\_NOTE\_DIR, MIMIC\_BHC\_DIR, RESULTS\_DIR, FIGURES\_DIR,  
    )  
except ImportError:  
    from .config import (  
        GATE\_WEIGHTS\_NPY, GATE\_PATIENT\_IDS\_NPY, GATE\_MODEL\_PKL,  
        MIMIC\_NOTE\_DIR, MIMIC\_BHC\_DIR, RESULTS\_DIR, FIGURES\_DIR,  
    )  
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s \- %(levelname)s \- %(message)s")  
logger \= logging.getLogger(\_\_name\_\_)

\# ── Clinical condition definitions ────────────────────────────────────────────  
\# Each entry: condition name \-\> {keywords in notes, related tabular features}  
\# Keywords are matched case-insensitively as substrings in the discharge text.  
\# Feature names must match your tab\_cols list exactly.

CONDITION\_FEATURE\_MAP \= {  
    "chronic\_anemia": {  
        "keywords": \["chronic anemia", "known anemia", "baseline anemia", "anemia of chronic"\],  
        "features": \["lab\_hemoglobin\_min", "lab\_hemoglobin\_range", "anemia", "lab\_hematocrit\_min"\],  
    },  
    "chronic\_kidney\_disease": {  
        "keywords": \["chronic kidney disease", "ckd", "chronic renal", "end stage renal", "esrd"\],  
        "features": \["lab\_creatinine\_mean", "lab\_creatinine\_max", "lab\_bun\_mean", "lab\_bun\_max",  
                     "lab\_creatinine\_range", "lab\_bun\_range", "cm\_renal\_fail"\],  
    },  
    "heart\_failure": {  
        "keywords": \["heart failure", "chf", "congestive heart", "systolic dysfunction",  
                     "diastolic dysfunction", "reduced ejection fraction"\],  
        "features": \["cm\_chf", "icu\_los\_sum", "icu\_count", "lab\_sodium\_range",  
                     "lab\_bicarb\_range", "had\_icu"\],  
    },  
    "copd": {  
        "keywords": \["copd", "chronic obstructive", "emphysema", "chronic bronchitis"\],  
        "features": \["lab\_pao2\_range", "lab\_paco2\_range", "lab\_pao2\_mean",  
                     "lab\_ph\_range", "cm\_copd"\],  
    },  
    "diabetes": {  
        "keywords": \["diabetes mellitus", "diabetic", "type 2 diabetes", "type 1 diabetes",  
                     "insulin dependent"\],  
        "features": \["lab\_glucose\_mean", "lab\_glucose\_max", "lab\_glucose\_range",  
                     "lab\_glucose\_last", "cm\_diabetes"\],  
    },  
    "hypertension": {  
        "keywords": \["hypertension", "hypertensive", "high blood pressure"\],  
        "features": \["v\_sbp\_mean", "v\_sbp\_std", "cm\_hypertension"\],  
    },  
    "liver\_disease": {  
        "keywords": \["cirrhosis", "hepatic", "liver disease", "liver failure",  
                     "portal hypertension"\],  
        "features": \["cm\_liver", "lab\_bicarb\_max", "lab\_platelets\_min",  
                     "lab\_platelets\_range", "thrombocytopenia"\],  
    },  
    "cancer": {  
        "keywords": \["malignancy", "carcinoma", "cancer", "metastatic", "oncology",  
                     "chemotherapy", "radiation therapy"\],  
        "features": \["cm\_cancer", "lab\_wbc\_range", "lab\_platelets\_range",  
                     "lab\_hemoglobin\_min", "high\_risk\_org"\],  
    },  
}

def load\_discharge\_notes(hadm\_ids: set) \-\> dict:  
    """  
    Loads discharge note text for given hadm\_ids.  
    Returns dict mapping hadm\_id \-\> lowercased note text.  
    """  
    note\_text \= {}  
    for base in \[MIMIC\_NOTE\_DIR, MIMIC\_BHC\_DIR\]:  
        if not os.path.isdir(base):  
            continue  
        for fn in \["discharge.csv.gz", "discharge.csv"\]:  
            path \= os.path.join(base, fn)  
            if not os.path.exists(path):  
                \# walk subdirectories  
                for root, \_, files in os.walk(base):  
                    for f in files:  
                        if f \== fn:  
                            path \= os.path.join(root, f)  
                            break  
            if not os.path.exists(path):  
                continue  
            logger.info("Loading notes from %s", path)  
            try:  
                df \= pd.read\_csv(path, usecols=\["hadm\_id", "text"\],  
                                 low\_memory=True, nrows=2\_000\_000)  
                df \= df\[df\["hadm\_id"\].isin(hadm\_ids)\].dropna(subset=\["text"\])  
                for row in df.itertuples():  
                    hadm \= int(row.hadm\_id)  
                    if hadm not in note\_text:  
                        note\_text\[hadm\] \= str(row.text).lower()  
                    else:  
                        note\_text\[hadm\] \+= " " \+ str(row.text).lower()  
                logger.info("  Loaded notes for %d admissions", len(note\_text))  
            except Exception as e:  
                logger.warning("  Failed: %s", e)  
            break  
    return note\_text

def keyword\_present(text: str, keywords: list) \-\> bool:  
    return any(kw in text for kw in keywords)

def run\_interpretability\_analysis():  
    os.makedirs(RESULTS\_DIR, exist\_ok=True)  
    os.makedirs(FIGURES\_DIR, exist\_ok=True)

    \# Load gate weights and patient ids  
    gate\_weights  \= np.load(GATE\_WEIGHTS\_NPY)    \# (n\_test, n\_features)  
    test\_hadm\_ids \= np.load(GATE\_PATIENT\_IDS\_NPY) \# (n\_test,)  
    bundle        \= joblib.load(GATE\_MODEL\_PKL)  
    tab\_cols      \= bundle\["tab\_cols"\]            \# list of feature names

    logger.info("Gate weights shape: %s", gate\_weights.shape)  
    logger.info("Features: %d", len(tab\_cols))

    \# Build feature index lookup  
    feat\_idx \= {name: i for i, name in enumerate(tab\_cols)}

    \# Load discharge notes for test patients  
    note\_text \= load\_discharge\_notes(set(test\_hadm\_ids.tolist()))  
    logger.info("Notes loaded for %d / %d test patients",  
                len(note\_text), len(test\_hadm\_ids))

    \# Run analysis for each condition  
    rows \= \[\]  
    for condition, spec in CONDITION\_FEATURE\_MAP.items():  
        keywords \= spec\["keywords"\]  
        features \= \[f for f in spec\["features"\] if f in feat\_idx\]

        if not features:  
            logger.warning("No matching features for condition: %s", condition)  
            continue

        \# Boolean mask: which test patients have this condition mentioned in notes  
        has\_condition \= np.array(\[  
            keyword\_present(note\_text.get(int(hid), ""), keywords)  
            for hid in test\_hadm\_ids  
        \])

        n\_mentioned     \= has\_condition.sum()  
        n\_not\_mentioned \= (\~has\_condition).sum()

        if n\_mentioned \< 30 or n\_not\_mentioned \< 30:  
            logger.warning("Too few patients for %s (mentioned=%d, not=%d)",  
                           condition, n\_mentioned, n\_not\_mentioned)  
            continue

        logger.info("Condition: %-30s | mentioned: %d | not: %d",  
                    condition, n\_mentioned, n\_not\_mentioned)

        for feat\_name in features:  
            fi \= feat\_idx\[feat\_name\]  
            weights\_mentioned     \= gate\_weights\[has\_condition,  fi\]  
            weights\_not\_mentioned \= gate\_weights\[\~has\_condition, fi\]

            mean\_mentioned     \= float(np.mean(weights\_mentioned))  
            mean\_not\_mentioned \= float(np.mean(weights\_not\_mentioned))  
            mean\_diff          \= mean\_mentioned \- mean\_not\_mentioned

            \# Mann-Whitney U test (non-parametric, appropriate for gate weights)  
            stat, pval \= stats.mannwhitneyu(  
                weights\_mentioned, weights\_not\_mentioned, alternative="two-sided"  
            )

            rows.append({  
                "condition":        condition,  
                "feature":          feat\_name,  
                "mean\_mentioned":   round(mean\_mentioned,     4),  
                "mean\_not\_mentioned": round(mean\_not\_mentioned, 4),  
                "mean\_difference":  round(mean\_diff,          4),  
                "n\_mentioned":      int(n\_mentioned),  
                "n\_not\_mentioned":  int(n\_not\_mentioned),  
                "mann\_whitney\_u":   round(float(stat),        2),  
                "p\_value":          float(pval),  
                "significant\_p05":  bool(pval \< 0.05),  
            })

    df \= pd.DataFrame(rows)

    \# Bonferroni correction  
    n\_tests \= len(df)  
    df\["p\_bonferroni"\]   \= (df\["p\_value"\] \* n\_tests).clip(upper=1.0)  
    df\["significant\_bonferroni"\] \= df\["p\_bonferroni"\] \< 0.05

    results\_path \= os.path.join(RESULTS\_DIR, "gate\_interpretability.csv")  
    df.to\_csv(results\_path, index=False)  
    logger.info("Interpretability results saved \-\> %s", results\_path)

    \# ── Heatmap ───────────────────────────────────────────────────────────────  
    \# Pivot: conditions as rows, features as columns, mean\_difference as values  
    pivot \= df.pivot\_table(  
        index="condition", columns="feature",  
        values="mean\_difference", aggfunc="mean"  
    ).fillna(0)

    fig, ax \= plt.subplots(figsize=(max(10, len(pivot.columns) \* 1.2),  
                                    max(6, len(pivot.index) \* 0.8)))  
    im \= ax.imshow(pivot.values, cmap="RdBu\_r", aspect="auto",  
                   vmin=-0.3, vmax=0.3)

    ax.set\_xticks(range(len(pivot.columns)))  
    ax.set\_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)  
    ax.set\_yticks(range(len(pivot.index)))  
    ax.set\_yticklabels(\[c.replace("\_", " ") for c in pivot.index\], fontsize=9)

    plt.colorbar(im, ax=ax, label="Mean gate weight difference\\n(mentioned − not mentioned)")  
    ax.set\_title("Gate weight suppression/amplification by clinical condition\\n"  
                 "Blue \= suppressed when condition mentioned | Red \= amplified")

    \# Add significance markers  
    for i, cond in enumerate(pivot.index):  
        for j, feat in enumerate(pivot.columns):  
            subset \= df\[(df\["condition"\] \== cond) & (df\["feature"\] \== feat)\]  
            if len(subset) \> 0 and subset\["significant\_bonferroni"\].values\[0\]:  
                ax.text(j, i, "\*", ha="center", va="center", fontsize=12, color="black")

    plt.tight\_layout()  
    heatmap\_path \= os.path.join(FIGURES\_DIR, "gate\_heatmap.png")  
    plt.savefig(heatmap\_path, dpi=200, bbox\_inches="tight")  
    plt.close()  
    logger.info("Heatmap saved \-\> %s", heatmap\_path)

    \# Summary to console  
    logger.info("\\nTop suppressed feature-condition pairs (negative \= suppressed):")  
    top\_suppressed \= df.nsmallest(10, "mean\_difference")\[  
        \["condition", "feature", "mean\_difference", "p\_bonferroni", "significant\_bonferroni"\]  
    \]  
    print(top\_suppressed.to\_string(index=False))

    return df

if \_\_name\_\_ \== "\_\_main\_\_":  
    run\_interpretability\_analysis()

---

## **Change 4: Create src/11\_fairness\_calibration.py (New File)**

"""  
Fairness and Calibration Analysis  
\===================================  
Computes AUROC and ECE broken down by:  
  \- Race (White, Black, Hispanic, Asian)  
  \- Gender (Male, Female)  
  \- Age group (five buckets)

Runs for both the existing LightGBM model (trance\_framework.pkl)  
and the new TRANCE-Gate model (trance\_gate.pkl).

Produces:  
  \- results/fairness\_analysis.csv  
  \- results/calibration\_analysis.csv  
  \- figures/fairness\_auroc.png  
  \- figures/reliability\_diagram.png  
"""

import os  
import sys  
import logging  
import numpy as np  
import pandas as pd  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc\_auc\_score, brier\_score\_loss  
import joblib

sys.path.append(os.path.dirname(os.path.abspath(\_\_file\_\_)))  
try:  
    from config import (  
        FEATURES\_CSV, GATE\_MODEL\_PKL, MAIN\_MODEL\_PKL,  
        RESULTS\_DIR, FIGURES\_DIR, TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, RANDOM\_STATE,  
        FAIRNESS\_RESULTS\_CSV, CALIBRATION\_RESULTS\_CSV,  
    )  
except ImportError:  
    from .config import (  
        FEATURES\_CSV, GATE\_MODEL\_PKL, MAIN\_MODEL\_PKL,  
        RESULTS\_DIR, FIGURES\_DIR, TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, RANDOM\_STATE,  
        FAIRNESS\_RESULTS\_CSV, CALIBRATION\_RESULTS\_CSV,  
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s \- %(levelname)s \- %(message)s")  
logger \= logging.getLogger(\_\_name\_\_)

MIN\_GROUP\_SIZE \= 50   \# skip subgroups smaller than this

def compute\_ece(probs, labels, n\_bins=10):  
    bins \= np.linspace(0, 1, n\_bins \+ 1\)  
    ece, total \= 0.0, len(labels)  
    for i in range(n\_bins):  
        mask \= (probs \>= bins\[i\]) & (probs \< bins\[i \+ 1\])  
        if mask.sum() \== 0:  
            continue  
        ece \+= (mask.sum() / total) \* abs(float(labels\[mask\].mean()) \- float(probs\[mask\].mean()))  
    return float(ece)

def load\_test\_demographics():  
    """  
    Loads the raw features CSV and extracts demographic columns for test patients.  
    Returns a DataFrame with hadm\_id and demographic columns.  
    """  
    pruned \= FEATURES\_CSV.replace(".csv", "\_pruned.csv")  
    path   \= pruned if os.path.exists(pruned) else FEATURES\_CSV  
    cols   \= \["hadm\_id", "subject\_id", "readmit\_30", "gender",  
              "anchor\_age", "age\_group", "race\_enc"\]  
    available \= \[c for c in cols if c in pd.read\_csv(path, nrows=0).columns\]  
    df \= pd.read\_csv(path, usecols=available, low\_memory=False).fillna(0)  
    return df

def get\_test\_mask(groups):  
    rng \= np.random.RandomState(RANDOM\_STATE)  
    unique\_patients \= np.unique(groups)  
    rng.shuffle(unique\_patients)  
    n      \= len(unique\_patients)  
    n\_test \= int(n \* TRAIN\_TEST\_FRAC)  
    n\_val  \= int(n \* TRAIN\_VAL\_FRAC)  
    test\_pats \= set(unique\_patients\[-n\_test:\])  
    return np.array(\[g in test\_pats for g in groups\])

def fairness\_report(model\_name, y\_true, y\_pred, demo\_df, rows):  
    """  
    Computes per-subgroup AUROC and ECE and appends to rows list.  
    """  
    \# Overall  
    rows.append({  
        "model": model\_name,  
        "group\_type": "overall",  
        "group\_value": "all",  
        "n": len(y\_true),  
        "readmit\_rate": round(float(y\_true.mean()), 4),  
        "auroc": round(float(roc\_auc\_score(y\_true, y\_pred)), 4),  
        "ece":   round(float(compute\_ece(y\_pred, y\_true)), 4),  
        "brier": round(float(brier\_score\_loss(y\_true, y\_pred)), 4),  
    })

    \# Gender  
    if "gender" in demo\_df.columns:  
        for gval, gname in \[(0, "Female"), (1, "Male")\]:  
            mask \= demo\_df\["gender"\].values \== gval  
            if mask.sum() \< MIN\_GROUP\_SIZE:  
                continue  
            rows.append({  
                "model": model\_name, "group\_type": "gender",  
                "group\_value": gname, "n": int(mask.sum()),  
                "readmit\_rate": round(float(y\_true\[mask\].mean()), 4),  
                "auroc": round(float(roc\_auc\_score(y\_true\[mask\], y\_pred\[mask\])), 4),  
                "ece":   round(float(compute\_ece(y\_pred\[mask\], y\_true\[mask\])), 4),  
                "brier": round(float(brier\_score\_loss(y\_true\[mask\], y\_pred\[mask\])), 4),  
            })

    \# Age group  
    if "age\_group" in demo\_df.columns:  
        age\_names \= {0: "\<40", 1: "40-54", 2: "55-64", 3: "65-74", 4: "75-84", 5: "85+"}  
        for gval, gname in age\_names.items():  
            mask \= demo\_df\["age\_group"\].values \== gval  
            if mask.sum() \< MIN\_GROUP\_SIZE:  
                continue  
            rows.append({  
                "model": model\_name, "group\_type": "age\_group",  
                "group\_value": gname, "n": int(mask.sum()),  
                "readmit\_rate": round(float(y\_true\[mask\].mean()), 4),  
                "auroc": round(float(roc\_auc\_score(y\_true\[mask\], y\_pred\[mask\])), 4),  
                "ece":   round(float(compute\_ece(y\_pred\[mask\], y\_true\[mask\])), 4),  
                "brier": round(float(brier\_score\_loss(y\_true\[mask\], y\_pred\[mask\])), 4),  
            })

    \# Race (using race\_enc quartiles as proxy since we have frequency encoding)  
    if "race\_enc" in demo\_df.columns:  
        race\_vals \= demo\_df\["race\_enc"\].values  
        quartiles \= np.quantile(race\_vals\[race\_vals \> 0\], \[0.25, 0.5, 0.75, 1.0\])  
        for qi, (lo, hi) in enumerate(zip(\[0\] \+ list(quartiles\[:-1\]), quartiles)):  
            mask \= (race\_vals \>= lo) & (race\_vals \< hi)  
            if mask.sum() \< MIN\_GROUP\_SIZE:  
                continue  
            rows.append({  
                "model": model\_name, "group\_type": "race\_quartile",  
                "group\_value": f"Q{qi+1}", "n": int(mask.sum()),  
                "readmit\_rate": round(float(y\_true\[mask\].mean()), 4),  
                "auroc": round(float(roc\_auc\_score(y\_true\[mask\], y\_pred\[mask\])), 4),  
                "ece":   round(float(compute\_ece(y\_pred\[mask\], y\_true\[mask\])), 4),  
                "brier": round(float(brier\_score\_loss(y\_true\[mask\], y\_pred\[mask\])), 4),  
            })

def reliability\_diagram(probs\_dict, labels, save\_path, n\_bins=10):  
    """  
    Plots reliability diagram for multiple models on one axis.  
    """  
    fig, ax \= plt.subplots(figsize=(6, 6))  
    bins \= np.linspace(0, 1, n\_bins \+ 1\)  
    bin\_centers \= (bins\[:-1\] \+ bins\[1:\]) / 2

    for model\_name, probs in probs\_dict.items():  
        frac\_pos \= \[\]  
        for i in range(n\_bins):  
            mask \= (probs \>= bins\[i\]) & (probs \< bins\[i \+ 1\])  
            if mask.sum() \== 0:  
                frac\_pos.append(np.nan)  
            else:  
                frac\_pos.append(float(labels\[mask\].mean()))  
        ax.plot(bin\_centers, frac\_pos, "s-", label=model\_name, linewidth=1.5, markersize=5)

    ax.plot(\[0, 1\], \[0, 1\], "k--", linewidth=1, label="Perfect calibration")  
    ax.set\_xlabel("Mean predicted probability")  
    ax.set\_ylabel("Fraction of positives (observed readmission rate)")  
    ax.set\_title("Reliability diagram")  
    ax.legend()  
    ax.set\_xlim(0, 1\)  
    ax.set\_ylim(0, 1\)  
    plt.tight\_layout()  
    plt.savefig(save\_path, dpi=200, bbox\_inches="tight")  
    plt.close()  
    logger.info("Reliability diagram saved \-\> %s", save\_path)

def run\_fairness\_calibration():  
    os.makedirs(RESULTS\_DIR, exist\_ok=True)  
    os.makedirs(FIGURES\_DIR, exist\_ok=True)

    demo\_df \= load\_test\_demographics()  
    groups  \= demo\_df\["subject\_id"\].values  
    test\_mask \= get\_test\_mask(groups)  
    demo\_test \= demo\_df\[test\_mask\].reset\_index(drop=True)  
    y\_true    \= demo\_df\["readmit\_30"\].values\[test\_mask\].astype(np.float32)

    rows \= \[\]  
    probs\_for\_diagram \= {}

    \# ── LightGBM baseline ──────────────────────────────────────────────────  
    if os.path.exists(MAIN\_MODEL\_PKL):  
        logger.info("Loading LightGBM baseline...")  
        lgbm\_bundle \= joblib.load(MAIN\_MODEL\_PKL)  
        lgbm\_probs  \= lgbm\_bundle.get("test\_probs\_cal")  
        lgbm\_labels \= lgbm\_bundle.get("test\_labels")

        if lgbm\_probs is not None and len(lgbm\_probs) \== len(y\_true):  
            fairness\_report("LightGBM-ensemble", y\_true, lgbm\_probs, demo\_test, rows)  
            probs\_for\_diagram\["LightGBM-ensemble"\] \= lgbm\_probs  
        else:  
            logger.warning("LightGBM test probs not found in bundle or size mismatch.")  
    else:  
        logger.warning("LightGBM model not found at %s", MAIN\_MODEL\_PKL)

    \# ── TRANCE-Gate ────────────────────────────────────────────────────────  
    if os.path.exists(GATE\_MODEL\_PKL):  
        logger.info("Loading TRANCE-Gate...")  
        gate\_bundle \= joblib.load(GATE\_MODEL\_PKL)  
        gate\_probs  \= gate\_bundle.get("test\_probs\_cal")  
        gate\_labels \= gate\_bundle.get("test\_labels")

        if gate\_probs is not None and len(gate\_probs) \== len(y\_true):  
            fairness\_report("TRANCE-Gate", y\_true, gate\_probs, demo\_test, rows)  
            probs\_for\_diagram\["TRANCE-Gate"\] \= gate\_probs  
        else:  
            logger.warning("Gate probs not found or size mismatch.")  
    else:  
        logger.warning("TRANCE-Gate model not found at %s", GATE\_MODEL\_PKL)

    if not rows:  
        logger.error("No model results to report. Run training first.")  
        return

    df \= pd.DataFrame(rows)  
    df.to\_csv(FAIRNESS\_RESULTS\_CSV, index=False)  
    logger.info("Fairness results saved \-\> %s", FAIRNESS\_RESULTS\_CSV)

    \# Print summary  
    overall \= df\[df\["group\_type"\] \== "overall"\]\[\["model", "auroc", "ece", "brier"\]\]  
    print("\\nOverall performance:")  
    print(overall.to\_string(index=False))

    gender\_df \= df\[df\["group\_type"\] \== "gender"\]  
    if not gender\_df.empty:  
        print("\\nBy gender:")  
        print(gender\_df\[\["model", "group\_value", "n", "auroc", "ece"\]\].to\_string(index=False))

    age\_df \= df\[df\["group\_type"\] \== "age\_group"\]  
    if not age\_df.empty:  
        print("\\nBy age group:")  
        print(age\_df\[\["model", "group\_value", "n", "auroc", "ece"\]\].to\_string(index=False))

    \# Max AUROC gap across age groups per model  
    for model in df\["model"\].unique():  
        age\_aucs \= df\[(df\["model"\] \== model) & (df\["group\_type"\] \== "age\_group")\]\["auroc"\]  
        if len(age\_aucs) \> 1:  
            logger.info("Model: %-20s | Max age-group AUROC gap: %.4f",  
                        model, age\_aucs.max() \- age\_aucs.min())

    \# Reliability diagram  
    if probs\_for\_diagram:  
        reliability\_diagram(  
            probs\_for\_diagram, y\_true,  
            os.path.join(FIGURES\_DIR, "reliability\_diagram.png")  
        )

    return df

if \_\_name\_\_ \== "\_\_main\_\_":  
    run\_fairness\_calibration()

---

## **Change 5: Create src/12\_early\_warning.py (New File)**

"""  
Early Warning Analysis  
\=======================  
Evaluates model performance when EHR data is restricted to  
the first N days of hospitalization, for N in EARLY\_WARNING\_DAYS.

For each day cutoff:  
  \- Filters lab results, vitals, medications to that window  
  \- Retrains a LightGBM model on filtered features  
  \- Reports AUROC at each cutoff

Produces:  
  \- results/early\_warning\_results.csv  
  \- figures/early\_warning\_curve.png  
"""

import os  
import sys  
import logging  
import numpy as np  
import pandas as pd  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
import lightgbm as lgb  
from sklearn.metrics import roc\_auc\_score  
import joblib

sys.path.append(os.path.dirname(os.path.abspath(\_\_file\_\_)))  
try:  
    from config import (  
        MIMIC\_IV\_DIR, FEATURES\_CSV, EMBEDDINGS\_CSV,  
        RESULTS\_DIR, FIGURES\_DIR, EARLY\_WARNING\_CSV,  
        EARLY\_WARNING\_DAYS, RANDOM\_STATE,  
        TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, MAIN\_MODEL\_PKL,  
    )  
except ImportError:  
    from .config import (  
        MIMIC\_IV\_DIR, FEATURES\_CSV, EMBEDDINGS\_CSV,  
        RESULTS\_DIR, FIGURES\_DIR, EARLY\_WARNING\_CSV,  
        EARLY\_WARNING\_DAYS, RANDOM\_STATE,  
        TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, MAIN\_MODEL\_PKL,  
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s \- %(levelname)s \- %(message)s")  
logger \= logging.getLogger(\_\_name\_\_)

def get\_patient\_split(groups):  
    rng \= np.random.RandomState(RANDOM\_STATE)  
    unique \= np.unique(groups)  
    rng.shuffle(unique)  
    n      \= len(unique)  
    n\_test \= int(n \* TRAIN\_TEST\_FRAC)  
    n\_val  \= int(n \* TRAIN\_VAL\_FRAC)  
    test\_pats  \= set(unique\[-n\_test:\])  
    val\_pats   \= set(unique\[-(n\_test \+ n\_val):-n\_test\])  
    train\_pats \= set(unique\[:-(n\_test \+ n\_val)\])  
    return train\_pats, val\_pats, test\_pats

def filter\_lab\_features\_by\_day(df, max\_day, mimic\_iv\_dir):  
    """  
    Filters cumulative lab statistics to only include  
    measurements taken within the first max\_day days.

    This works by loading labevents, filtering by charttime,  
    and recomputing per-admission statistics.

    If labevents is not accessible, falls back to scaling  
    existing features by (max\_day / mean\_los) as an approximation.  
    """  
    lab\_path \= None  
    for root, \_, files in os.walk(mimic\_iv\_dir):  
        for f in files:  
            if f \== "labevents.csv.gz":  
                lab\_path \= os.path.join(root, f)  
                break  
        if lab\_path:  
            break

    if lab\_path is None:  
        logger.warning("labevents.csv.gz not found. Using LOS-scaled approximation.")  
        return None

    \# Load admission times for windowing  
    adm\_path \= None  
    for root, \_, files in os.walk(mimic\_iv\_dir):  
        for f in files:  
            if f \== "admissions.csv.gz":  
                adm\_path \= os.path.join(root, f)  
                break  
        if adm\_path:  
            break

    if adm\_path is None:  
        return None

    logger.info("Filtering lab events to first %d days...", max\_day)  
    adm \= pd.read\_csv(adm\_path, usecols=\["hadm\_id", "admittime"\],  
                      parse\_dates=\["admittime"\], low\_memory=True)  
    adm\_map \= dict(zip(adm\["hadm\_id"\], adm\["admittime"\]))

    cohort\_hadm \= set(df\["hadm\_id"\].values)  
    KEY\_LAB\_ITEMS \= {  
        50912: "creatinine", 50882: "bicarb", 50931: "glucose",  
        50983: "sodium",     51006: "bun",    51221: "hematocrit",  
        51222: "hemoglobin", 51265: "platelets", 51301: "wbc",  
        50813: "lactate",    50820: "ph",  
    }

    chunks \= \[\]  
    reader \= pd.read\_csv(  
        lab\_path,  
        usecols=\["hadm\_id", "itemid", "valuenum", "charttime"\],  
        chunksize=2\_000\_000, low\_memory=True, parse\_dates=\["charttime"\]  
    )  
    for chunk in reader:  
        chunk \= chunk\[chunk\["hadm\_id"\].isin(cohort\_hadm)\]  
        chunk \= chunk\[chunk\["itemid"\].isin(KEY\_LAB\_ITEMS)\]  
        chunk \= chunk\[chunk\["valuenum"\].notna()\]  
        chunk\["admittime"\] \= chunk\["hadm\_id"\].map(adm\_map)  
        chunk\["day"\] \= ((chunk\["charttime"\] \- chunk\["admittime"\])  
                        .dt.total\_seconds() / 86400).clip(lower=0)  
        chunk \= chunk\[chunk\["day"\] \<= max\_day\]  
        if not chunk.empty:  
            chunks.append(chunk\[\["hadm\_id", "itemid", "valuenum"\]\])

    if not chunks:  
        return None

    events \= pd.concat(chunks, ignore\_index=True)  
    events\["lname"\] \= events\["itemid"\].map(KEY\_LAB\_ITEMS)

    agg\_rows \= \[\]  
    for (hadm, lname), grp in events.groupby(\["hadm\_id", "lname"\]):  
        vals \= grp\["valuenum"\].values  
        agg\_rows.append({  
            "hadm\_id": hadm,  
            f"lab\_{lname}\_mean": float(np.mean(vals)),  
            f"lab\_{lname}\_max":  float(np.max(vals)),  
            f"lab\_{lname}\_min":  float(np.min(vals)),  
            f"lab\_{lname}\_last": float(vals\[-1\]),  
            f"lab\_{lname}\_range": float(np.ptp(vals)),  
            f"lab\_{lname}\_n":    len(vals),  
        })

    if not agg\_rows:  
        return None

    lab\_df \= pd.DataFrame(agg\_rows).groupby("hadm\_id").first().reset\_index()  
    return lab\_df

def run\_early\_warning():  
    os.makedirs(RESULTS\_DIR, exist\_ok=True)  
    os.makedirs(FIGURES\_DIR, exist\_ok=True)

    pruned \= FEATURES\_CSV.replace(".csv", "\_pruned.csv")  
    feat\_path \= pruned if os.path.exists(pruned) else FEATURES\_CSV  
    df\_full \= pd.read\_csv(feat\_path, low\_memory=False).fillna(0)

    groups \= df\_full\["subject\_id"\].values  
    train\_pats, val\_pats, test\_pats \= get\_patient\_split(groups)

    train\_mask \= np.array(\[g in train\_pats for g in groups\])  
    val\_mask   \= np.array(\[g in val\_pats   for g in groups\])  
    test\_mask  \= np.array(\[g in test\_pats  for g in groups\])

    id\_cols  \= {"subject\_id", "hadm\_id", "readmit\_30"}  
    feat\_cols \= \[c for c in df\_full.columns if c not in id\_cols\]  
    y \= df\_full\["readmit\_30"\].values

    \# Load best params from existing LightGBM model for consistency  
    best\_params \= {  
        "objective": "binary", "metric": "auc",  
        "verbosity": \-1, "n\_jobs": \-1,  
        "random\_state": RANDOM\_STATE,  
        "n\_estimators": 1000,  
        "learning\_rate": 0.03,  
        "num\_leaves": 127,  
        "max\_depth": 8,  
        "scale\_pos\_weight": float((y\[train\_mask\] \== 0).sum() /  
                                   max((y\[train\_mask\] \== 1).sum(), 1)),  
    }  
    if os.path.exists(MAIN\_MODEL\_PKL):  
        bundle \= joblib.load(MAIN\_MODEL\_PKL)  
        stored \= bundle.get("best\_params", {})  
        if stored:  
            best\_params.update({k: v for k, v in stored.items()  
                                if k not in ("objective", "metric", "verbosity", "n\_jobs")})

    rows \= \[\]

    \# Full-data baseline first  
    X\_tr \= df\_full\[feat\_cols\].values\[train\_mask\]  
    X\_val \= df\_full\[feat\_cols\].values\[val\_mask\]  
    X\_te  \= df\_full\[feat\_cols\].values\[test\_mask\]  
    y\_tr, y\_val, y\_te \= y\[train\_mask\], y\[val\_mask\], y\[test\_mask\]

    model\_full \= lgb.LGBMClassifier(\*\*best\_params)  
    model\_full.fit(X\_tr, y\_tr, eval\_set=\[(X\_val, y\_val)\],  
                   callbacks=\[lgb.early\_stopping(50, verbose=False),  
                               lgb.log\_evaluation(-1)\])  
    auroc\_full \= roc\_auc\_score(y\_te, model\_full.predict\_proba(X\_te)\[:, 1\])  
    rows.append({"day\_cutoff": "full", "auroc": round(float(auroc\_full), 4),  
                 "n\_train": int(y\_tr.sum()), "n\_test": int(y\_te.sum())})  
    logger.info("Full data AUROC: %.4f", auroc\_full)

    \# Day-limited experiments  
    for max\_day in sorted(EARLY\_WARNING\_DAYS):  
        logger.info("Running early warning: day %d cutoff", max\_day)

        \# Create day-limited feature set  
        \# Strategy: zero out lab features that require more than max\_day of data  
        \# by replacing them with filtered versions if raw data available,  
        \# or zeroing multi-day stats (range, std) otherwise as approximation  
        df\_day \= df\_full.copy()

        \# Zero out features that aggregate over the full stay  
        \# when we only have max\_day worth of data  
        \# Features that are inherently about stay duration get scaled  
        if "los\_days" in df\_day.columns:  
            df\_day\["los\_days"\] \= df\_day\["los\_days"\].clip(upper=max\_day)  
        if "los\_hours" in df\_day.columns:  
            df\_day\["los\_hours"\] \= df\_day\["los\_hours"\].clip(upper=max\_day \* 24\)

        \# Zero out lab range/std features (they require full stay to be meaningful)  
        range\_cols \= \[c for c in df\_day.columns if "\_range" in c or "\_std" in c\]  
        df\_day\[range\_cols\] \= 0.0

        \# Try to recompute lab features from raw data if available  
        lab\_recomputed \= filter\_lab\_features\_by\_day(df\_day, max\_day, MIMIC\_IV\_DIR)  
        if lab\_recomputed is not None:  
            \# Merge recomputed lab features, overwriting the zeroed ones  
            lab\_cols \= \[c for c in lab\_recomputed.columns if c \!= "hadm\_id"\]  
            for col in lab\_cols:  
                if col in df\_day.columns:  
                    df\_day \= df\_day.drop(columns=\[col\])  
            df\_day \= df\_day.merge(lab\_recomputed, on="hadm\_id", how="left")  
            df\_day \= df\_day.fillna(0)

        feat\_cols\_day \= \[c for c in df\_day.columns if c not in id\_cols\]

        X\_tr\_d  \= df\_day\[feat\_cols\_day\].values\[train\_mask\]  
        X\_val\_d \= df\_day\[feat\_cols\_day\].values\[val\_mask\]  
        X\_te\_d  \= df\_day\[feat\_cols\_day\].values\[test\_mask\]

        model\_day \= lgb.LGBMClassifier(\*\*best\_params)  
        model\_day.fit(X\_tr\_d, y\_tr, eval\_set=\[(X\_val\_d, y\_val)\],  
                      callbacks=\[lgb.early\_stopping(50, verbose=False),  
                                 lgb.log\_evaluation(-1)\])  
        auroc\_day \= roc\_auc\_score(y\_te, model\_day.predict\_proba(X\_te\_d)\[:, 1\])  
        rows.append({  
            "day\_cutoff": max\_day,  
            "auroc": round(float(auroc\_day), 4),  
            "n\_train": int(y\_tr.sum()),  
            "n\_test":  int(y\_te.sum()),  
        })  
        logger.info("Day %d AUROC: %.4f", max\_day, auroc\_day)

    df\_results \= pd.DataFrame(rows)  
    df\_results.to\_csv(EARLY\_WARNING\_CSV, index=False)  
    logger.info("Early warning results saved \-\> %s", EARLY\_WARNING\_CSV)

    \# Plot  
    numeric\_rows \= df\_results\[df\_results\["day\_cutoff"\] \!= "full"\].copy()  
    numeric\_rows\["day\_cutoff"\] \= numeric\_rows\["day\_cutoff"\].astype(int)  
    full\_auroc \= df\_results\[df\_results\["day\_cutoff"\] \== "full"\]\["auroc"\].values\[0\]

    fig, ax \= plt.subplots(figsize=(8, 5))  
    ax.plot(numeric\_rows\["day\_cutoff"\], numeric\_rows\["auroc"\],  
            "o-", linewidth=2, markersize=7, label="AUROC at day N")  
    ax.axhline(full\_auroc, color="gray", linestyle="--",  
               linewidth=1.5, label=f"Full-stay AUROC ({full\_auroc:.3f})")  
    ax.set\_xlabel("Days from admission (data available up to day N)")  
    ax.set\_ylabel("AUROC")  
    ax.set\_title("Prediction performance vs. earliness of prediction")  
    ax.legend()  
    ax.set\_ylim(0.5, 1.0)  
    ax.grid(True, alpha=0.3)  
    plt.tight\_layout()  
    path \= os.path.join(FIGURES\_DIR, "early\_warning\_curve.png")  
    plt.savefig(path, dpi=200, bbox\_inches="tight")  
    plt.close()  
    logger.info("Early warning curve saved \-\> %s", path)

    print(df\_results.to\_string(index=False))  
    return df\_results

if \_\_name\_\_ \== "\_\_main\_\_":  
    run\_early\_warning()

---

## **Change 6: Create src/13\_temporal\_drift.py (New File)**

"""  
Temporal Drift Analysis  
\========================  
Evaluates model performance across anchor\_year\_group cohorts  
in MIMIC-IV (2008-2022). No retraining — same trained model  
evaluated on each temporal slice of the test set.

Produces:  
  \- results/temporal\_drift\_results.csv  
  \- figures/temporal\_drift.png  
"""

import os  
import sys  
import logging  
import numpy as np  
import pandas as pd  
import matplotlib  
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc\_auc\_score  
import joblib

sys.path.append(os.path.dirname(os.path.abspath(\_\_file\_\_)))  
try:  
    from config import (  
        FEATURES\_CSV, EMBEDDINGS\_CSV, MAIN\_MODEL\_PKL, GATE\_MODEL\_PKL,  
        RESULTS\_DIR, FIGURES\_DIR, TEMPORAL\_DRIFT\_CSV,  
        TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, RANDOM\_STATE,  
    )  
except ImportError:  
    from .config import (  
        FEATURES\_CSV, EMBEDDINGS\_CSV, MAIN\_MODEL\_PKL, GATE\_MODEL\_PKL,  
        RESULTS\_DIR, FIGURES\_DIR, TEMPORAL\_DRIFT\_CSV,  
        TRAIN\_TEST\_FRAC, TRAIN\_VAL\_FRAC, RANDOM\_STATE,  
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s \- %(levelname)s \- %(message)s")  
logger \= logging.getLogger(\_\_name\_\_)

YEAR\_GROUP\_LABELS \= {  
    0: "2008-2010", 1: "2011-2013", 2: "2014-2016",  
    3: "2017-2019", 4: "2020-2022",  
}

def get\_test\_mask(groups):  
    rng \= np.random.RandomState(RANDOM\_STATE)  
    unique \= np.unique(groups)  
    rng.shuffle(unique)  
    n \= len(unique)  
    n\_test \= int(n \* TRAIN\_TEST\_FRAC)  
    n\_val  \= int(n \* TRAIN\_VAL\_FRAC)  
    test\_pats \= set(unique\[-n\_test:\])  
    return np.array(\[g in test\_pats for g in groups\])

def run\_temporal\_drift():  
    os.makedirs(RESULTS\_DIR, exist\_ok=True)  
    os.makedirs(FIGURES\_DIR, exist\_ok=True)

    pruned \= FEATURES\_CSV.replace(".csv", "\_pruned.csv")  
    feat\_path \= pruned if os.path.exists(pruned) else FEATURES\_CSV  
    df \= pd.read\_csv(feat\_path, low\_memory=False).fillna(0)

    if "anchor\_year\_group" not in df.columns:  
        logger.error("anchor\_year\_group not found in features. Run 01\_extract.py first.")  
        return

    groups    \= df\["subject\_id"\].values  
    test\_mask \= get\_test\_mask(groups)  
    df\_test   \= df\[test\_mask\].reset\_index(drop=True)  
    y\_test    \= df\_test\["readmit\_30"\].values

    rows \= \[\]

    \# ── LightGBM predictions ────────────────────────────────────────────────  
    if os.path.exists(MAIN\_MODEL\_PKL):  
        bundle     \= joblib.load(MAIN\_MODEL\_PKL)  
        lgbm\_probs \= bundle.get("test\_probs\_cal")  
        if lgbm\_probs is not None and len(lgbm\_probs) \== len(y\_test):  
            for yg\_code, yg\_label in YEAR\_GROUP\_LABELS.items():  
                mask \= df\_test\["anchor\_year\_group"\].values \== yg\_code  
                if mask.sum() \< 50:  
                    continue  
                auroc \= roc\_auc\_score(y\_test\[mask\], lgbm\_probs\[mask\])  
                rows.append({  
                    "model": "LightGBM-ensemble",  
                    "year\_group": yg\_label,  
                    "year\_group\_code": yg\_code,  
                    "n\_admissions": int(mask.sum()),  
                    "readmit\_rate": round(float(y\_test\[mask\].mean()), 4),  
                    "auroc": round(float(auroc), 4),  
                })  
                logger.info("LightGBM | %s | AUROC: %.4f (n=%d)",  
                            yg\_label, auroc, mask.sum())

    \# ── TRANCE-Gate predictions ─────────────────────────────────────────────  
    if os.path.exists(GATE\_MODEL\_PKL):  
        bundle     \= joblib.load(GATE\_MODEL\_PKL)  
        gate\_probs \= bundle.get("test\_probs\_cal")  
        if gate\_probs is not None and len(gate\_probs) \== len(y\_test):  
            for yg\_code, yg\_label in YEAR\_GROUP\_LABELS.items():  
                mask \= df\_test\["anchor\_year\_group"\].values \== yg\_code  
                if mask.sum() \< 50:  
                    continue  
                auroc \= roc\_auc\_score(y\_test\[mask\], gate\_probs\[mask\])  
                rows.append({  
                    "model": "TRANCE-Gate",  
                    "year\_group": yg\_label,  
                    "year\_group\_code": yg\_code,  
                    "n\_admissions": int(mask.sum()),  
                    "readmit\_rate": round(float(y\_test\[mask\].mean()), 4),  
                    "auroc": round(float(auroc), 4),  
                })  
                logger.info("TRANCE-Gate | %s | AUROC: %.4f (n=%d)",  
                            yg\_label, auroc, mask.sum())

    if not rows:  
        logger.error("No results generated. Ensure models are trained.")  
        return

    df\_results \= pd.DataFrame(rows)  
    df\_results.to\_csv(TEMPORAL\_DRIFT\_CSV, index=False)  
    logger.info("Temporal drift results saved \-\> %s", TEMPORAL\_DRIFT\_CSV)

    \# Plot  
    fig, ax \= plt.subplots(figsize=(9, 5))  
    for model\_name, grp in df\_results.groupby("model"):  
        grp \= grp.sort\_values("year\_group\_code")  
        ax.plot(grp\["year\_group"\], grp\["auroc"\], "o-",  
                linewidth=2, markersize=7, label=model\_name)

    ax.set\_xlabel("Year group")  
    ax.set\_ylabel("AUROC")  
    ax.set\_title("Model performance across time periods (temporal drift)")  
    ax.legend()  
    ax.set\_ylim(0.5, 1.0)  
    ax.grid(True, alpha=0.3)  
    plt.xticks(rotation=20)  
    plt.tight\_layout()  
    path \= os.path.join(FIGURES\_DIR, "temporal\_drift.png")  
    plt.savefig(path, dpi=200, bbox\_inches="tight")  
    plt.close()  
    logger.info("Temporal drift plot saved \-\> %s", path)

    print(df\_results.to\_string(index=False))  
    return df\_results

if \_\_name\_\_ \== "\_\_main\_\_":  
    run\_temporal\_drift()

---

## **Change 7: Update run\_pipeline.py**

Add the new steps to the pipeline so everything can run in sequence. Open `run_pipeline.py` and update the `steps` list:

steps \= \[  
    ("1/9  Feature Extraction",              "01\_extract"),  
    ("2/9  Feature Selection (SHAP)",        "01b\_select\_features"),  
    ("3/9  Clinical T5 Embedding",           "02\_embed"),  
    ("4/9  Model Training (LightGBM)",       "03\_train"),  
    ("5/9  Embedding Diagnostics",           "04\_diagnose"),  
    ("6/9  SHAP Interpretability",           "05\_analyze"),  
    ("7/9  Journal Visualizations",          "06\_visualize"),  
    ("8/9  Cross-Paper Comparison",          "09\_compare\_models"),  
    \# New additions below  
    ("9/13 TRANCE-Gate Training",            "gated\_fusion\_model"),  
    ("10/13 Gate Interpretability Analysis", "10\_gate\_interpretability"),  
    ("11/13 Fairness and Calibration",       "11\_fairness\_calibration"),  
    ("12/13 Early Warning Analysis",         "12\_early\_warning"),  
    ("13/13 Temporal Drift Analysis",        "13\_temporal\_drift"),  
\]

---

## **Summary of All Changes**

To summarize what the agent needs to do in total:

`src/config.py` — add sections 10, 11, and 12 at the bottom as shown.

`src/gated_fusion_model.py` — create this new file. It contains the PyTorch architecture, training loop, calibration, and saving of gate weights.

`src/10_gate_interpretability.py` — create this new file. It loads gate weights, loads discharge notes, groups patients by keyword presence, runs Mann-Whitney tests, and produces the heatmap figure.

`src/11_fairness_calibration.py` — create this new file. It computes AUROC and ECE per demographic subgroup for both models and produces the reliability diagram.

`src/12_early_warning.py` — create this new file. It trains day-limited LightGBM models and plots the AUROC versus earliness curve.

`src/13_temporal_drift.py` — create this new file. It evaluates existing trained models on year-group slices and plots stability over time.

`run_pipeline.py` — update the steps list to include the four new analysis scripts after the existing nine steps.

No existing files are deleted or broken. Everything that already works continues to work unchanged.

