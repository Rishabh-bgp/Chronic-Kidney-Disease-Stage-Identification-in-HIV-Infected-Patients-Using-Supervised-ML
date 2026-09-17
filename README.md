# Chronic Kidney Disease Stage Identification in HIV-Infected Patients Using Supervised Machine Learning: A Review

**Author:** Er. Rishabh Aryan  
**Affiliation:** M.Tech (Artificial Intelligence and Data Science), Department of Computer Science and Engineering, Indian Institute of Information Technology, Bhagalpur (Bihar)  
**Email:** rishabh.250201011@iiitbh.ac.in  

**Publication:** *International Journal of Versatile Research and Analysis* (IJVRA), Volume 4, Issue 4, April 2026  
**Paper ID:** IJVRA2604095  
**ISSN:** 2984-8903  
**DOI / URL:** [https://ijpub.org/ijvra/viewpaperforall.php?paper=IJVRA2604095](https://ijpub.org/ijvra/viewpaperforall.php?paper=IJVRA2604095)

---

## Abstract

Chronic Kidney Disease (CKD) constitutes a substantial global health burden that is markedly amplified among persons living with Human Immunodeficiency Virus (HIV). Viral-mediated glomerular injury, nephrotoxic antiretroviral therapy, and conventional cardiometabolic risk factors accelerate renal decline and frequently remain clinically silent until advanced stages. This work reviews supervised machine-learning approaches for automated identification of CKD and, where reported, eGFR-based stage assignment in HIV-associated cohorts.

Algorithms examined include Support Vector Machine (SVM), *k*-Nearest Neighbours (KNN), Decision Tree (DT), Random Forest (RF), AdaBoost, XGBoost, and Deep Neural Networks (DNN). Feature-selection strategies—particularly Recursive Feature Elimination (RFE)—and the Modification of Diet in Renal Disease (MDRD) eGFR equation are synthesised. Across the literature, DNN architectures attain the highest reported performance (up to 99% accuracy and 98% recall on the 158-instance HIV-CKD cohort of Darveshwala et al.). Classical ensembles remain competitive when computational resources constrain deep models. The review identifies small sample size, class imbalance, and heterogeneous validation protocols as principal limitations and recommends multi-modal imaging, explainable AI, and federated learning as future directions.

---

## 1. Motivation and Clinical Context

CKD is defined by a persistent reduction in glomerular filtration rate and/or markers of kidney damage. In HIV-positive patients the phenotype is compounded by:

- HIV-associated nephropathy (HIVAN), a collapsing form of focal segmental glomerulosclerosis;
- tubular toxicity of agents such as tenofovir disoproxil fumarate;
- coexistent hypertension and diabetes mellitus.

Because early and intermediate stages are largely asymptomatic, laboratory-driven classifiers that map routinely collected attributes onto binary CKD status and eGFR stages can reduce diagnostic latency and support timely modification of antiretroviral regimens.

---

## 2. Dataset

### 2.1 Primary literature cohort

The most comprehensive experimental study reviewed (Darveshwala et al., 2021) uses a UCI-derived HIV-CKD collection of **158 instances** and **27 attributes** (numerical laboratory values plus encoded demographic and comorbidity indicators). After binary classification, 44 patients were labelled CKD-positive; eGFR staging of those 44 cases showed heavy concentration in Stages 4–5 (10 and 25 patients, respectively).

### 2.2 Attributes (HIV-CKD schema)

| No. | Feature | Type | Description |
|-----|---------|------|-------------|
| 1 | Age | Numerical | Years |
| 2 | Gender | Categorical | Male / Female |
| 3 | Ethnicity | Numerical | Encoded group |
| 4 | Blood pressure | Numerical | Systolic, mmHg |
| 5–11 | Urinalysis panel | Mixed | Specific gravity, albumin, sugar, RBC, pus cells, clumps, bacteria |
| 12–20 | Serum / haematology | Numerical | Glucose, urea, creatinine, Na⁺, K⁺, Hb, PCV, WBC, RBC count |
| 21–26 | Comorbidities / symptoms | Binary / categorical | Hypertension, diabetes, CAD, appetite, pedal oedema, anaemia |
| 27 | Class | Categorical | CKD / not CKD |

### 2.3 Companion notebook data

The accompanying Colab notebook loads a CSV (`upload.csv`) whose schema matches the widely used **UCI Chronic Kidney Disease** repository (age, bp, sg, al, su, rbc, pc, pcc, ba, …, classification). That file is the conventional 400-record general-CKD table rather than the 158-record HIV-specific subset. Results reported in the notebook (perfect scores across all seven models) should therefore be interpreted with caution: they are not a reproduction of the 93–99% figures tabulated in the review for the HIV cohort.

---

## 3. Methodological Framework

### 3.1 Preprocessing

Typical pipeline used in the reviewed studies and in the notebook:

1. Mean imputation for numeric missing values; mode imputation for categorical fields.
2. One-hot or label encoding of nominal variables.
3. Standard scaling (or min–max normalisation) so that distance-based learners (KNN, SVM) are not dominated by high-range features such as white-cell count.
4. Optional wrapper selection: **Recursive Feature Elimination** reduced 27 attributes to 14 in the core HIV-CKD experiment.

### 3.2 Classifiers

| Algorithm | Role in the review | Notes |
|-----------|--------------------|-------|
| SVM | Linear / RBF margin classifier | 93% accuracy on 14 RFE features |
| KNN | Non-parametric neighbourhood vote | 97% |
| Decision Tree | Information-gain partitions | 97%; prone to overfit on *n* = 158 |
| Random Forest | Bagged trees | 95% |
| AdaBoost | Sequential re-weighting | 97% accuracy, 97% recall |
| XGBoost | Regularised gradient boosting | 97% |
| DNN | Fully connected network with dropout | **99% accuracy, 99% precision, 98% recall** on all 24 predictors |

The notebook implements the same seven families with `sklearn`, `xgboost`, and a Keras sequential network (64–32 ReLU units, dropout 0.3, sigmoid output, binary cross-entropy, 20 epochs).

### 3.3 eGFR staging (MDRD)

\[
\mathrm{eGFR} = 175 \times \left(\frac{\mathrm{Creatinine}}{88.4}\right)^{-1.154} \times (\mathrm{Age})^{-0.203} \times (0.742\ \mathrm{if\ female}) \times (1.210\ \mathrm{if\ Black\ ethnicity})
\]

| Stage | Description | eGFR (mL/min/1.73 m²) | HIV risk |
|-------|-------------|------------------------|----------|
| 1 | Damage with normal GFR | > 90 | Moderate |
| 2 | Mild reduction | 60–89 | Moderate |
| 3A | Mild–moderate | 45–59 | High |
| 3B | Moderate–severe | 30–44 | High |
| 4 | Severe | 15–29 | Very high |
| 5 | Kidney failure | < 15 | Critical |

---

## 4. Principal Findings (Literature Synthesis)

Table 5 of the review (Darveshwala et al. HIV-CKD split) is reproduced below.

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | Features |
|------------|--------------|---------------|------------|----------|
| SVM | 93 | 91 | 92 | 14 (RFE) |
| KNN | 97 | 95 | 96 | 14 (RFE) |
| Decision Tree | 97 | 96 | 96 | 14 (RFE) |
| Random Forest | 95 | 95 | 94 | 14 (RFE) |
| AdaBoost | 97 | 96 | 97 | 14 (RFE) |
| XGBoost | 97 | 95 | 96 | 14 (RFE) |
| **DNN** | **99** | **99** | **98** | **24 (all)** |

DNN superiority is attributed to hierarchical representation learning and implicit regularisation; ensembles remain the recommended fallback when GPU resources or sample size preclude deep models.

---

## 5. Limitations Identified in the Review

- Most published experiments use fewer than 400 records; variance of performance estimates is therefore high.
- Class imbalance is inconsistently treated; oversampling improves recall (Amin et al.) but is not standard practice.
- Preprocessing, feature sets, and validation (hold-out versus *k*-fold) differ across papers, limiting meta-comparison.
- The notebook’s 100% scores on the general UCI table are consistent with leakage, an overly small test partition, or near-linear separability after imputation—not with the HIV-specific metrics in the paper.

---

## 6. Recommended Future Work

1. Prospective, geographically diverse HIV-CKD registries.
2. Multi-modal architectures that fuse laboratory vectors with renal ultrasound or MRI.
3. Post-hoc explanation (SHAP, LIME) to meet clinical transparency requirements.
4. Federated training so that institutions can collaborate without exchanging identifiable records.

---

## 7. Repository Contents

```
.
├── README.md
├── IJVRA2604095.pdf          # published review article
└── CHRONIC_KIDNEY_DISEASE_STAGE_IDENTIFICATION_IN_HIV_INFECTED_PATIENTS_USING_SUPERVISED_MACHINE_LEARNING_A_REVIEW.ipynb
```

The notebook assumes a Colab path `/content/upload.csv`. Place the feature table at that location, or edit the `pd.read_csv` call. Required packages: `pandas`, `scikit-learn`, `xgboost`, `tensorflow`.

---

## 8. How to Cite

Aryan, R. (2026). Chronic kidney disease stage identification in HIV infected patients using supervised machine learning: A review. *International Journal of Versatile Research and Analysis, 4*(4), 717–725. Paper IJVRA2604095.

Primary experimental source reviewed:

Darveshwala, A. Y., Singh, D. K., & Farooqui, Y. (2021). Chronic kidney disease stage identification in HIV infected patients using machine learning. *Proceedings of the 5th International Conference on Computing Methodologies and Communication (ICCMC)*, 1509–1514. IEEE. https://doi.org/10.1109/ICCMC51019.2021.9418430

---

## 9. Licence and Disclaimer

The journal states that authors retain copyright and that the article is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The models and notebook are research artefacts; they are **not** certified diagnostic devices and must not be used for clinical decision-making without independent validation, regulatory clearance, and clinician oversight.
