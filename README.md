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

## 2. Datasets

Two related tabular sources underpin this review and the companion notebook.

### 2.1 UCI Chronic Kidney Disease (canonical public benchmark)

**Source:** Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository.  
**URL:** [https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)  
**DOI:** [10.24432/C5G020](https://doi.org/10.24432/C5G020)  
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

| Property | Value |
|----------|--------|
| Donated | 3 July 2015 |
| Task | Binary classification (`ckd` / `notckd`) |
| Instances | 400 (hospital collection over approximately two months) |
| Features | 24 predictors + 1 class label (11 numeric, 14 nominal) |
| Missing values | Yes |
| Subject area | Health / other |

Attribute dictionary (UCI official encoding):

| Code | Feature | Scale / values |
|------|---------|----------------|
| age | Age | years |
| bp | Blood pressure | mmHg |
| sg | Specific gravity | 1.005, 1.010, 1.015, 1.020, 1.025 |
| al | Albumin | 0–5 |
| su | Sugar | 0–5 |
| rbc | Red blood cells | normal / abnormal |
| pc | Pus cell | normal / abnormal |
| pcc | Pus cell clumps | present / notpresent |
| ba | Bacteria | present / notpresent |
| bgr | Blood glucose random | mg/dL |
| bu | Blood urea | mg/dL |
| sc | Serum creatinine | mg/dL |
| sod | Sodium | mEq/L |
| pot | Potassium | mEq/L |
| hemo | Haemoglobin | g |
| pcv | Packed cell volume | % |
| wc | White blood cell count | cells/cmm |
| rc | Red blood cell count | millions/cmm |
| htn | Hypertension | yes / no |
| dm | Diabetes mellitus | yes / no |
| cad | Coronary artery disease | yes / no |
| appet | Appetite | good / poor |
| pe | Pedal oedema | yes / no |
| ane | Anaemia | yes / no |
| class | Target | ckd / notckd |

Programmatic access:

```python
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo

chronic_kidney_disease = fetch_ucirepo(id=336)
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets
```

The companion notebook loads a CSV (`upload.csv`) whose column names match this schema. Perfect scores obtained in that notebook on the 400-record table should not be read as a reproduction of the HIV-specific metrics in Section 4.

### 2.2 HIV-CKD literature cohort (Darveshwala et al., 2021)

The primary experimental study synthesised in the review reports a UCI-derived **HIV-CKD** collection of **158 instances** and **27 attributes** (the UCI laboratory panel plus HIV-oriented demographic encodings such as gender and ethnicity). After binary classification, 44 patients were labelled CKD-positive; eGFR staging of those 44 cases concentrated in Stages 4–5 (10 and 25 patients, respectively).

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

## 7. How to Cite

Aryan, R. (2026). Chronic kidney disease stage identification in HIV infected patients using supervised machine learning: A review. *International Journal of Versatile Research and Analysis, 4*(4), 717–725. Paper IJVRA2604095.

Primary experimental source reviewed:

Darveshwala, A. Y., Singh, D. K., & Farooqui, Y. (2021). Chronic kidney disease stage identification in HIV infected patients using machine learning. *Proceedings of the 5th International Conference on Computing Methodologies and Communication (ICCMC)*, 1509–1514. IEEE. https://doi.org/10.1109/ICCMC51019.2021.9418430

Canonical public dataset:

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

---

## 8. Licence and Disclaimer

The journal states that authors retain copyright and that the article is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The models and notebook are research artefacts; they are **not** certified diagnostic devices and must not be used for clinical decision-making without independent validation, regulatory clearance, and clinician oversight.

## 🔗 Other repositories by Rishabh Aryan

| Repository | Description | Live |
|---|---|---|
| [100-days-of-python](https://github.com/Rishabh-bgp/100-days-of-python) | My original, offline, no-API-key solutions to all 100 projects of the 100 Days of Code Python bootcamp curriculum… | — |
| [Agentic-Engineering-Ebook](https://github.com/Rishabh-bgp/Agentic-Engineering-Ebook) | Agentic Engineering is a practical notebook-book on specifying, generating, verifying, and refining multi-tenant… | — |
| [beed-ml-study](https://github.com/Rishabh-bgp/beed-ml-study) | Provenance audit and leakage-aware benchmark of seizure-type classifiers on the Bangalore EEG Epilepsy Dataset — IIT… | — |
| [Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction](https://github.com/Rishabh-bgp/Brain-Computer-Interfaces-and-Cognitive-Computing-An-Introduction) | Brain–computer interfaces (BCIs) link neural activity to external devices, translating brain signals via algorithms… | — |
| [building-a-virtual-cell-in-python](https://github.com/Rishabh-bgp/building-a-virtual-cell-in-python) | A cell has a normal gene-activity pattern. Knock out one gene and that pattern changes. This project trains a small… | — |
| [cobratate](https://github.com/Rishabh-bgp/cobratate) | Cobratate, an esoteric imperative programming language. Tate-flavored syntax, ordinary semantics, and a Python… | [open](https://rishabh-bgp.github.io/cobratate/) |
| [coda-synth](https://github.com/Rishabh-bgp/coda-synth) | CoDA-Synth: stylometric detection of LLM-rewritten dark-web documents with sparse TF-IDF + LinearSVC. Research lab,… | [open](https://shale-ember-delta-kite.grok.me/) |
| [IIT-ROPAR-GENAI-FOUNDATIONS](https://github.com/Rishabh-bgp/IIT-ROPAR-GENAI-FOUNDATIONS) | Free open-source Generative AI foundations ebooks (2026) covering Claude, Grok, Gemini, Cursor, OpenAI and more —… | — |
| [kbc-game](https://github.com/Rishabh-bgp/kbc-game) | Kaun Banega Crorepati (KBC) quiz game for the terminal in Python: 16-question prize ladder, safe levels and 4… | — |
| [knn-classifier](https://github.com/Rishabh-bgp/knn-classifier) | Simple K-Nearest Neighbors classifier on the classic Iris dataset using scikit-learn. Loads data, trains the model,… | — |
| [linear-regression-model-](https://github.com/Rishabh-bgp/linear-regression-model-) | Supervised linear regression on the scikit-learn diabetes dataset. A single feature (BMI) is used to predict… | — |
| [logistic-regression](https://github.com/Rishabh-bgp/logistic-regression) | This graduate capstone applies multi-class logistic regression to the Iris dataset. It includes EDA, stratified… | — |
| [logistics-website-capstone](https://github.com/Rishabh-bgp/logistics-website-capstone) | Logistics website capstone: Kargonaut Logistics, a premium freight site mixing claymorphism, glassmorphism and… | [open](https://rishabh-bgp.github.io/logistics-website-capstone/) |
| [lovelight-leap](https://github.com/Rishabh-bgp/lovelight-leap) | 💖 Lovelight Leap: a love-themed 2D platformer in vanilla JS canvas. 4 worlds, a boss you cheer up, original… | [open](https://rishabh-bgp.github.io/lovelight-leap/) |
| [machine-learning-part-1](https://github.com/Rishabh-bgp/machine-learning-part-1) | Machine learning in Python with scikit-learn: heart disease classification and medical insurance cost regression… | — |
| [Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail](https://github.com/Rishabh-bgp/Meeting-of-Machines-Consumer-Agents-Seller-Agents-and-Hidden-Fulfilment-Risk-in-Online-Retail) | Agentic commerce lets shopping and seller agents place orders without showing operational truth. This IS case argues… | — |
| [morocco-city-aware-load-forecasting](https://github.com/Rishabh-bgp/morocco-city-aware-load-forecasting) | City-aware short-term load forecasting and peak detection on high-resolution Moroccan smart-meter data (5 zones).… | — |
| [NIELIT-DATA-SCIENCE](https://github.com/Rishabh-bgp/NIELIT-DATA-SCIENCE) | Hands-on NumPy, Pandas, and Matplotlib Jupyter notebooks for the NIELIT Data Science curriculum — practical,… | — |
| [NIELIT-MACHINE-LEARNING-PROJECTS](https://github.com/Rishabh-bgp/NIELIT-MACHINE-LEARNING-PROJECTS) | Nineteen end-to-end machine learning projects: classification, regression, clustering, and recommendation, with… | — |
| [NIELIT-PYTHON-](https://github.com/Rishabh-bgp/NIELIT-PYTHON-) | Open-source Python teaching notebooks: a language reference and five capstone programs, with detailed documentation.… | — |
| [pan-cancer-classification](https://github.com/Rishabh-bgp/pan-cancer-classification) | Leakage, equivalence, conformal reliability and data efficiency on the UCI TCGA Pan-Cancer RNA-Seq dataset — IIT… | — |
| [Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks](https://github.com/Rishabh-bgp/Predicting-Hot-Events-in-the-Early-Period-through-Bayesian-Model-for-Social-Networks) | This notebook details the implementation and evaluation of a Bayesian perspective Early stage Event Prediction… | — |
| [prediction-as-power](https://github.com/Rishabh-bgp/prediction-as-power) | Laboratory and working paper on prediction as power in European electricity markets. A DE-like hourly simulator… | — |
| [snakes-and-ladders](https://github.com/Rishabh-bgp/snakes-and-ladders) | 🐍🪜 Original Snakes & Ladders browser game: hand-drawn SVG snakes and ladders, 3D dice, 2-4 humans or bots, sound,… | [open](https://rishabh-bgp.github.io/snakes-and-ladders/) |
| [spam-scan](https://github.com/Rishabh-bgp/spam-scan) | Explainable spam, phishing & Indian-scam detector for SMS/email (English, Hindi, Hinglish): TF-IDF + Logistic… | [open](https://spam-scan.onrender.com) |
| [vedic-math-hybrid-kt](https://github.com/Rishabh-bgp/vedic-math-hybrid-kt) | Hybrid Vedic Math AI: rule-based sutras (Urdhva-Tiryagbhyam, Nikhilam, Yavadunam) plus RF classifier that cleans… | — |
| [wind-turbine-power-model](https://github.com/Rishabh-bgp/wind-turbine-power-model) | Single-turbine power curves + wind-farm simulation. Weibull wind stats, hourly NASA-style speed data, and… | — |
| [zoo-website-capstone](https://github.com/Rishabh-bgp/zoo-website-capstone) | Capstone project: a fictional zoo website built twice — a beginner-style simple version and a cinematic advanced… | [open](https://rishabh-bgp.github.io/zoo-website-capstone/) |
