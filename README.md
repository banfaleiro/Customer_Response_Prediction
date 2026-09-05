[README.md](https://github.com/user-attachments/files/31862550/README.md)
# Evaluating the Performance of ANN‑SVM Classifiers in Predicting Customer Response to a Marketing Campaign

**A Research Incubation Project**

This repository contains the research pipeline for a study comparing a **baseline Support Vector Machine (SVM)** classifier against a **proposed hybrid ANN‑SVM classifier** (an Artificial Neural Network used as a feature extractor, feeding into an SVM) for predicting whether a customer will respond to a retail marketing campaign. The work was carried out as part of a **Research Incubation (RI) programme**.

**Research Team:**
- Dr. Debrup Banerjee
- Jenifa X
- Nikita Sara M
- Brendan Faleiro
- Harshinidevi S
- Kamal P
- Sri Durga S

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Epoch Optimization](#epoch-optimization)
- [Results](#results)
- [How to Reproduce](#how-to-reproduce)
- [Tech Stack](#tech-stack)
- [Report & Full Write‑up](#report--full-write-up)
- [Citation](#citation)

---

## Overview

Customer response prediction is a core problem in marketing analytics: identifying, ahead of time, which customers are likely to respond to a campaign lets a business target outreach efficiently and profitably. This project asks a specific question:

> **Can an Artificial Neural Network, used purely as a feature extractor, improve the performance of an SVM classifier over using SVM alone on a marketing campaign dataset?**

To answer this, the study:

1. Trains and tunes a **baseline SVM** (Linear, RBF, and Exponential kernels across three train/test splits) directly on the raw campaign features.
2. Trains **22 ANN architectures/configurations** (varying hidden‑layer depth, neuron counts, and the SVM kernel applied to the extracted features) that use an ANN's hidden‑layer activations as engineered features for a downstream SVM classifier.

## Repository Structure

```
.
├── README.md                          # You are here
├── report/                            # Written report and compiled results
│   ├── Final_Report.docx              # Full research paper (intro, methodology, results, discussion)
│   └── ANN_Architecture_Search_Results.docx   # Compiled metrics table for all 22 ANN-SVM configurations
├── data/
│   └── marketing_campaign.xlsx        # Raw dataset (2,240 customers, 29 attributes)
├── notebooks/
│   └── architecture_search/           # 22 notebooks, one per ANN-SVM configuration searched
│       ├── Archi1(1).ipynb            # Input-100-50-SVM  | Hidden Layer 1 | RBF
│       ├── Archi1(2).ipynb            # Input-100-50-SVM  | Hidden Layer 2 | RBF
│       ├── Archi2(1).ipynb            # Input-500-100-SVM | Hidden Layer 1 | RBF
│       ├── Epoch2(2).ipynb            # Input-500-100-SVM | Hidden Layer 2 | RBF
│       └── Final_of_ANN2(...)_..._Epoch.ipynb  # Remaining 18 configurations (see note below)
└── assets/                            # Figures used in this README
    ├── correlation_heatmap.png
    ├── class_imbalance.png
    ├── methodology_diagram.png
    ├── epoch_optimization_convergence.png      # Combined convergence curve, all 22 configs
    └── epoch_optimization_by_architecture/     # Per-architecture F1-vs-epoch curves (18 configs)
        ├── Input-1000-500-50-SVM_HiddenLayer1_RBF.png
        ├── Input-1000-500-50-SVM_HiddenLayer1_EXPONENTIAL.png
        ├── ... (one plot per Hidden Layer × Kernel combination)
        └── Input-2000-1000-50-SVM_HiddenLayer3_EXPONENTIAL.png
```

> **Note on notebook naming:** each notebook under `notebooks/architecture_search/` is a self‑contained, end‑to‑end pipeline (load data → clean → encode → train ANN → extract features → train SVM → evaluate → plot epoch‑optimization curve) for **one** of the 22 architecture/hidden‑layer/kernel combinations listed in the report's Table 4. Filenames follow the pattern `Final_of_ANN2(<final hidden layer size>)_<run index>_Epoch.ipynb`, reflecting the number of neurons in the ANN's final hidden (feature) layer used for that run.

## Dataset

The **Marketing Campaign** dataset (`data/marketing_campaign.xlsx`) is a widely used, publicly available simulated dataset originally published on Kaggle by Rodolfo Saldanha (2020, CC0 Public Domain license), also used by iFood for data‑analyst recruitment case studies.

| Property | Value |
|---|---|
| Total records | 2,240 customers |
| Total attributes | 29 |
| Target variable | `Response` (binary: accepted campaign offer or not) |
| Missing values | 24 (in `Income` only) |
| Duplicate records | 0 |
| Class balance | 85.1% did not respond (0) · 14.9% responded (1) |

<p align="center">
  <img src="assets/class_imbalance.png" alt="Class imbalance in the Response target variable" width="600">
</p>

Because the target is imbalanced, **Accuracy alone is misleading** — Precision, Recall, F1‑score, ROC‑AUC, and Cohen's Kappa were all tracked when comparing configurations.

**Data preparation steps applied:**
- Median imputation for the 24 missing `Income` values (robust to outliers).
- Removal of 3 customer records with implausible ages (>100 years).
- Dropped non‑predictive columns: `ID`, `Dt_Customer`, `Z_CostContact`, `Z_Revenue`.
- Categorical encoding of `Education` and `Marital_Status`.

<p align="center">
  <img src="assets/correlation_heatmap.png" alt="Correlation heatmap of dataset features" width="700">
</p>

## Methodology

**1. Baseline model — SVM.** Three kernels (Linear, RBF, Exponential) were each tested across three train/test splits (70:30, 80:20, 90:10) — 9 configurations total. **Linear SVM at an 80:20 split** was the strongest and became the baseline.

**2. Proposed model — ANN‑SVM.** An ANN is trained purely as a **feature extractor**; the activations from a chosen hidden layer are then classified by a non‑linear SVM (RBF or Exponential kernel), instead of the ANN's own output layer.

<p align="center">
  <img src="assets/methodology_diagram.png" alt="ANN feature-extraction into SVM classifier methodology diagram" width="500">
</p>

Five ANN depths/widths were tested, and for each, features could be tapped from any of its hidden layers and classified with either kernel — yielding **22 total ANN‑SVM configurations**:

| Architecture | Hidden Layers | Neurons per Layer |
|---|---|---|
| 1 | 2 | 100 – 50 |
| 2 | 2 | 500 – 100 |
| 3 | 3 | 1000 – 500 – 50 |
| 4 | 3 | 2000 – 1000 – 100 |
| 5 | 3 | 2000 – 1000 – 50 |

**3. Epoch selection.** A held‑out validation split (20% of training data) was used to identify, for every architecture/hidden‑layer/kernel combination, the epoch at which validation performance peaks. Plotting all 22 configurations together produced the combined convergence curve below, which showed that validation performance had stabilised by **2,000 epochs** across the board — so every model was subsequently trained for a full 2,000 epochs before its single, final evaluation on the untouched test set.

<p align="center">
  <img src="assets/epoch_optimization_convergence.png" alt="Validation accuracy vs. epoch for all 22 ANN-SVM models" width="750">
</p>

*This combined curve was built up from the individual per‑architecture F1‑vs‑epoch curves described in the [Epoch Optimization](#epoch-optimization) section below — each one was used to locate that architecture's own best epoch before the overall convergence point was determined.*

## Epoch Optimization

Before settling on a shared training length, **each individual ANN‑SVM configuration was optimized separately**: for every architecture, hidden‑layer tap point, and SVM kernel, F1‑score on the validation split was tracked across epochs (50 → 2,000) to find that specific configuration's own best‑performing epoch. These per‑architecture curves are what fed into the combined convergence view above, and are archived in [`assets/epoch_optimization_by_architecture/`](assets/epoch_optimization_by_architecture/) for the 18 three‑hidden‑layer configurations (architectures 3–5 in the table above, each with 3 hidden‑layer tap points × 2 kernels).

| Architecture | Hidden Layer Tapped | Kernel | Best Epoch | F1 at Best Epoch |
|---|---|---|---|---|
| Input‑1000‑500‑50‑SVM | 1 | RBF | 400 | 46.46% |
| Input‑1000‑500‑50‑SVM | 1 | Exponential | 50 | 5.80% |
| Input‑1000‑500‑50‑SVM | 2 | RBF | 200 | 56.20% |
| Input‑1000‑500‑50‑SVM | 2 | Exponential | 700 | 33.73% |
| Input‑1000‑500‑50‑SVM | 3 | RBF | 1000 | 55.56% |
| Input‑1000‑500‑50‑SVM | 3 | Exponential | 2000 | 59.50% |
| Input‑2000‑1000‑100‑SVM | 1 | RBF | 150 | 43.75% |
| Input‑2000‑1000‑100‑SVM | 1 | Exponential | 2000 | 8.57% |
| Input‑2000‑1000‑100‑SVM | 2 | RBF | 250 | 55.93% |
| Input‑2000‑1000‑100‑SVM | 2 | Exponential | 600 | 34.15% |
| Input‑2000‑1000‑100‑SVM | 3 | RBF | 300 | 56.25% |
| Input‑2000‑1000‑100‑SVM | 3 | Exponential | 200 | 56.14% |
| Input‑2000‑1000‑50‑SVM | 1 | RBF | 450 | 44.44% |
| Input‑2000‑1000‑50‑SVM | 1 | Exponential | 100 | 5.80% |
| Input‑2000‑1000‑50‑SVM | 2 | RBF | 350 | 56.67% |
| Input‑2000‑1000‑50‑SVM | 2 | Exponential | 800 | 34.15% |
| Input‑2000‑1000‑50‑SVM | 3 | RBF | 150 | 55.00% |
| Input‑2000‑1000‑50‑SVM | 3 | Exponential | 400 | 55.74% |

A few examples of these per‑architecture curves:

<p align="center">
  <img src="assets/epoch_optimization_by_architecture/Input-1000-500-50-SVM_HiddenLayer3_EXPONENTIAL.png" alt="Epoch optimization: Input-1000-500-50-SVM, Hidden Layer 3, Exponential kernel" width="480">
  <img src="assets/epoch_optimization_by_architecture/Input-2000-1000-100-SVM_HiddenLayer3_RBF.png" alt="Epoch optimization: Input-2000-1000-100-SVM, Hidden Layer 3, RBF kernel" width="480">
</p>
<p align="center">
  <img src="assets/epoch_optimization_by_architecture/Input-2000-1000-50-SVM_HiddenLayer2_RBF.png" alt="Epoch optimization: Input-2000-1000-50-SVM, Hidden Layer 2, RBF kernel" width="480">
  <img src="assets/epoch_optimization_by_architecture/Input-2000-1000-50-SVM_HiddenLayer1_EXPONENTIAL.png" alt="Epoch optimization: Input-2000-1000-50-SVM, Hidden Layer 1, Exponential kernel" width="480">
</p>

The remaining 14 per‑architecture curves are available in the [`epoch_optimization_by_architecture/`](assets/epoch_optimization_by_architecture/) folder. Notably, **Exponential‑kernel configurations tapped from Hidden Layer 1** consistently underperformed (F1 around 5–9%), reinforcing the report's finding that deeper feature taps (Hidden Layer 2 or 3) are needed for the Exponential kernel to work well, while RBF‑kernel models were comparatively stable across all three tap points.

## Results

The full metrics table for all 22 ANN‑SVM configurations (Accuracy, Precision, Recall, F1, ROC‑AUC, Cohen's Kappa) is in [`report/ANN_Architecture_Search_Results.docx`](report/ANN_Architecture_Search_Results.docx). Key finding: **architectures with fewer neurons in the final (feature) hidden layer tended to perform better**, consistent with that layer acting as an information bottleneck that compresses the input into a more discriminative representation before classification.

The best-performing configuration was **Input‑2000‑1000‑50, features from Hidden Layer 2, Exponential‑kernel SVM**, which outperformed the baseline Linear SVM across the tracked metrics. Full discussion of these results is in the [final report](#report--full-write-up).

## How to Reproduce

1. **Clone this repository** and open the notebooks in Google Colab or Jupyter (the code was originally developed in Google Colab / Lightning AI Notebook).
2. **Baseline + architecture search:** run any notebook in `notebooks/architecture_search/` — each is self‑contained and, when prompted, expects `data/marketing_campaign.xlsx` to be uploaded. Each notebook performs its own preprocessing, so any single notebook can be run independently.

**Minimal environment:**
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn openpyxl
```

## Tech Stack

- **Language:** Python (Google Colab / Lightning AI Notebook)
- **Modeling:** `tensorflow`/`keras` (ANN feature extractor), `scikit-learn` (SVM, metrics)
- **Data handling & EDA:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Other:** MS Excel (initial data checks)

## Report & Full Write‑up

The complete write‑up — including the literature survey, full methodology, all metrics tables, discussion, and references — is available at [`report/Final_Report.docx`](report/Final_Report.docx).

## Citation

If you use this work, please cite the dataset's original source:

> Saldanha, R. (2020). *Marketing Campaign*. Kaggle. https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign
