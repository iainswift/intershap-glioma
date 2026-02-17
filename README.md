# Quantifying Cross-Modal Interactions in Multimodal Glioma Survival Prediction

**InterSHAP analysis for multimodal deep learning — Evidence for additive signal integration and prognostic biomarker potential**

This repository contains the implementation for replicating and extending a state-of-the-art multimodal glioma survival framework [1], and applying InterSHAP [7] to quantify cross-modal interactions between whole slide images (WSI) and RNA-seq data. The codebase includes the full training pipeline, four fusion architectures, the InterSHAP computation module, and prognostic validation analyses.

**Paper**: *Quantifying Cross-Modal Interactions in Multimodal Glioma Survival Prediction: Evidence for Additive Signal Integration and Prognostic Biomarker Potential* — Iain Swift, Jing Hua Ye, Ruairí O'Reilly (Munster Technological University)

---

## Table of Contents

1. [Key Findings](#key-findings)
2. [Overview](#overview)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Preparation](#data-preparation)
6. [Configuration](#configuration)
7. [Training Pipeline](#training-pipeline)
8. [Model Architectures](#model-architectures)
9. [InterSHAP Analysis](#intershap-analysis)
10. [Prognostic Biomarker Validation](#prognostic-biomarker-validation)
11. [Evaluation](#evaluation)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

---

## Key Findings

| Finding | Summary |
|---------|---------|
| **Inverse performance–interaction relationship** | Models achieving superior discrimination (C-index: 0.64→0.82) exhibit equivalent or *lower* cross-modal interaction (4.8%→3.0%) |
| **Stable additive decomposition** | WSI ≈ 40%, RNA ≈ 55%, Interaction ≈ 5% across all architectures |
| **Prognostic biomarker potential** | Elevated InterSHAP predicts worse survival (HR = 1.96, p < 10⁻³⁵; median 33.6 vs 114.0 months) |

---

## Overview

### Objective

This work addresses a fundamental question in multimodal learning: **do multimodal cancer survival models learn genuine cross-modal interactions, or do they simply combine independent signals additively?**

InterSHAP (a Shapley-based interaction metric) is adapted from classification to Cox survival models and applied to a replication and extension of the multimodal glioma survival framework described in [1], combining:
- **Histopathology Images**: Whole Slide Images (WSI) in SVS format
- **Gene Expression**: RNA-seq FPKM values (12,778 genes)

from the TCGA-GBM/LGG cohorts (n = 575).

### Fusion Strategies

Four fusion architectures of increasing complexity are evaluated:

| Architecture | Params | Description | Interaction Mechanism |
|-------------|--------|-------------|----------------------|
| **Early Fusion MLP** | 8.8M | Feature concatenation + MLP (baseline, replicating [1]) | Implicit (hidden layers) |
| **Cross-Attention** | 1.8M | Bidirectional attention between modalities | Explicit (learned attention) |
| **Bilinear Fusion** | 0.54M | Low-rank multiplicative interaction | Explicit (outer product) |
| **Gated Fusion** | 3.2M | Dynamic weighting conditioned on both modalities | Dynamic weighting |

Additionally, two unimodal baselines are trained:

| Model | Description | Input |
|-------|-------------|-------|
| **Unimodal WSI** | ResNet-50 on histopathology patches | 224×224 image patches |
| **Unimodal RNA** | MLP on gene expression | 12,778 gene features |

### Evaluation Metrics

- **Concordance Index (C-Index)**: Primary discrimination metric
- **Time-dependent Brier Score**: Calibration at 12, 36, and 60 months
- **InterSHAP (%)**: Cross-modal interaction as percentage of total model behaviour

---

## Project Structure

```
intershap-glioma/
│
├── 1_HistoPathology/               # WSI processing and training
│   ├── 1_WSI2Patches.py            # Extract patches from SVS files
│   ├── 2_HistoPath_train.py        # Train ResNet-50 model
│   ├── 3_HistoPath_savescore.py    # Save model predictions
│   ├── 4_HistoPath_extractfeatures.py  # Extract 2048-dim embeddings
│   ├── models.py                   # Dataset and model definitions
│   └── resnet.py                   # Modified ResNet-50 architecture
│
├── 2_GeneExpression/               # RNA-seq processing and training
│   ├── 1_GeneExpress_train.py      # Train MLP model
│   ├── 2_GeneExpress_savescore.py  # Save model predictions
│   ├── 3_GeneExpress_extractfeatures.py  # Extract embeddings
│   ├── datasets.py                 # RNA dataset loader
│   ├── models.py                   # MLP architecture
│   ├── process_rna.py              # RNA preprocessing script
│   └── genes.txt                   # List of 12,778 target genes
│
├── 3_EarlyFusion/                  # Early fusion + attention architectures
│   ├── 1_Concat2Features.py        # Concatenate WSI + RNA features
│   ├── 2_EarlyFusion_train.py      # Train fusion MLP
│   ├── 3_EarlyFusion_savescore.py  # Save predictions
│   ├── split_features.py           # Split features into train/test
│   ├── datasets.py                 # Feature dataset loader
│   ├── models.py                   # Fusion MLP architecture
│   ├── models_attention.py         # Cross-Attention, Bilinear, Gated fusion
│   ├── train_attention_models.py   # Train attention-based architectures
│   ├── run_attention_experiment.py  # Run attention experiments (5 seeds)
│   ├── analyze_attention_intershap.py  # Compare InterSHAP across architectures
│   └── apply_intershap.py          # Apply InterSHAP to early fusion model
│
├── 4_LateFusion/                   # Late Fusion (validation only)
│   ├── 1_MergeScores.py            # Combine WSI + RNA scores
│   ├── 2_LateFusion.R              # Cox regression in R
│   ├── apply_intershap.py          # Apply InterSHAP to late fusion
│   └── apply_intershap_direct.py   # Direct InterSHAP computation
│
├── 5_JointFusion/                  # Joint Fusion model
│   ├── 1_JointFusion_train.py      # End-to-end multimodal training
│   ├── 2_JointFusion_savescore.py  # Save predictions
│   ├── datasets.py                 # Multimodal dataset loader
│   ├── models.py                   # Joint architecture
│   ├── resnet.py                   # ResNet-50 for joint model
│   └── apply_intershap.py          # Apply InterSHAP to joint fusion
│
├── Intershap/                      # InterSHAP computation and validation
│   ├── intershap_utils.py          # Core library: ModalityMasker, Shapley computation
│   ├── synthetic_verification.py   # Synthetic tests (Uniqueness, XOR, Redundancy)
│   ├── paper_analysis.py           # Multi-seed cross-fusion InterSHAP orchestrator
│   ├── full_dataset_analysis.py    # Full cohort InterSHAP (→ full_dataset_intershap.csv)
│   ├── cv_robustness_analysis.py   # Cross-fold model stability
│   ├── statistical_validation.py   # Cox PH, bootstrap, RMST, permutation tests
│   └── run_system.sh               # Master execution script
│
├── config/                         # Configuration files
│   ├── survcox/                    # Cox loss configurations
│   └── survbin/                    # Survival bin configurations
│
├── ExampleData/                    # Example CSV formats for each modality
│
├── train_with_cv.py                # Cross-validation training wrapper
├── analyze_results.py              # Results analysis script
├── create_joint_splits.py          # Create joint modality splits
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Pip requirements
└── README.md                       # This file
```

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 3060 12GB)
- Conda (Miniconda or Anaconda)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/iainswift/intershap-glioma.git
cd intershap-glioma
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate fyp_multimodal
```

### Step 3: Install PyTorch with CUDA

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Dependencies

Key packages included in the environment:

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.1 | Deep learning framework |
| torchvision | 0.20.1 | Image transforms and pretrained models |
| scikit-learn | 1.3+ | Cross-validation and metrics |
| scikit-survival | 0.21+ | Survival analysis utilities |
| lifelines | 0.27+ | Concordance index, Kaplan-Meier, Cox regression |
| openslide-python | 1.3+ | WSI file reading |
| pandas | 2.0+ | Data manipulation |
| matplotlib | 3.7+ | Publication-ready figures |
| scipy | 1.10+ | Statistical tests |
| R (r-base) | 4.3+ | Late Fusion Cox regression |
| r-survival | — | R survival analysis |
| r-glmnet | — | Regularised Cox regression |

---

## Data Preparation

### Required Data Sources

1. **Whole Slide Images (WSI)**: SVS format files from TCGA-GBM and TCGA-LGG
2. **RNA-seq Data**: RSEM-normalised gene expression profiles from TCGA
3. **Clinical Data**: Overall survival, censoring status, tumour type (GBM/LGG)

### Cohort Summary

After quality filtering for complete WSI, RNA-seq, and survival data:

| Property | Value |
|----------|-------|
| Total patients | 575 |
| GBM | 67 (11.7%) |
| LGG | 508 (88.3%) |
| Death events | 195 (33.9%) |
| Median follow-up (censored) | 14.3 months |

### Step 1: Extract Patches from WSI

Whole-slide images are processed at 20× magnification into 224×224 patches:

```bash
python 1_HistoPathology/1_WSI2Patches.py \
    --wsi_path "MyData/wsi_files/" \
    --patch_path "MyData/patches/" \
    --mask_path "MyData/masks/" \
    --patch_size 224 \
    --max_patches_per_slide 2000 \
    --num_process 10
```

### Step 2: Process RNA Data

RSEM-normalised RNA-seq profiles are filtered to genes with >80% non-zero expression, yielding 12,778 features:

```bash
python 2_GeneExpression/process_rna.py
```

This script applies log2(x+1) normalisation, selects the 12,778 target genes, and merges with clinical data.

### Step 3: Prepare Data Splits

An 80%/20% train/test split stratified by outcome is used, with 10-fold cross-validation on the full dataset for robustness assessment:

```bash
python create_joint_splits.py
```

---

## Configuration

All training parameters are specified in JSON configuration files in `config/survcox/`. See existing configs for WSI, RNA, Early Fusion, and Joint Fusion models.

### Training Protocol (All Architectures)

| Parameter | Value |
|-----------|-------|
| Loss function | Cox partial likelihood |
| Optimiser | Adam |
| Learning rate | 10⁻⁴ (attention models), 10⁻⁶ (MLP baseline) |
| Weight decay | 10⁻⁵ |
| Batch size | 128 |
| Early stopping | Validation C-index, patience 20 epochs |
| Epochs | 80 (attention models) |
| Multi-seed training | 5 random seeds per architecture |

---

## Training Pipeline

### Overview

The complete pipeline consists of five stages:

```
Stage 1: Unimodal Training
    WSI Model (ResNet-50) ────────┐
    RNA Model (MLP) ──────────────┤
                                  │
Stage 2: Feature Extraction       │
    Extract WSI embeddings (2048D)┤
    Extract RNA embeddings (2048D)┤
    Save model scores ────────────┤
                                  │
Stage 3: Fusion Training          │
    Early Fusion MLP (baseline) ──┤
    Cross-Attention ──────────────┤
    Bilinear Fusion ──────────────┤
    Gated Fusion ─────────────────┤
    Late Fusion (Cox, validation) ┤
                                  │
Stage 4: InterSHAP Computation    │
    Synthetic validation ─────────┤
    Coalition evaluation ─────────┤
    Variance decomposition ───────┤
                                  │
Stage 5: Prognostic Validation    │
    Cox regression ───────────────┤
    Kaplan-Meier analysis ────────┤
    Bootstrap / Permutation ──────┘
```

### Stage 1: Unimodal Model Training

```bash
# Train WSI model (~7.5 hours for 10 folds)
python train_with_cv.py \
    --modality ffpe \
    --config config/survcox/config_ffpe_train_survcox.json \
    --n_folds 10

# Train RNA model (~1.5 hours for 10 folds)
python train_with_cv.py \
    --modality rna \
    --config config/survcox/config_rna_train_survcox.json \
    --n_folds 10
```

### Stage 2: Feature and Score Extraction

```bash
# Save prediction scores
python 1_HistoPathology/3_HistoPath_savescore.py \
    --config config/survcox/config_ffpe_savescore_survcox.json
python 2_GeneExpression/2_GeneExpress_savescore.py \
    --config config/survcox/config_rna_savescore_survcox.json

# Extract 2048-dimensional feature embeddings
python 1_HistoPathology/4_HistoPath_extractfeatures.py \
    --config config/survcox/config_ffpe_extractfeatures_survcox.json
python 2_GeneExpression/3_GeneExpress_extractfeatures.py \
    --config config/survcox/config_rna_extractfeatures_survcox.json

# Prepare fusion data
python 3_EarlyFusion/1_Concat2Features.py
python 3_EarlyFusion/split_features.py
python 4_LateFusion/1_MergeScores.py
```

### Stage 3: Fusion Model Training

```bash
# Early Fusion MLP — baseline replicating [1] (~2 hours)
python train_with_cv.py \
    --modality early \
    --config config/survcox/config_feature_train_survcox.json \
    --n_folds 10

# Late Fusion — validation zero-check (~1 minute)
Rscript 4_LateFusion/2_LateFusion.R

# Attention-based architectures (5 seeds each)
python 3_EarlyFusion/train_attention_models.py
python 3_EarlyFusion/run_attention_experiment.py
```

### Stage 4: InterSHAP Computation

```bash
# Run full InterSHAP pipeline (synthetic validation → real data)
bash Intershap/run_system.sh

# Or run individually:
python Intershap/synthetic_verification.py      # Verify on known interaction patterns
python Intershap/full_dataset_analysis.py        # Full cohort InterSHAP scores
python Intershap/paper_analysis.py               # Multi-seed cross-fusion comparison
python Intershap/cv_robustness_analysis.py       # Cross-fold stability
```

This executes:
1. **Synthetic validation** — verifies implementation on data with known interaction patterns
2. **Late Fusion zero-check** — confirms |InterSHAP| < 10⁻¹⁵ for linear combinations
3. **Coalition evaluation** — 4 forward passes × 575 patients × 4 architectures
4. **Variance decomposition** — WSI vs RNA vs Interaction breakdown
5. **Masking ablation** — mean imputation vs random shuffle vs zero imputation

Runtime: ~3 minutes per architecture on an RTX 3060.

### Stage 5: Prognostic Validation

```bash
# All 14 validation analyses in a single script:
# Cox PH, quartile dose-response, bootstrap (n=1000), subgroup analysis,
# sensitivity, Schoenfeld test, C-index, 10-fold CV, unimodal comparison,
# Cohen's d / NNH, RMST, landmark, piecewise, permutation (n=10,000)
python Intershap/statistical_validation.py
```

---

## Model Architectures

### WSI Encoder (Histopathology)

```
Input: 224×224 RGB patches (100 patches per WSI at 20× magnification)
    → ResNet-50 (ImageNet-pretrained, last 2 blocks trainable)
    → Slide-level averaging
    → Output: 2048-dimensional embedding
```

### RNA Encoder (Gene Expression)

```
Input: 12,778 gene expression values (log2 RSEM)
    → Linear(12778 → 4096) + ReLU + Dropout(0.5)
    → Linear(4096 → 2048) + ReLU + Dropout(0.5)
    → Output: 2048-dimensional embedding
```

### Early Fusion MLP (Baseline)

```
Input: 4096 concatenated features (2048 WSI ∥ 2048 RNA)
    → Linear(4096 → 2048) + ReLU + Dropout(0.25)
    → Linear(2048 → 200) + ReLU + Dropout(0.25)
    → Linear(200 → 1)
    → Output: Log-risk score ĥ
```

Replicates the architecture in [1]. Can learn interactions implicitly through hidden layers but does not explicitly encode them.

### Cross-Attention Fusion

Bidirectional attention where each modality queries the other:

```
A_WSI→RNA = softmax(Q_WSI · K_RNA^T / √d) · V_RNA
```

Explicitly models cross-modal dependencies through learned attention weights (1.8M params).

### Bilinear Fusion

Low-rank multiplicative interaction:

```
z_bilinear = (W₁ · RNA) ⊙ (W₂ · WSI),   Wᵢ ∈ ℝ^(64×2048)
```

Forces the model to compute outer-product interactions between modalities (0.54M params).

### Gated Fusion

Dynamic weighting conditioned on both modalities:

```
α = σ(W_g[RNA; WSI])
z = α ⊙ f(RNA) + (1 − α) ⊙ g(WSI)
```

Learns to dynamically balance modality contributions (3.2M params).

### Late Fusion (Validation)

Linear combination of unimodal risk scores via L1-regularised Cox regression. Used as a zero-check: InterSHAP must be exactly zero by construction (verified: |InterSHAP| < 10⁻¹⁵).

---

## InterSHAP Analysis

### Adaptation for Cox Survival Models

InterSHAP [7] is adapted for survival prediction with three key design choices:

1. **Output space**: Shapley values are computed on the log-risk score ĥ (raw network output) rather than the hazard exp(ĥ), to avoid conflating model-learned interactions with mathematical artefacts of the Cox formulation.

2. **Coalition evaluation**: For two modalities, four coalitions are evaluated per patient:
   - v(∅): Both modalities masked (baseline prediction)
   - v({WSI}): WSI present, RNA masked
   - v({RNA}): RNA present, WSI masked
   - v({WSI, RNA}): Full model (both present)

3. **Masking strategy**: Missing modalities replaced by dataset-mean embeddings. Validated against random-shuffle and zero-imputation baselines.

### Shapley Computation (M = 2 modalities)

```
ϕ_WSI = ½[v({WSI}) − v(∅)] + ½[v({WSI, RNA}) − v({RNA})]
ϕ_RNA = ½[v({RNA}) − v(∅)] + ½[v({WSI, RNA}) − v({WSI})]
ϕ_int = ½[v({WSI, RNA}) − v({WSI}) − v({RNA}) + v(∅)]
```

### Global InterSHAP

Interaction as a percentage of total model behaviour:

```
InterSHAP = Σᵢ |ϕ_int⁽ⁱ⁾| / Σᵢ (|ϕ_WSI⁽ⁱ⁾| + |ϕ_RNA⁽ⁱ⁾| + |ϕ_int⁽ⁱ⁾|) × 100%
```

### Synthetic Validation

Three scenarios with known interaction patterns verify implementation correctness:

| Scenario | Expected | Deterministic | FCNN (Learned) |
|----------|----------|---------------|----------------|
| Uniqueness (single modality) | 0% | 0.0% | 2.1% |
| Synergy (XOR, both required) | 100% | 100% | 99.7% |
| Redundancy (shared info) | 30–50% | 40% | 35.2% |

### Results: Performance vs Interaction

| Architecture | Params | Test C-Index | InterSHAP (%) | Brier@36mo |
|-------------|--------|-------------|---------------|------------|
| Early Fusion MLP | 8.8M | 0.636 ± 0.02 | 4.82 ± 4.65 | 0.192 |
| Cross-Attention | 1.8M | 0.814 ± 0.01 | 3.03 ± 2.64 | 0.156 |
| Bilinear Fusion | 0.54M | 0.819 ± 0.01 | 3.72 ± 3.16 | 0.151 |
| Gated Fusion | 3.2M | 0.807 ± 0.02 | 4.45 ± 4.94 | 0.162 |

Values show mean ± SD from 5-seed training. InterSHAP computed on test set (n = 115).

### Variance Decomposition

Stable across all architectures:

| Component | MLP | Cross-Attn | Bilinear | Gated |
|-----------|-----|-----------|----------|-------|
| WSI | 42.2 ± 2.1% | 43.2 ± 1.8% | 36.8 ± 2.4% | 43.7 ± 2.0% |
| RNA | 53.0 ± 2.3% | 53.8 ± 1.9% | 59.5 ± 2.6% | 51.8 ± 2.2% |
| Interaction | 4.8 ± 0.8% | 3.0 ± 0.5% | 3.7 ± 0.6% | 4.5 ± 0.9% |

---

## Prognostic Biomarker Validation

While global InterSHAP is low (~5%), individual patient-level variation carries strong prognostic signal.

### Cox Regression

| Model | HR (95% CI) | p-value | C-index |
|-------|------------|---------|---------|
| Univariate (per SD) | 1.96 (1.76–2.18) | 1.5 × 10⁻³⁵ | 0.755 |
| Univariate (high vs low) | 3.49 (2.53–4.83) | 4.1 × 10⁻¹⁴ | — |
| Multivariate (per SD, adjusted for tumour type) | 1.60 (1.39–1.83) | 3.9 × 10⁻¹¹ | 0.782 |
| Fully adjusted (age, sex, tumour type, KPS) | 1.42 (1.21–1.67) | < 0.0001 | — |

### Kaplan-Meier Stratification

Stratification by median InterSHAP:
- **High InterSHAP** (≥ median): 33.6 months median survival
- **Low InterSHAP** (< median): 114.0 months median survival
- Difference: 80.4 months (p < 10⁻¹⁵, log-rank test)
- 5-year survival: 34% vs 70%

### Dose-Response by Quartile

| Quartile | N | Mean InterSHAP (%) | Median Survival | Events |
|----------|---|-------------------|-----------------|--------|
| Q1 (Lowest) | 144 | 0.16 | 114.0 months | 27 |
| Q2 | 144 | 0.54 | 133.6 months | 29 |
| Q3 | 143 | 1.12 | 69.8 months | 37 |
| Q4 (Highest) | 144 | 2.86 | 16.8 months | 102 |

Trend test: p < 10⁻³⁶

### Robustness Analyses

- **Bootstrap** (n = 1,000): HR = 2.02 (95% CI: 1.71–2.52)
- **Permutation test** (n = 10,000): p = 0.0032
- **Landmark analysis**: Effect persists with attenuation (HR 1.96 → 1.23 at 36 months, all p < 0.05)
- **RMST**: High-InterSHAP patients lost 16.3 months of expected survival at 60 months
- **Subgroup**: Effect concentrated in LGG (HR = 1.99, n = 508); GBM underpowered (HR = 1.06, n = 67)

---

## Evaluation

### Concordance Index (C-Index)

```
C-Index = P(risk(i) > risk(j) | survival(i) < survival(j))
```

| Score | Interpretation |
|-------|---------------|
| 0.5 | Random predictions |
| 0.6–0.7 | Acceptable discrimination |
| 0.7–0.8 | Good discrimination |
| 0.8+ | Excellent discrimination |

### Results Summary

| Model | Test C-Index |
|-------|-------------|
| WSI Only (unimodal) | ~0.65 |
| RNA Only (unimodal) | ~0.70 |
| Early Fusion MLP | 0.636 ± 0.02 |
| Cross-Attention | 0.814 ± 0.01 |
| Bilinear Fusion | 0.819 ± 0.01 |
| Gated Fusion | 0.807 ± 0.02 |

### Replication Note

The Early Fusion MLP (C-index 0.636) underperforms the original implementation in [1] (C-index 0.84). This replication gap is discussed in the paper; the interaction analysis retains internal validity as it compares architectures trained identically under controlled conditions.

### Output Files

Each trained model generates predictions in:

```
MyData/results/{modality}/checkpoints/outputs/{model_name}/
├── train_output_best.csv
├── val_output_best.csv
├── test_output_best.csv
└── model_best.pth
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce `batch_size` in config (try 64 or 32) or reduce `max_patch_per_wsi_train` (try 50).

### Zero or Negative Survival Times

Ensure all `survival_months` values are > 0. Preprocessing scripts set a minimum of 0.1 months.

### Missing Patches

Verify patch extraction completed: each folder should contain `loc.txt` and `*_patch_*.png` files.

### R Package Errors

Install Bioconductor packages:
```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("survcomp")
```

### InterSHAP Zero-Check Failure

If the Late Fusion zero-check yields |InterSHAP| > 10⁻¹⁰, there is likely an implementation error in the coalition evaluation. Verify that the Late Fusion model is a strict linear combination of unimodal scores.

---

## References

1. Steyaert, S., et al.: Multimodal deep learning to predict prognosis in brain tumors. *Commun. Med.* 3, 44 (2023). https://doi.org/10.1038/s43856-023-00349-y
2. Zhou, H., et al.: Multimodal data integration for precision oncology: a survey. arXiv:2406.19611 (2024)
3. Zheng, Y., et al.: Spatial cellular architecture predicts prognosis in glioblastoma. *Nat. Commun.* 14, 4122 (2023)
4. Mobadersany, P., et al.: Predicting cancer outcomes from histology and genomics. *PNAS* 115(13), E2970–E2979 (2018)
5. Guarrasi, V., et al.: Finding optimal fusion points in multimodal medical imaging. *IEEE TMI* 44(1), 234–247 (2025)
6. Wenderoth, L., et al.: Measuring cross-modal interactions in multimodal models. *AAAI-25*, pp. 21501–21509 (2025)
7. Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. *NeurIPS 30*, 4765–4774 (2017)
8. Cox, D.R.: Regression models and life-tables. *JRSS-B* 34(2), 187–220 (1972)

### Data Sources

- TCGA-GBM: https://portal.gdc.cancer.gov/projects/TCGA-GBM
- TCGA-LGG: https://portal.gdc.cancer.gov/projects/TCGA-LGG

---

## License

This project is for academic research purposes. Please cite the paper if using this code.

## Citation

```bibtex
@inproceedings{swift2026intershap,
  title={Quantifying Cross-Modal Interactions in Multimodal Glioma Survival Prediction: Evidence for Additive Signal Integration and Prognostic Biomarker Potential},
  author={Swift, Iain and Ye, Jing Hua and O'Reilly, Ruair{\'\i}},
  booktitle={Proceedings of the xAI World Conference},
  year={2026}
}
```

## Contact

For questions regarding this implementation, please open an issue on the GitHub repository.
