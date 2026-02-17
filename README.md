# Multimodal Deep Learning for Brain Tumor Survival Prediction

A deep learning framework for predicting prognosis in brain tumors by integrating histopathology whole slide images (WSI) with gene expression profiles. This implementation is based on the methodology described in "Multimodal deep learning to predict prognosis in adult and pediatric brain tumors" (Steyaert et al., Communications Medicine, 2023).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Configuration](#configuration)
6. [Training Pipeline](#training-pipeline)
7. [Model Architectures](#model-architectures)
8. [Evaluation](#evaluation)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Overview

### Objective

Predict patient survival outcomes using multimodal data fusion of:
- **Histopathology Images**: Whole Slide Images (WSI) in SVS format
- **Gene Expression**: RNA-seq FPKM values (12,778 genes)

### Fusion Strategies

This framework implements four distinct approaches:

| Model | Description | Input |
|-------|-------------|-------|
| **Unimodal WSI** | ResNet50 on histopathology patches | 224x224 image patches |
| **Unimodal RNA** | MLP on gene expression | 12,778 gene features |
| **Early Fusion** | MLP on concatenated extracted features | 4,096 features (2,048 WSI + 2,048 RNA) |
| **Late Fusion** | Cox regression on model scores | 2 scores (WSI + RNA) |
| **Joint Fusion** | End-to-end multimodal network | Raw patches + gene expression |

### Evaluation Metric

All models are evaluated using the **Concordance Index (C-Index)**, which measures the model's ability to correctly rank patient survival times. A C-Index of 0.5 indicates random performance, while 1.0 indicates perfect ranking.

---

## Project Structure

```
Thesis-MultiModal-Survival/
|
|-- 1_HistoPathology/           # WSI processing and training
|   |-- 1_WSI2Patches.py        # Extract patches from SVS files
|   |-- 2_HistoPath_train.py    # Train ResNet50 model
|   |-- 3_HistoPath_savescore.py    # Save model predictions
|   |-- 4_HistoPath_extractfeatures.py  # Extract 2048-dim embeddings
|   |-- models.py               # Dataset and model definitions
|   +-- resnet.py               # Modified ResNet50 architecture
|
|-- 2_GeneExpression/           # RNA-seq processing and training
|   |-- 1_GeneExpress_train.py  # Train MLP model
|   |-- 2_GeneExpress_savescore.py  # Save model predictions
|   |-- 3_GeneExpress_extractfeatures.py  # Extract embeddings
|   |-- datasets.py             # RNA dataset loader
|   |-- models.py               # MLP architecture
|   |-- process_rna.py          # RNA preprocessing script
|   +-- genes.txt               # List of 12,778 target genes
|
|-- 3_EarlyFusion/              # Early fusion model
|   |-- 1_Concat2Features.py    # Concatenate WSI + RNA features
|   |-- 2_EarlyFusion_train.py  # Train fusion MLP
|   |-- 3_EarlyFusion_savescore.py  # Save predictions
|   |-- split_features.py       # Split features into train/test
|   |-- datasets.py             # Feature dataset loader
|   +-- models.py               # Fusion MLP architecture
|
|-- 4_LateFusion/               # Late fusion model
|   |-- 1_MergeScores.py        # Combine WSI + RNA scores
|   +-- 2_LateFusion.R          # Cox regression in R
|
|-- 5_JointFusion/              # Joint fusion model
|   |-- 1_JointFusion_train.py  # End-to-end multimodal training
|   |-- 2_JointFusion_savescore.py  # Save predictions
|   |-- datasets.py             # Multimodal dataset loader
|   |-- models.py               # Joint architecture
|   +-- resnet.py               # ResNet50 for joint model
|
|-- config/                     # Configuration files
|   |-- survcox/                # Cox loss configurations
|   |   |-- config_ffpe_train_survcox.json
|   |   |-- config_rna_train_survcox.json
|   |   |-- config_joint_train_survcox.json
|   |   |-- config_feature_train_survcox.json
|   |   +-- ... (savescore and extractfeatures configs)
|   +-- survbin/                # Survival bin configurations (alternative)
|
|-- MyData/                     # Your data directory
|   |-- patientinfo.csv         # Patient metadata
|   |-- patches/                # Extracted WSI patches
|   |-- masks/                  # Tissue segmentation masks
|   |-- splits/                 # Train/test CSV files
|   +-- results/                # Model outputs
|
|-- ExampleData/                # Example CSV formats
|
|-- train_with_cv.py            # Cross-validation training wrapper
|-- analyze_results.py          # Results analysis script
|-- create_joint_splits.py      # Create joint modality splits
|-- environment.yml             # Conda environment specification
|-- requirements.txt            # Pip requirements
+-- README.md                   # This file
```

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 3060 12GB)
- Conda (Miniconda or Anaconda)
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/ISwift7/Thesis-MultiModal-Survival.git
cd Thesis-MultiModal-Survival
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

Expected output:
```
PyTorch: 2.5.1
CUDA Available: True
```

### Dependencies

Key packages included in the environment:

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.1 | Deep learning framework |
| torchvision | 0.20.1 | Image transforms and pretrained models |
| scikit-learn | 1.3+ | Cross-validation and metrics |
| scikit-survival | 0.21+ | Survival analysis utilities |
| lifelines | 0.27+ | Concordance index calculation |
| openslide-python | 1.3+ | WSI file reading |
| pandas | 2.0+ | Data manipulation |
| R (r-base) | 4.3+ | Late fusion Cox regression |
| r-survival | - | R survival analysis |
| r-glmnet | - | Regularized Cox regression |

---

## Data Preparation

### Required Data Sources

1. **Whole Slide Images (WSI)**: SVS format files from TCGA
2. **RNA-seq Data**: FPKM gene expression files from TCGA
3. **Clinical Data**: Survival time, vital status, and grade information

### Directory Structure

```
MyData/
|-- patientinfo.csv         # Patient metadata (required)
|-- wsi_files/              # Raw SVS files (for patch extraction)
|-- patches/                # Extracted patches (created by Step 1)
|   +-- TCGA-XX-XXXX-01Z-00-DX1/
|       |-- loc.txt
|       +-- TCGA-XX-XXXX-01Z-00-DX1_patch_0.png
|       +-- ...
|-- masks/                  # Tissue masks (created by Step 1)
|-- splits/                 # Train/test CSV files (required)
|   |-- train_wsi.csv
|   |-- test_wsi.csv
|   |-- rna_train.csv
|   |-- rna_test.csv
|   |-- joint_train.csv
|   +-- joint_test.csv
+-- results/                # Model outputs (created during training)
```

### Step 1: Extract Patches from WSI

```bash
python 1_HistoPathology/1_WSI2Patches.py \
    --wsi_path "MyData/wsi_files/" \
    --patch_path "MyData/patches/" \
    --mask_path "MyData/masks/" \
    --patch_size 224 \
    --max_patches_per_slide 2000 \
    --num_process 10
```

Parameters:
- `--patch_size`: Patch dimensions (default: 224x224 for ResNet)
- `--max_patches_per_slide`: Maximum patches per WSI (default: 2000)
- `--num_process`: Parallel processing workers

### Step 2: Process RNA Data

If starting from raw TCGA RNA-seq files:

```bash
python 2_GeneExpression/process_rna.py
```

This script:
1. Loads FPKM values from TSV files
2. Maps samples to TCGA barcodes
3. Applies log2(x+1) normalization
4. Selects the 12,778 target genes
5. Merges with clinical data
6. Creates survival bins (quartiles)

### Step 3: Prepare Data Splits

Create the required CSV files in `MyData/splits/`:

**train_wsi.csv / test_wsi.csv format:**
```csv
case,wsi_file_name,survival_months,vital_status,survival_bin
TCGA-FG-A60L,TCGA-FG-A60L-01Z-00-DX1,21.52,0,2
```

**rna_train.csv / rna_test.csv format:**
```csv
case,survival_months,vital_status,survival_bin,grade_binary,rna_0,rna_1,...,rna_12777
TCGA-FG-A60L,21.52,0,2,1,3.45,2.12,...,0.89
```

**joint_train.csv / joint_test.csv format:**
```csv
case,survival_months,vital_status,survival_bin,wsi_file_name,rna_0,rna_1,...,rna_12777
TCGA-FG-A60L,21.52,0,2,TCGA-FG-A60L-01Z-00-DX1,3.45,2.12,...,0.89
```

### Data Split Guidelines

Following the paper methodology:
- **Train/Test Split**: 80% / 20% stratified by survival bin and vital status
- **Cross-Validation**: 10-fold stratified CV on training set
- **No Validation Leakage**: Test set is never seen during training or model selection

Use the provided script to create joint splits:
```bash
python create_joint_splits.py
```

---

## Configuration

### Configuration Files

All training parameters are specified in JSON configuration files located in `config/survcox/`.

### WSI Model Configuration (config_ffpe_train_survcox.json)

```json
{
    "model_name": "resnet50",
    "num_classes": 1,
    "batch_size": 128,
    "num_epochs": 10,
    "img_size": 224,
    "lr": 0.0005,
    "weight_decay": 0.001,
    "pretrained": true,
    "n_layers_to_train": 2,
    "aggregator": "identity",
    "aggregator_hdim": 2048,
    "max_patch_per_wsi_train": 100,
    "task": "survival_prediction",
    "data_path": "MyData/patches",
    "train_csv_path": "MyData/splits/train_wsi.csv",
    "test_csv_path": "MyData/splits/test_wsi.csv",
    "checkpoint_path": "MyData/results/ffpe/checkpoints/",
    "flag": "ffpe_model_survcox"
}
```

### RNA Model Configuration (config_rna_train_survcox.json)

```json
{
    "task": "survival_prediction",
    "num_classes": 1,
    "batch_size": 128,
    "num_epochs": 20,
    "lr_rna": 1e-05,
    "lr_mlp": 1e-05,
    "weight_decay": 1e-05,
    "train_csv_path": "MyData/splits/rna_train.csv",
    "test_csv_path": "MyData/splits/rna_test.csv",
    "checkpoint_path": "MyData/results/rna/checkpoints/",
    "flag": "rna_model_survcox"
}
```

### Joint Fusion Configuration (config_joint_train_survcox.json)

```json
{
    "model_name": "resnet50",
    "task": "survival_prediction",
    "num_classes": 1,
    "batch_size": 32,
    "num_epochs": 10,
    "lr_rna": 1e-06,
    "lr_histo": 5e-05,
    "lr_mlp": 0.01,
    "weight_decay": 1e-05,
    "n_layers_to_train": 2,
    "max_patch_per_wsi_train": 100,
    "data_path": "MyData/patches",
    "train_csv_path": "MyData/splits/joint_train.csv",
    "test_csv_path": "MyData/splits/joint_test.csv",
    "checkpoint_path": "MyData/results/joint_fusion/checkpoints/",
    "flag": "jointfusion_model_survcox"
}
```

### Key Parameters Explained

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `task` | Loss function type | `survival_prediction` (Cox) or `survival_bin` (NLL) |
| `num_classes` | Output dimensions | 1 for Cox, 4 for survival bins |
| `n_layers_to_train` | ResNet layers to fine-tune | 2 (last 2 blocks) |
| `aggregator` | Patch aggregation method | `identity`, `attention`, or `mean` |
| `max_patch_per_wsi_train` | Patches sampled per WSI | 100 |
| `pretrained` | Use ImageNet weights | `true` |

---

## Training Pipeline

### Overview

The complete pipeline consists of three stages executed sequentially:

```
Stage 1: Unimodal Training
    WSI Model (ResNet50) ---------+
    RNA Model (MLP) -------------+
                                  |
Stage 2: Feature Extraction       |
    Extract WSI embeddings ------+
    Extract RNA embeddings ------+
    Save model scores -----------+
                                  |
Stage 3: Fusion Training          |
    Early Fusion (MLP) ----------+
    Late Fusion (Cox) -----------+
    Joint Fusion (E2E) ----------+
```

### Stage 1: Unimodal Model Training

Train the WSI and RNA models with 10-fold cross-validation:

```bash
# Train WSI model (approximately 7-8 hours)
python train_with_cv.py \
    --modality ffpe \
    --config config/survcox/config_ffpe_train_survcox.json \
    --n_folds 10

# Train RNA model (approximately 1-2 hours)
python train_with_cv.py \
    --modality rna \
    --config config/survcox/config_rna_train_survcox.json \
    --n_folds 10
```

### Stage 2: Feature and Score Extraction

After training unimodal models, extract features and scores:

```bash
# Save prediction scores
python 1_HistoPathology/3_HistoPath_savescore.py \
    --config config/survcox/config_ffpe_savescore_survcox.json

python 2_GeneExpression/2_GeneExpress_savescore.py \
    --config config/survcox/config_rna_savescore_survcox.json

# Extract feature embeddings (2048-dim vectors)
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

Train all fusion models:

```bash
# Early Fusion (approximately 2 hours)
python train_with_cv.py \
    --modality early \
    --config config/survcox/config_feature_train_survcox.json \
    --n_folds 10

# Late Fusion (approximately 1 minute)
Rscript 4_LateFusion/2_LateFusion.R

# Joint Fusion (approximately 8 hours)
python train_with_cv.py \
    --modality joint \
    --config config/survcox/config_joint_train_survcox.json \
    --n_folds 10
```

### Complete Pipeline Script

For convenience, run the entire pipeline:

```bash
#!/bin/bash
# run_full_pipeline.sh

echo "Stage 1: Training Unimodal Models"
python train_with_cv.py --modality ffpe --config config/survcox/config_ffpe_train_survcox.json --n_folds 10
python train_with_cv.py --modality rna --config config/survcox/config_rna_train_survcox.json --n_folds 10

echo "Stage 2: Extracting Features and Scores"
python 1_HistoPathology/3_HistoPath_savescore.py --config config/survcox/config_ffpe_savescore_survcox.json
python 2_GeneExpression/2_GeneExpress_savescore.py --config config/survcox/config_rna_savescore_survcox.json
python 1_HistoPathology/4_HistoPath_extractfeatures.py --config config/survcox/config_ffpe_extractfeatures_survcox.json
python 2_GeneExpression/3_GeneExpress_extractfeatures.py --config config/survcox/config_rna_extractfeatures_survcox.json
python 3_EarlyFusion/1_Concat2Features.py
python 3_EarlyFusion/split_features.py
python 4_LateFusion/1_MergeScores.py

echo "Stage 3: Training Fusion Models"
python train_with_cv.py --modality early --config config/survcox/config_feature_train_survcox.json --n_folds 10
Rscript 4_LateFusion/2_LateFusion.R
python train_with_cv.py --modality joint --config config/survcox/config_joint_train_survcox.json --n_folds 10

echo "Pipeline Complete"
python analyze_results.py
```

### Estimated Training Times

| Model | Time per Fold | Total (10 folds) |
|-------|---------------|------------------|
| WSI (FFPE) | 45 minutes | 7.5 hours |
| RNA | 10 minutes | 1.5 hours |
| Early Fusion | 12 minutes | 2 hours |
| Late Fusion | N/A | 1 minute |
| Joint Fusion | 50 minutes | 8 hours |
| **Total** | - | **19-20 hours** |

Times measured on NVIDIA RTX 3060 12GB.

---

## Model Architectures

### WSI Model (Histopathology)

```
Input: 224x224 RGB patches (100 patches per WSI)
    |
    v
ResNet50 (pretrained ImageNet, last 2 blocks trainable)
    |
    v
Aggregation Layer (identity/attention/mean)
    |
    v
Output: Risk score (1 value) or survival bin (4 classes)
```

Architecture details:
- **Backbone**: ResNet50 with frozen early layers
- **Fine-tuning**: Last 2 residual blocks (layer3, layer4)
- **Output dimension**: 2048 (before final layer)
- **Aggregation**: Identity (per-patch predictions averaged)

### RNA Model (Gene Expression)

```
Input: 12,778 gene expression values (log2 FPKM)
    |
    v
Linear(12778 -> 4096) + ReLU + Dropout(0.5)
    |
    v
Linear(4096 -> 2048) + ReLU + Dropout(0.5)
    |
    v
Linear(2048 -> 1) (Cox) or Linear(2048 -> 4) (Survival Bin)
    |
    v
Output: Risk score or survival class probabilities
```

### Early Fusion Model

```
Input: 4096 concatenated features (2048 WSI + 2048 RNA)
    |
    v
Linear(4096 -> 2048) + ReLU + Dropout(0.5)
    |
    v
Linear(2048 -> 200) + ReLU + Dropout(0.5)
    |
    v
Linear(200 -> 1)
    |
    v
Output: Risk score
```

### Late Fusion Model

```
Input: 2 scores (WSI risk score + RNA risk score)
    |
    v
Cox Proportional Hazards with L1 Regularization (Lasso)
    |
    v
Output: Combined risk score (weighted sum)
```

The Lasso penalty automatically learns optimal weights for each modality.

### Joint Fusion Model

```
Input: 224x224 patches + 12,778 gene values
    |
    +--------------------+--------------------+
    |                                         |
    v                                         v
ResNet50                                  RNA MLP
(2048-dim)                               (2048-dim)
    |                                         |
    +-----------+-----------------------------+
                |
                v
          Concatenate (4096-dim)
                |
                v
          Linear(4096 -> 1) + Dropout(0.8)
                |
                v
          Output: Risk score
```

Key features:
- End-to-end training with separate learning rates per modality
- High dropout (0.8) on fusion layer to prevent overfitting
- Shared gradients enable cross-modal learning

---

## Evaluation

### Concordance Index (C-Index)

The primary evaluation metric is the Concordance Index:

```
C-Index = P(risk(i) > risk(j) | survival(i) < survival(j))
```

Interpretation:
- **0.5**: Random predictions (no discrimination)
- **0.6-0.7**: Acceptable discrimination
- **0.7-0.8**: Good discrimination
- **0.8+**: Excellent discrimination

### Expected Results

Based on the original paper (TCGA glioma cohort):

| Model | Test C-Index (Expected) |
|-------|-------------------------|
| WSI Only | 0.70 - 0.75 |
| RNA Only | 0.75 - 0.80 |
| Early Fusion | 0.76 - 0.81 |
| Late Fusion | 0.77 - 0.82 |
| Joint Fusion | 0.78 - 0.83 |

### Analyzing Results

After training, analyze results:

```bash
python analyze_results.py
```

This script reads the output CSV files and calculates C-Index for train/val/test splits.

### Output Files

Each trained model generates:

```
MyData/results/{modality}/checkpoints/outputs/{model_name}/
|-- train_output_best.csv    # Best model predictions on train
|-- val_output_best.csv      # Best model predictions on validation
|-- test_output_best.csv     # Best model predictions on test
|-- train_output_last.csv    # Last epoch predictions on train
|-- val_output_last.csv      # Last epoch predictions on validation
|-- test_output_last.csv     # Last epoch predictions on test
+-- model_best.pth           # Saved model weights
```

Output CSV format:
```csv
case,score,survival_months,vital_status,survival_bin
TCGA-XX-XXXX,0.234,45.6,1,3
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `batch_size` in config (try 64 or 32)
- Reduce `max_patch_per_wsi_train` (try 50)
- Close other GPU applications

#### 2. Zero or Negative Survival Times

```
ValueError: Survival times must be positive
```

**Solution:**
Ensure all `survival_months` values are > 0. The preprocessing scripts set a minimum of 0.1 months.

#### 3. Missing Patches

```
FileNotFoundError: No patches found for WSI
```

**Solution:**
Verify patch extraction completed:
```bash
ls MyData/patches/TCGA-XX-XXXX-01Z-00-DX1/
```
Each folder should contain `loc.txt` and `*_patch_*.png` files.

#### 4. R Package Errors

```
Error in library(survcomp): there is no package called 'survcomp'
```

**Solution:**
Install Bioconductor packages:
```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("survcomp")
```

#### 5. Cross-Validation Fold Failures

If a fold fails during CV training, check:
1. Sufficient samples in each survival bin
2. No empty batches (reduce batch size)
3. GPU memory (first fold may work, later folds fail)

### Logging and Debugging

Enable verbose logging:
```bash
python train_with_cv.py --modality ffpe --config config.json --n_folds 10 2>&1 | tee training.log
```

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

---

## References

### Original Paper

Steyaert, S., Qiu, Y.L., Zheng, Y., Mukherjee, P., Vogel, H., and Gevaert, O. (2023). 
"Multimodal data fusion of adult and pediatric brain tumors with deep learning." 
*Communications Medicine*, 3, 124.

DOI: https://doi.org/10.1038/s43856-023-00349-y

### Original Repository

https://github.com/gevaertlab/MultiModalBrainSurvival

### TCGA Data

- GBM (Glioblastoma): https://portal.gdc.cancer.gov/projects/TCGA-GBM
- LGG (Lower Grade Glioma): https://portal.gdc.cancer.gov/projects/TCGA-LGG

### Key Dependencies

- PyTorch: https://pytorch.org/
- OpenSlide: https://openslide.org/
- scikit-survival: https://scikit-survival.readthedocs.io/
- lifelines: https://lifelines.readthedocs.io/

---

## License

This project is for academic research purposes. Please cite the original paper if using this code.

## Contact

For questions regarding this implementation, please open an issue on the GitHub repository.
