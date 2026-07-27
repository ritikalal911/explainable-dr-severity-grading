# Explainable Diabetic Retinopathy Severity Grading

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-APTOS%202019-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-oriented deep learning project for grading **diabetic retinopathy (DR) severity** from retinal fundus images. The project currently provides a reproducible exploratory data analysis and a controlled comparison of five image-preprocessing pipelines using an ImageNet-pretrained ResNet50.

The long-term goal is to build a severity-grading workflow that is not only accurate, but also transparent about the image regions and evidence influencing its predictions.

> **Current project stage:** Exploratory data analysis and preprocessing experiments are complete. Explainability methods are listed in the roadmap and should not yet be considered implemented unless their corresponding notebooks or source files are added.

> **Medical disclaimer:** This repository is intended for research and education only. It is not a validated medical device and must not be used for diagnosis, treatment decisions, or clinical screening without appropriate clinical validation and regulatory review.

---

## Table of Contents

- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing Study](#preprocessing-study)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Main Findings](#main-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Explainability Roadmap](#explainability-roadmap)
- [License](#license)

---

## Project Objectives

This project is designed to:

1. Examine the quality, structure, and class distribution of the APTOS 2019 retinal-image dataset.
2. Create one fixed, stratified train-validation-test split for fair comparison across experiments.
3. Compare multiple preprocessing techniques while keeping the model and training settings unchanged.
4. Evaluate performance using metrics suitable for an imbalanced, ordinal classification problem.
5. Identify a preprocessing baseline for later model-development and explainability studies.
6. Develop future visual explanations that show which retinal regions contribute to each predicted severity grade.

---

## Dataset

The project uses the **APTOS 2019 Blindness Detection** dataset, containing **3,662 labelled retinal fundus images**.

Each image is assigned one of five ordered diabetic-retinopathy severity grades:

| Label | Class | Meaning | Images | Dataset share |
|---:|---|---|---:|---:|
| 0 | No DR | No visible diabetic retinopathy | 1,805 | 49.3% |
| 1 | Mild | Mild diabetic retinopathy | 370 | 10.1% |
| 2 | Moderate | Moderate diabetic retinopathy | 999 | 27.3% |
| 3 | Severe | Severe diabetic retinopathy | 193 | 5.3% |
| 4 | Proliferative | Proliferative diabetic retinopathy | 295 | 8.1% |

The labels are **ordinal**: an error between adjacent grades is less severe than an error between distant grades. For that reason, **Quadratic Weighted Kappa (QWK)** is used as the primary model-selection metric.

### Fixed stratified split

A single split is created with random seed `42` and reused throughout the project:

| Split | Images | Approximate share | Purpose |
|---|---:|---:|---|
| Training | 2,563 | 70% | Model fitting |
| Validation | 549 | 15% | Monitoring and checkpoint selection |
| Test | 550 | 15% | Final held-out evaluation |

The split is stored at:

```text
data/splits/aptos_seed42.csv
```

---

## Exploratory Data Analysis

The EDA notebook checks data integrity, class imbalance, image dimensions, brightness, contrast, aspect ratio, and black-border content.

### Key EDA findings

- All **3,662 labels** have corresponding readable image files.
- No missing or unreadable images were detected in the analysed dataset copy.
- The dataset is strongly imbalanced: **No DR** represents almost half of all images, while **Severe** represents only about 5%.
- Images occur in **17 different resolutions**.
- Raw image dimensions range from **474 × 358** to **4,288 × 2,848** pixels.
- Brightness, contrast, framing, and dark-border proportions vary substantially.
- Brightness and contrast distributions overlap heavily across classes, so they do not provide a simple shortcut for separating severity grades.
- Cropping the retinal field and resizing images to a fixed input size are necessary before modelling.

The full analysis is available in:

- [`01_eda.ipynb`](01_eda.ipynb)
- [`EDA_REPORT.md`](EDA_REPORT.md)

---

## Preprocessing Study

Five preprocessing pipelines are compared under a controlled experimental design. Every pipeline first crops the retinal field, resizes the image to `224 × 224`, converts it to a tensor, and applies ImageNet normalisation.

| Pipeline | Enhancement | Purpose |
|---|---|---|
| `A_baseline` | No enhancement | Preserves the original cropped image and acts as the control condition |
| `B_clahe` | CLAHE on LAB luminance | Improves local contrast while preserving colour channels |
| `C_gamma` | Gamma correction, γ = 1.4 | Brightens darker regions using a smooth nonlinear transformation |
| `D_histeq` | Global histogram equalisation on luminance | Strongly redistributes global brightness and contrast |
| `E_green_clahe` | CLAHE on green channel, replicated to RGB | Emphasises vessels and green-channel detail but removes original colour variation |

Visual comparisons include:

- original and processed images;
- absolute difference maps;
- RGB intensity histograms;
- Structural Similarity Index Measure (SSIM);
- brightness, contrast, and entropy comparisons;
- preprocessing-speed benchmarks.

The complete experiment is available in:

- [`02_preprocessing.ipynb`](02_preprocessing.ipynb)
- [`PREPROCESSING_REPORT.md`](PREPROCESSING_REPORT.md)

---

## Experimental Setup

Only the preprocessing method changes between runs. The following settings remain fixed:

| Setting | Value |
|---|---|
| Architecture | ResNet50 |
| Initial weights | ImageNet pretrained |
| Input size | 224 × 224 |
| Loss | Cross-entropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | `1e-4` |
| Batch size | 64 |
| Maximum epochs | 30 |
| Early-stopping patience | 5 |
| LR-reduction patience | 2 |
| LR-reduction factor | 0.1 |
| Checkpoint criterion | Highest validation QWK |

No weighted loss, weighted sampler, or targeted minority-class augmentation is used in this preprocessing experiment. This keeps the comparison focused on preprocessing, but it also means the study does not attempt to solve class imbalance.

### Evaluation metrics

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1 score
- Quadratic Weighted Kappa
- Per-class recall
- Confusion matrix
- Cross-entropy loss

---

## Results

### Validation performance

| Pipeline | Accuracy | Balanced accuracy | Macro F1 | QWK | Loss |
|---|---:|---:|---:|---:|---:|
| `B_clahe` | **0.838** | **0.656** | **0.686** | **0.884** | 0.762 |
| `C_gamma` | 0.818 | 0.614 | 0.637 | 0.881 | 0.565 |
| `E_green_clahe` | 0.823 | 0.632 | 0.663 | 0.872 | 0.624 |
| `A_baseline` | 0.812 | 0.647 | 0.657 | 0.854 | **0.511** |
| `D_histeq` | 0.772 | 0.581 | 0.579 | 0.850 | 0.611 |

### Held-out test performance

| Pipeline | Accuracy | Balanced accuracy | Macro F1 | QWK | Loss |
|---|---:|---:|---:|---:|---:|
| `B_clahe` | **0.827** | 0.640 | 0.658 | **0.883** | 0.690 |
| `A_baseline` | 0.825 | **0.655** | **0.666** | 0.881 | **0.466** |
| `E_green_clahe` | 0.825 | 0.645 | 0.664 | 0.879 | 0.576 |
| `C_gamma` | 0.813 | 0.638 | 0.656 | 0.874 | 0.530 |
| `D_histeq` | 0.782 | 0.600 | 0.598 | 0.850 | 0.579 |

### Selected preprocessing pipeline

The selection rule was defined before examining the held-out test results:

1. Select the pipeline with the highest validation QWK.
2. Use validation macro F1 as the secondary criterion.

Under this rule, the selected pipeline is:

```text
B_clahe
```

| Selected-pipeline metric | Score |
|---|---:|
| Validation QWK | 0.8842 |
| Validation macro F1 | 0.6858 |
| Test QWK | 0.8828 |
| Test macro F1 | 0.6579 |

The validation-to-test QWK gap is only `0.0014`, indicating stable performance across the internal validation and test splits.

---

## Main Findings

- **CLAHE on LAB luminance is the strongest pipeline under the predeclared QWK-based selection rule.**
- `B_clahe` achieves the highest validation and test QWK.
- The unenhanced baseline remains a strong practical alternative and achieves the best test macro F1, balanced accuracy, macro recall, and loss.
- Gamma correction preserves structure well and performs comparatively strongly for some advanced disease classes.
- Global histogram equalisation changes images most aggressively and produces the weakest overall results.
- No preprocessing method is best for every severity class.
- Severe-class recall remains low across all pipelines.
- Training and validation loss curves show overfitting for all five experiments.

The recommended conclusion is therefore not that CLAHE is universally superior, but that it is the best preprocessing baseline **for this dataset, split, architecture, seed, and primary metric**.

---

## Repository Structure

```text
explainable-dr-severity-grading/
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── EDA_REPORT.md
├── PREPROCESSING_REPORT.md
├── LICENSE
├── .gitignore
│
├── aptos2019/                         # Not committed; create locally
│   ├── dataset.csv
│   └── images/
│
├── data/
│   └── splits/
│       └── aptos_seed42.csv
│
├── artifacts/
│   ├── checkpoints/
│   └── results/
│
└── studies/
    ├── 01_eda/
    │   └── figures/
    └── 02_preprocessing/
        ├── figures/
        ├── processed_examples/
        └── best_preprocessing_baseline_checkpoint.pth
```

> **Checkpoint naming note:** The current preprocessing notebook may copy the selected checkpoint to `best_preprocessing_baseline_checkpoint.pth`. Despite that filename, the selected pipeline is `B_clahe`, not `A_baseline`. Renaming it to `best_B_clahe_checkpoint.pth` is recommended for clarity.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ritikalal911/explainable-dr-severity-grading.git
cd explainable-dr-severity-grading
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install torch torchvision opencv-python-headless numpy pandas matplotlib seaborn scikit-learn scikit-image tqdm pyyaml jupyter
```

A CUDA-capable GPU is recommended for model training. The reported experiments were executed with CUDA acceleration.

---

## Dataset Setup

Download the APTOS 2019 Blindness Detection data from Kaggle and arrange it as follows:

```text
aptos2019/
├── dataset.csv
└── images/
    ├── 000c1434d8d7.png
    ├── 001639a390f0.png
    └── ...
```

The labels CSV must contain:

| Column | Description |
|---|---|
| `id_code` | Image filename without the `.png` extension |
| `diagnosis` | Integer severity label from 0 to 4 |

The dataset is intentionally excluded from Git because of its size and external distribution terms.

---

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in this order:

### Step 1 — Exploratory data analysis

Open and run:

```text
01_eda.ipynb
```

This notebook:

- verifies labels and image paths;
- creates or reuses the fixed stratified split;
- analyses class balance and image quality;
- saves EDA figures under `studies/01_eda/figures/`.

### Step 2 — Preprocessing comparison

Open and run:

```text
02_preprocessing.ipynb
```

This notebook:

- loads the shared split;
- defines pipelines A–E;
- creates before-and-after visual evidence;
- trains missing runs or reuses completed checkpoints and histories;
- evaluates validation and held-out test performance;
- selects the best pipeline using validation QWK;
- saves figures, metrics, histories, and checkpoints.

---

## Reproducibility Notes

- Keep random seed `42` unchanged when reproducing the reported experiment.
- Run the EDA notebook first so the shared split exists.
- Do not regenerate the split separately for each pipeline.
- Keep the model architecture, hyperparameters, augmentations, and evaluation code fixed while comparing preprocessing methods.
- Existing checkpoints should only be reused when the code, data split, and experimental settings are unchanged.
- Report the best validation checkpoint rather than the final training epoch.
- Do not use test performance to revise the selection rule after results are observed.

---

## Limitations

The current results should be interpreted carefully:

- The dataset is imbalanced, especially for the Severe class.
- The preprocessing experiment uses only one architecture: ResNet50.
- Results are based on one random seed and one internal train-validation-test split.
- No class-weighted loss, focal loss, balanced sampler, or targeted minority-class augmentation is used.
- All models show evidence of overfitting.
- Severe recall remains inadequate for a safety-critical screening system.
- The study does not include probability calibration or uncertainty estimation.
- Validation and test data come from the same dataset distribution.
- No external hospital, camera, population, or country-level validation has been performed.
- Model explanations have not yet been clinically validated.

---

## Explainability Roadmap

Planned next stages include:

- Grad-CAM or Grad-CAM++ heatmaps for class-specific visual explanations;
- comparison of explanation maps across severity grades;
- confidence and uncertainty estimation;
- review of correctly classified and misclassified cases;
- targeted analysis of Severe and Proliferative false negatives;
- checks for attention on retinal lesions rather than borders, glare, or camera artefacts;
- quantitative explanation evaluation where suitable annotations are available;
- ordinal-loss experiments;
- class-aware training and stronger regularisation;
- repeated runs, cross-validation, and external-dataset evaluation.

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright © 2026 **Ritika Lal**
