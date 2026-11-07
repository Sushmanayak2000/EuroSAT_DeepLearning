# EuroSAT Land-Use and Land-Cover Classification

## Project Overview

This project reproduces a deep learning baseline for land-use and land-cover classification using the EuroSAT RGB dataset.

The project is based on the following research paper:

Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226.

The project aims to reproduce the ResNet-50 baseline presented in the paper, evaluate its performance on the EuroSAT RGB dataset, and investigate potential improvements to the baseline model.

## Objectives

- Reproduce the ResNet-50 baseline described in the original paper.
- Evaluate the model on the EuroSAT RGB dataset.
- Investigate an improvement to the baseline model.
- Compare experimental results with the published results.
- Analyze the model's performance across different land-use and land-cover classes.

## Dataset

The EuroSAT dataset is a land-use and land-cover classification dataset based on Sentinel-2 satellite imagery.

The RGB version contains 27,000 images distributed across 10 classes.

### Dataset Details

- **Dataset:** EuroSAT RGB
- **Number of images:** 27,000
- **Number of classes:** 10
- **Image size:** 64 × 64 pixels
- **Image format:** RGB

### Classes

The dataset contains the following land-use and land-cover classes:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

### Dataset Source

The dataset can be downloaded from:

[EuroSAT RGB Dataset — Zenodo](https://zenodo.org/records/7711810)

The dataset is not included in this repository.

## Model

The baseline model uses ResNet-50 pretrained on ImageNet.

The original fully connected classification layer is replaced with a new fully connected layer containing 10 output units, corresponding to the 10 EuroSAT classes.

The baseline follows a two-stage training procedure:

- **Stage 1:** Train the newly added classification layer while keeping the pretrained feature extraction layers frozen.
- **Stage 2:** Fine-tune the entire network using a smaller learning rate to adapt the pretrained features to the EuroSAT dataset.

This approach allows the model to first learn the new classification task and then adapt the pretrained network to the satellite imagery.

## Experimental Setup

The baseline experiment uses the following configuration:

| Parameter | Value |
|---|---|
| Model | ResNet-50 |
| Pretraining | ImageNet |
| Dataset | EuroSAT RGB |
| Number of classes | 10 |
| Image size | 64 × 64 |
| Training-validation-test split | 72% / 8% / 20% |
| Optimizer | AdamW |
| Batch size | 64 |
| Training procedure | Two-stage training |
| Random seeds | 42, 43, 44 |
| Framework | PyTorch |
| Environment | Google Colab |

The dataset is split into training, validation, and test sets.

The validation set is used for model selection and monitoring during training. The test set is reserved for final evaluation.

The experiment is repeated using three random seeds to assess the reproducibility and stability of the results.

## Work Distribution

The project is divided into two parts:

- **Part A – How Many Labels Do We Need?**: Sushma
- **Part B – Evaluating a Smaller Network**: Ishika Thakur

Both team members collaborate on baseline reproduction, dataset preparation, and the final report.

## Repository Contents

The repository contains the following project files:

- `EuroSAT_ResNet50_Baseline.ipynb` — notebook for dataset preparation, baseline training, and evaluation.
- `report.md` — project report.
- `requirements.txt` — Python dependencies required to run the project.

Additional files may be generated during training and evaluation. Dataset files and trained model checkpoints are not included in the repository.

## How to Run

### 1. Clone the Repository

Clone the repository from GitHub:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd <YOUR_REPOSITORY_NAME>
```

Replace the placeholders with your actual GitHub repository URL and repository name.

### 2. Create a Python Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the EuroSAT RGB dataset from [Zenodo](https://zenodo.org/records/7711810).

Extract the dataset and configure its path in the notebook according to the dataset preparation instructions.

### 5. Run the Notebook

Open `EuroSAT_ResNet50_Baseline.ipynb` using Jupyter Notebook or Google Colab.

Run the notebook cells sequentially to:

- Prepare and validate the dataset.
- Create the training, validation, and test splits.
- Initialize the pretrained ResNet-50 model.
- Train the baseline model.
- Evaluate the model.
- Save the experimental results.

### 6. Hardware Requirements

A GPU is recommended for model training.

The notebook can be executed using Google Colab with GPU acceleration enabled.

Training time depends on the hardware, dataset setup, and training configuration.

## Reproducibility

The baseline experiment uses random seeds 42, 43, and 44.

The dataset split, training configuration, and evaluation procedure are documented in the notebook and project report.

## Results

The experimental results and comparison with the published baseline will be presented in the project report.

## Reference

Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226.

https://doi.org/10.1109/JSTARS.2019.2918242
