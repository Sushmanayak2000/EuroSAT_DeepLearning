# EuroSAT Land-Use and Land-Cover Classification

## Project Overview

This project investigates deep learning approaches for land-use and land-cover classification using the EuroSAT RGB satellite image dataset.

The project is based on the research paper:

Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226.

The main objective is to reproduce a pretrained ResNet-50 baseline and investigate two extensions: the effect of training-data availability and the performance of a smaller neural network architecture.

## Project Objectives

- Reproduce a ResNet-50 classification baseline using ImageNet-pretrained weights.
- Investigate how the amount of labelled training data affects classification performance.
- Evaluate a smaller ImageNet-pretrained ResNet-18 architecture.
- Compare experimental results with the published EuroSAT results.
- Analyse classification performance using accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset

The project uses the RGB version of the EuroSAT dataset, based on Sentinel-2 satellite imagery.

### Dataset details

| Property | Description |
|---|---|
| Dataset | EuroSAT RGB |
| Number of images | 27,000 |
| Number of classes | 10 |
| Image dimensions | 64 × 64 pixels |
| Image channels | 3 (RGB) |
| Dataset licence | MIT Licence |

### Classes

The dataset contains ten land-use and land-cover classes:

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

### Dataset source

The dataset is available from:

[EuroSAT RGB Dataset – Zenodo](https://zenodo.org/records/7711810)

The dataset is not included in this repository and must be downloaded separately.

### Data preparation

A stratified train-validation-test split was used:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 19,440 | 72% |
| Validation | 2,160 | 8% |
| Testing | 5,400 | 20% |
| **Total** | **27,000** | **100%** |

The same saved split was used across the baseline and subsequent experiments to maintain consistency.

Images were loaded as RGB, converted into PyTorch tensors, and normalised using ImageNet mean and standard deviation. The original 64 × 64 image dimensions were retained.

## Models and Experimental Contributions

The project consists of a shared baseline and two individual experimental contributions.

### Shared Baseline – ImageNet-pretrained ResNet-50

The baseline uses ResNet-50 pretrained on ImageNet. Its final fully connected layer was replaced with a new classification layer containing ten output units.

A two-stage training strategy was used:

- **Stage 1:** Train the newly added classification layer while keeping the pretrained backbone frozen.
- **Stage 2:** Fine-tune the entire network using a smaller learning rate.

The main experimental configuration was:

| Parameter | Value |
|---|---|
| Architecture | ResNet-50 |
| Pretraining | ImageNet |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Input resolution | 64 × 64 |
| Head-only training | 3 epochs |
| Full-network fine-tuning | 7 epochs |
| Random seeds | 42, 43, 44 |

**Baseline result:** The ResNet-50 achieved a mean test accuracy of **96.64% ± 0.27 percentage points**, averaged across three random seeds.

### Part A – Effect of Pretraining with Limited Training Data (Sushma)

Part A investigates how the amount of available labelled training data affects the performance of an ImageNet-pretrained ResNet-50.

Five proportions of the original training partition were evaluated:

- 10% – 1,944 images
- 30% – 5,832 images
- 50% – 9,720 images
- 70% – 13,608 images
- 90% – 17,496 images

The validation and test sets remained fixed across the experiments. Each training-data proportion was evaluated using three random seeds: 42, 43, and 44.

**Part A results:**

| Training data | Mean test accuracy | Standard deviation |
|---|---:|---:|
| 10% | 90.07% | 0.43 pp |
| 30% | 94.10% | 0.30 pp |
| 50% | 95.02% | 0.32 pp |
| 70% | 95.91% | 0.27 pp |
| 90% | 96.31% | 0.17 pp |

The results show that the pretrained ResNet-50 achieved 90.07% mean test accuracy with only 10% of the original training partition. Accuracy increased to 96.31% when 90% of the training partition was used.

The learning curve was also compared with the scratch-trained ResNet-50 results reported in Table II of the original paper. This comparison is indicative rather than a controlled experiment isolating the effect of pretraining, because the experimental settings differ.

### Part B – Evaluating a Smaller Network (Ishika Thakur)

Part B investigates whether a smaller architecture can achieve competitive classification performance while using fewer model parameters.

An ImageNet-pretrained ResNet-18 was evaluated using the same dataset split, input resolution, and main training configuration as the ResNet-50 baseline.

**Part B results:**

| Metric | ResNet-18 |
|---|---:|
| Mean test accuracy | 95.74% |
| Standard deviation | 0.02 pp |
| Macro precision | 95.65% |
| Macro recall | 95.52% |
| Macro F1-score | 95.57% |
| Number of parameters | 11.18 million |
| Mean training time | Approximately 2.83 minutes |

The ResNet-18 achieved a mean test accuracy approximately 0.90 percentage points below the ResNet-50 baseline.

Its parameter count was approximately 11.18 million, compared with 23.53 million for ResNet-50, representing a reduction of approximately 52.5%.

The observed mean accuracy difference meets the project's descriptive one-percentage-point comparison target. However, the limited number of random seeds does not establish statistical non-inferiority.

## Evaluation and Error Analysis

The models were evaluated using:

- Overall test accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Confusion matrices
- Class-wise classification reports

The baseline and Part A experiments include error analysis to examine misclassified images and differences in class-level performance.

The Part A learning curve compares the pretrained ResNet-50 results with the scratch-trained learning curve reported in the original paper.

## Repository Structure

The repository is organised into the following main directories and files:

```text
EuroSAT_DeepLearning/
│
├── Baseline/
│   └── EuroSAT_ResNet50_Baseline.ipynb
│
├── Part A/
│   └── EuroSAT_Part_A_Label_Efficiency.ipynb
│
├── Part B/
│   └── Part B experiment notebook and results
│
├── baseline_checks/
│
├── figures/
│   ├── Class distribution.png
│   └── Representative EuroSAT image samples.png
│
├── report.md
├── README.md
└── requirements.txt
```

The experiment folders contain the corresponding notebooks and supporting results. Additional outputs include evaluation metrics, training histories, classification reports, confusion matrices, and plots.

Dataset files and trained model checkpoints are not included in the repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EuroSAT_DeepLearning
```

Replace the placeholder with the URL of your GitHub repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the EuroSAT RGB dataset from [Zenodo](https://zenodo.org/records/7711810).

Extract the dataset and configure the dataset path in the relevant notebook.

### 5. Run the experiments

The project notebooks are organised into three parts:

1. **Baseline:** Run `Baseline/EuroSAT_ResNet50_Baseline.ipynb` to reproduce the shared ResNet-50 baseline.
2. **Part A:** Run `Part A/EuroSAT_Part_A_Label_Efficiency.ipynb` to evaluate different training-data proportions.
3. **Part B:** Open the experiment notebook in the `Part B/` directory to evaluate ResNet-18.

Run the notebook cells sequentially, following the dataset preparation and experiment instructions in each notebook.

## Hardware and Software

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pillow
- tqdm
- Google Colab or a local Jupyter environment

GPU acceleration is recommended for model training. The experiments were conducted using GPU-enabled environments.

## Reproducibility

The experiments use random seeds 42, 43, and 44.

The dataset split, model configuration, training procedure, and evaluation settings are documented in the respective notebooks and project report.

Mean performance and standard deviation are reported across the three random seeds.

## Reference

Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226.

https://doi.org/10.1109/JSTARS.2019.2918242
