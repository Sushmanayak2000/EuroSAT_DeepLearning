# EuroSAT Land-Use and Land-Cover Classification

## Project Overview

This project reproduces a deep learning baseline for land-use and land-cover classification using the EuroSAT RGB dataset.

The project is based on the paper:

Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226.

## Objectives

- Reproduce the ResNet-50 baseline described in the paper.
- Evaluate the model on the EuroSAT RGB dataset.
- Investigate an improvement to the baseline model.
- Compare experimental results with the published results.

## Dataset

The EuroSAT dataset contains 27,000 RGB satellite images across 10 land-use and land-cover classes.

- Image size: 64 × 64 pixels
- Number of classes: 10
- Dataset: EuroSAT RGB

Dataset source: [EuroSAT on Zenodo](https://zenodo.org/records/7711810)

## Model

The baseline model uses ResNet-50 pretrained on ImageNet.

The final fully connected layer is replaced to classify the images into 10 EuroSAT classes.

## Work Distribution

- **Part A – How Many Labels Do We Need?**: Sushma
- **Part B – Improvement to the Baseline**: Ishika Thakur

Both team members collaborate on the baseline reproduction, dataset preparation, and final report.

## Experimental Setup

- Training-validation-test split: 72% / 8% / 20%
- Optimizer: AdamW
- Batch size: 64
- Training: two-stage procedure
- Random seeds: 42, 43, and 44
- Framework: PyTorch
- Environment: Google Colab

## Repository Contents

- `EuroSAT_ResNet50_Baseline.ipynb` — notebook for the baseline experiment
- `report.md` — project report

## Authors

Sushma and Ishika Thakur

## Reference

Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE JSTARS, 12(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242
