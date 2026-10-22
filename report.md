# 1. Original Paper

## Paper

- **Authors:** Patrick Helber, Benjamin Bischke, Andreas Dengel and Damian Borth
- **Title:** EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification
- **Year:** 2019
- **Link:** https://arxiv.org/abs/1709.00029

## Task

The network receives a 64 × 64 RGB satellite image and predicts
one of 10 land-use or land-cover classes, such as Forest, River
or Residential.

This is a supervised, single-label, multiclass classification task.

## Method

The authors use ResNet-50, a convolutional neural network with
50 layers. Its residual connections allow information to bypass
some layers, helping a deep network train effectively.

The model is first pretrained on ImageNet (ILSVRC-2012), then
fine-tuned on EuroSAT. This transfers previously learned visual
features to satellite-image classification.

The authors first train the final classification layer with a
learning rate of 0.01. They then fine-tune the entire network
using a learning rate between 0.001 and 0.0001.

## Main Result

Table III reports the following EuroSAT RGB results using
a class-wise 80% training / 20% test split:

| Model | Pretraining | Test accuracy |
|---|---|---:|
| ResNet-50 | ImageNet | **98.57%** |
| GoogLeNet | ImageNet | 98.18% |

Our main reproduction target is the **98.57% accuracy achieved
by fine-tuned ResNet-50**, reported in Table III.
