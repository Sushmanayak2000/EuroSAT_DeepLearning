# 1. Original Paper

## Paper

- **Authors:** Patrick Helber, Benjamin Bischke, Andreas Dengel and Damian Borth
- **Title:** EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification
- **Year:** 2019
- **Link:** https://zenodo.org/records/7711810

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

## 3 · Reproducing the baseline (part of 15 %, and 25 % implementation) 
**The recipe:** read the paper (or its code) like a recipe. One row per ingredient; leave nothing empty: if the paper does not say it, write "not stated" and what you chose.  
| | The paper says | Where (page, table, code) | What we use | |---|---|---|---| | Data: version, split | EuroSAT RGB dataset, using a class-wise 80/20 training-test split | EuroSAT paper, Section IV-A, page 6; Table 3 | Official EuroSAT RGB dataset (27,000 images, 10 classes). Stratified 70/10/20 train-validation-test split using seed 42 | | Model: layers, sizes, pretrained? | ResNet-50 CNN pretrained on ILSVRC-2012 (ImageNet)|EuroSAT paper, Section IV, page 6; Table 3 | ImageNet-pretrained ResNet-50 with the final fully connected layer replaced to classify 10 EuroSAT classes| | Training: optimiser, learning rate, schedule | First train the last layer with LR 0.01, then fine-tune the entire network using a low LR between 0.001 and 0.0001 | EuroSAT paper, Section IV-A, page 6 | AdamW optimizer. Stage 1: LR 0.01. Stage 2: LR 0.0001. Two-stage transfer learning: frozen backbone followed by full-network fine-tuning|  | Training: epochs, batch size | Not stated in the paper's described fine-tuning procedure|EuroSAT paper, Section IV-A, page 6; training epochs and batch size not specified there | 3 epochs for classification-head training, followed by 7 epochs for full-network fine-tuning. Batch size 64| | Augmentation and preprocessing | The paper uses RGB images created from Sentinel-2 red, green, and blue bands. Further preprocessing and augmentation details are not specified in the cited baseline description | EuroSAT paper, Section IV-A, page 6 | 64×64 RGB images. Convert images to tensors and normalize using ImageNet mean and standard deviation. No data augmentation in our current pipeline | | Metric, and how it was averaged | Overall classification accuracy; the paper reports accuracy for the 80/20 split | EuroSAT paper, Table III, page 6; ResNet-50 EuroSAT result: 98.57% | Accuracy and other classification metrics will be evaluated on the held-out test set. We use three random seeds (42, 43, 44) and report mean ± standard deviation| 
- **Setup:** [ The experiment was implemented using PyTorch 2.11.0 and torchvision 0.26.0 in Google Colab. Training was performed using a Tesla T4 GPU. The ResNet-50 model was initialized with ImageNet-pretrained weights and adapted for 10 EuroSAT classes]
- **What the paper did not say,** and what you chose instead: [The paper reports a classification accuracy of 98.57% for ResNet-50 using an 80/20 training-test split. Our current baseline uses a stratified 70/10/20 train-validation-test split and selects the best model based on validation accuracy. Differences in the data split, training configuration, and evaluation procedure may contribute to differences in performance. The paper's exact implementation details are not fully specified, so the results should be interpreted as a reproduction attempt using a documented experimental setup rather than an exact reproduction]
| Paper | Ours (mean ± spread, n seeds) | |---|---|---| | [Classification accuracy (%)] | [98.57] | [96.73 ± 0.08] |
[The paper reports a classification accuracy of 98.57% for ResNet-50 using an 80/20 training-test split. Our current baseline uses a stratified 70/10/20 train-validation-test split and selects the best model based on validation accuracy. Differences in the data split, training configuration, and evaluation procedure may contribute to differences in performance. The paper's exact implementation details are not fully specified, so the results should be interpreted as a reproduction attempt using a documented experimental setup rather than an exact reproduction]

