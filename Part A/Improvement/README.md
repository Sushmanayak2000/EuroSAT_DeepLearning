# Part A – Effect of Pretraining with Limited Training Data

## 1. Overview

This improvement investigates how the amount of labelled training data affects EuroSAT land-use and land-cover classification performance using transfer learning.

The objective is to evaluate whether an ImageNet-pretrained ResNet-50 can achieve strong classification performance with different amounts of labelled training data, reducing the dependence on large labelled datasets.

## 2. Proposed Improvement

The original EuroSAT paper reports results for both scratch-trained and pretrained models. Table II presents the learning curve for ResNet-50 trained from scratch, while Table III reports results for ImageNet-pretrained models.

In this project, we investigate the performance of an **ImageNet-pretrained ResNet-50** across different training-data proportions.

Instead of training the network entirely from random initialisation, the model benefits from previously learned visual features and is adapted to the EuroSAT dataset through transfer learning.

The experiment evaluates five training-data proportions:

- 10%
- 30%
- 50%
- 70%
- 90%

Each experiment is repeated using three random seeds (42, 43 and 44) to examine the consistency of the results.

This results in **15 experimental runs**.

## 3. Experimental Setup

- **Dataset:** EuroSAT RGB
- **Total images:** 27,000
- **Number of classes:** 10
- **Model architecture:** ResNet-50
- **Initialisation:** ImageNet-pretrained weights
- **Training strategy:** Two-stage transfer learning (classification-head training followed by full-network fine-tuning)
- **Training proportions:** 10%, 30%, 50%, 70% and 90% of the original training partition
- **Random seeds:** 42, 43 and 44

The dataset was divided into fixed training (72%), validation (8%) and test (20%) partitions. The same validation and test partitions were maintained across all experiments.

## 4. Results and Comparison

The following table presents the mean test accuracy and standard deviation across three random seeds.

| Training proportion | Mean test accuracy ± SD |
|---|---:|
| 10% | 90.07% ± 0.43 pp |
| 30% | 94.10% ± 0.30 pp |
| 50% | 95.02% ± 0.32 pp |
| 70% | 95.91% ± 0.27 pp |
| 90% | 96.31% ± 0.17 pp |

### Learning Curve

The following figure compares our ImageNet-pretrained ResNet-50 with the scratch-trained ResNet-50 learning curve reported in Table II of the original EuroSAT paper.

![EuroSAT Learning Curve](part_a_learning_curve.png)

The results show that classification accuracy improves as more labelled training data becomes available.

The pretrained model achieves **90.07% mean test accuracy using only 10% of the original training partition**, increasing to **96.31% using 90% of the training partition**.

The comparison with the original paper is indicative rather than a controlled measurement of the effect of pretraining alone, because the experimental setups and data-splitting procedures differ.

## 5. Key Findings

- The ImageNet-pretrained ResNet-50 achieved 90.07% mean test accuracy using 10% of the training partition.
- Classification performance improved as more labelled training data became available.
- The highest mean test accuracy was 96.31%, achieved using 90% of the training partition.
- Performance gains became smaller at higher training-data proportions.
- Repeating the experiments with three random seeds provided an assessment of result consistency.

## 6. Conclusion

This improvement investigates the effectiveness of transfer learning for data-efficient EuroSAT image classification.

The experiments demonstrate that an ImageNet-pretrained ResNet-50 can achieve strong classification performance even with reduced labelled training data. Accuracy increased from 90.07% with 10% of the training partition to 96.31% with 90%.

These findings highlight the potential of ImageNet pretraining for satellite image classification when labelled training data is limited.
