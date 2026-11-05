# Part A – Evaluating Label Efficiency

## 1. Overview

This part of the project investigates the effect of training data availability on EuroSAT land-use and land-cover classification.

The main objective is to evaluate how the amount of labelled training data affects the classification performance of an ImageNet-pretrained ResNet-50 model.

The experiment also compares our results with the ResNet-50 learning curve reported in the original EuroSAT research paper.

## 2. Individual Contribution

**Student:** Sushma  
**Contribution:** Part A – Effect of Pretraining with Limited Training Data

The individual improvement investigates whether ImageNet pretraining can help a ResNet-50 model achieve strong classification performance when only a limited amount of labelled EuroSAT training data is available.

The original paper's learning curve provides a reference for comparing the effect of training data size. Our experiments use an ImageNet-pretrained ResNet-50 and evaluate its performance across different training-data proportions.

### Research question

How does the amount of labelled training data affect the classification performance of an ImageNet-pretrained ResNet-50 on the EuroSAT dataset?

### Hypothesis

We hypothesise that ImageNet-pretrained ResNet-50 will achieve higher classification accuracy with limited EuroSAT training data because it can benefit from visual features learned during pretraining. We also expect the performance difference to become smaller as more training data becomes available.

## 3. Dataset

The experiments use the EuroSAT RGB dataset.

| Property | Description |
|---|---|
| Dataset | EuroSAT RGB |
| Total images | 27,000 |
| Number of classes | 10 |
| Image dimensions | 64 × 64 pixels |
| Input channels | 3 (RGB) |

### Dataset split

A fixed stratified split was used across the experiments.

| Partition | Images | Percentage |
|---|---:|---:|
| Training | 19,440 | 72% |
| Validation | 2,160 | 8% |
| Test | 5,400 | 20% |
| **Total** | **27,000** | **100%** |

The same validation and test partitions were maintained across all training-data experiments to ensure consistent evaluation.

The shared dataset split and class mapping files are available in the `Baseline` folder.

## 4. Experimental Methodology

### Model architecture

- Architecture: ResNet-50
- Initialisation: ImageNet-pretrained weights
- Framework: PyTorch
- Dataset: EuroSAT RGB
- Training approach: Two-stage training, consisting of classification-head training followed by full-network fine-tuning.

### Training-data proportions

Five proportions of the training partition were evaluated:

| Training proportion | Number of training images |
|---|---:|
| 10% | 1,944 |
| 30% | 5,832 |
| 50% | 9,720 |
| 70% | 13,608 |
| 90% | 17,496 |

Each experiment was repeated using three random seeds:

- 42
- 43
- 44

Therefore, a total of **15 experimental runs** were conducted.

The fixed validation and test sets were used for all experiments.

## 5. Evaluation Metrics

The model performance was evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Confusion matrices
- Class-wise classification reports

Mean and standard deviation across the three random seeds were calculated to examine the consistency of the results.

## 6. Experimental Results

The following table presents the mean test accuracy across the three random seeds.

| Training proportion | Mean test accuracy |
|---|---:|
| 10% | 90.07% |
| 30% | 94.10% |
| 50% | 95.02% |
| 70% | 95.91% |
| 90% | 96.31% |

The results indicate that classification accuracy improves as more labelled training data is provided. The improvements become smaller at higher training-data proportions.

### Learning Curve and Comparison

The following figure compares our ImageNet-pretrained ResNet-50 with the ResNet-50 trained from scratch reported in the original paper.

![EuroSAT Learning Curve](Improvement/part_a_learning_curve.png)

The comparison illustrates the performance of the pretrained model across different training-data sizes relative to the original paper's reported learning curve.

The comparison should be interpreted in the context of the respective experimental setups.

## 7. Key Findings

- The ImageNet-pretrained ResNet-50 achieved 90.07% mean test accuracy using 10% of the training partition.
- Increasing the training-data proportion improved classification performance.
- The highest mean test accuracy was 96.31%, achieved using 90% of the training partition.
- The improvement in accuracy became smaller as the training-data proportion increased.
- Repeating the experiments with three random seeds provided an assessment of result consistency.

These findings demonstrate the importance of evaluating model performance at different levels of labelled data availability.

## 8. Repository Structure

The Part A folder contains the following files and directories:

| File or folder | Description |
|---|---|
| `EuroSAT_Part_A_Label_Efficiency.ipynb` | Main experimental notebook containing implementation and analysis |
| `part_a_combined_training_history.csv` | Epoch-wise training history across experimental runs |
| `part_a_results_summary.csv` | Aggregated experimental results |
| `part_a_validation_summary.csv` | Validation performance summary |
| `part_a_test_results.csv` | Final test evaluation results |
| `Classification_Reports/` | Classification reports for the experimental runs |
| `Confusion_Matrices/` | Confusion matrix CSV files |
| `Improvement/` | Description and visualisation of the individual improvement |

The shared dataset split and class mapping are maintained in the `Baseline` folder to avoid unnecessary duplication.

## 9. Reproducibility

The main Jupyter notebook documents the implementation, training procedure, evaluation and experimental analysis.

To reproduce the experiments:

1. Clone the project repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Make the EuroSAT RGB dataset available.
4. Ensure the shared dataset split and class mapping files are available.
5. Run the Part A notebook using the documented experiment configuration.

The experiments use three random seeds (42, 43 and 44), with the same validation and test partitions across all training-data proportions.

## 10. Conclusion

Part A evaluates the effect of training data availability on EuroSAT image classification using an ImageNet-pretrained ResNet-50.

Across five training-data proportions and three random seeds, the experiments demonstrate that strong classification performance can be achieved even with a reduced amount of labelled training data.

The findings provide insight into the relationship between pretraining, labelled data availability and classification performance in satellite image classification.
