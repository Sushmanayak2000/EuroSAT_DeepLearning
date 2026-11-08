# Title: Reproducing EuroSAT Land-Use and Land-Cover Classification and Evaluating Label Efficiency

**Deep Learning · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Sushma, Ishika Thakur · **Repository:** https://github.com/Sushmanayak2000/EuroSAT_DeepLearning · **Date:** 20/10/2026

## Abstract

Land-use and land-cover (LULC) classification from satellite imagery is an important task in remote sensing, enabling the identification of geographical and environmental features from satellite images. This project investigates deep-learning-based LULC classification using the EuroSAT RGB dataset, which contains 27,000 labelled images across ten classes. Based on Helber et al. (2019), we implemented an ImageNet-pretrained ResNet-50 baseline using a stratified 72/8/20 training-validation-test split and evaluated it across three random seeds. The reproduced baseline achieved a mean test accuracy of 96.64% ± 0.27 percentage points, compared with the 98.57% accuracy reported in the original paper.

The project also investigated two individual improvements. Part A examined the effect of training-data availability by fine-tuning ImageNet-pretrained ResNet-50 using five proportions of the training partition (10%, 30%, 50%, 70% and 90%). Mean test accuracy increased from 90.07% ± 0.43 percentage points with 10% of the training data to 96.31% ± 0.17 percentage points with 90%. These results were compared with the scratch-trained ResNet-50 learning curve reported in the original paper. Part B evaluated ImageNet-pretrained ResNet-18 as a smaller alternative to ResNet-50. It achieved a mean test accuracy of 95.74% ± 0.02 percentage points, with 11.18 million parameters.

Overall, the experiments demonstrate how training-data availability affects classification performance and evaluate the accuracy trade-off associated with using a smaller network. All experimental results were evaluated on the fixed EuroSAT test partition using three random seeds.

## 1 · The paper and the task (research understanding, part of 15 %)

- **Paper:** Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242

- **Task:** The task is multiclass land-use and land-cover (LULC) classification using satellite imagery. The input is a 64 × 64 RGB satellite image, and the objective is to classify it into one of ten land-use and land-cover categories. For Part A, the five training-data proportions were selected from the original training partition of 19,440 images. The corresponding training sizes were 1,944 (10%), 5,832 (30%), 9,720 (50%), 13,608 (70%) and 17,496 (90%). The validation and test partitions remained unchanged across all experiments.

- **Their method in our own words:** Helber et al. introduce the EuroSAT dataset and evaluate different deep learning architectures for satellite image classification. Their experiments include convolutional neural networks and ResNet architectures. They also investigate transfer learning by using models pretrained on ImageNet and fine-tuning them for EuroSAT classification.

- **Their main results:** The authors report an accuracy of 98.57% for an ImageNet-pretrained ResNet-50 on the EuroSAT RGB dataset (Table III). They also present a learning curve for a ResNet-50 trained from scratch using different proportions of training data (Table II). The reported accuracies for the 10%, 30%, 50%, 70% and 90% training-data proportions are 75.06%, 93.75%, 94.45%, 95.32% and 96.37%, respectively.

- **Relevance to our project:** This paper provides the reference approach and benchmark for our experiments. We first implemented an ImageNet-pretrained ResNet-50 as the shared baseline and compared its performance with the paper's pretrained ResNet-50 result in Table III. In Part A, we investigated the effect of training-data availability by evaluating the pretrained ResNet-50 using five training-data proportions. We compared these results with the scratch-trained ResNet-50 learning curve reported in Table II. Part B further investigated the use of ResNet-18 as a smaller alternative to ResNet-50.

## 2 · The data

The experiments use the RGB version of the EuroSAT dataset introduced by Helber et al. (2019). It contains 27,000 labelled satellite image patches covering ten land-use and land-cover classes. Each image has a spatial resolution of 64 × 64 pixels and three RGB channels.

| | |
|---|---|
| Source and version | Official EuroSAT RGB dataset from Zenodo: https://zenodo.org/records/7711810. Downloaded for this project |
| Licence | MIT Licence |
| Size | 27,000 images, each with dimensions 64 × 64 pixels and 3 RGB channels |
| Classes or targets | 10 land-use and land-cover classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake.|
| Splits | Stratified train-validation-test split: 19,440 training images (72%), 2,160 validation images (8%), and 5,400 test images (20%).|
| Preprocessing | Images are loaded as RGB and converted to PyTorch tensors. ImageNet mean and standard deviation are used for channel-wise normalisation. Original image dimensions are retained. |
| Leakage checks | Checked for unreadable images, unexpected image dimensions or colour modes, and exact duplicate images. Verified that train, validation, and test splits have no overlapping image paths. The split was saved for reproducibility |
| Shared with Biomedical Image Analysis? | No |

The dataset was divided using a stratified random split rather than a geographic or time-based split. The same saved split was used for the baseline and the subsequent experiments to maintain consistency. The validation set was used during model development and checkpoint selection, while the test set was kept separate for final evaluation.

![Representative EuroSAT image samples](figures/Representative%20EuroSAT%20image%20samples.png)

*Figure 1. Representative RGB image samples from the ten EuroSAT land-use and land-cover classes.*


![EuroSAT class distribution](figures/Class%20distribution.png)

*Figure 2. Distribution of images across the ten EuroSAT land-use and land-cover classes.*
Visual inspection of the representative samples shows that some land-cover categories can be visually similar, which may make classification more challenging. These observations provide context for the error analysis presented in Section 6.

## 3 · Reproducing the baseline (part of 15 %, and 25 % implementation)

**The recipe:** The aim of this stage was to reproduce the ResNet-50 classification approach described by Helber et al. (2019) using the EuroSAT RGB dataset. The original paper was used as the reference for the model architecture, training procedure and evaluation. Where implementation details were not specified, we selected and documented our own settings.
| | The paper says | Where (page, table, code) | What we use |
|---|---|---|---|
| Data: version, split |EuroSAT RGB dataset, 80/20 training-test split applied class-wise| Section IV-A, Page 5|EuroSAT RGB, stratified 72/8/20 train-validation-test split |
| Model: layers, sizes, pretrained? |ResNet-50 pretrained on ILSVRC-2012 (ImageNet) | Table III, Page 5, Section IV-A, Page 6| ImageNet-pretrained ResNet-50, final fully connected layer replaced with 10 output classes|
| Training: optimiser, learning rate, schedule | First train the last layer with LR 0.01, then fine-tune the entire network with LR between 0.001 and 0.0001. Optimiser and schedule not stated| Section IV-A, Page 6|  AdamW with weight decay of 0.0001. Two-stage training: classification-head training with a learning rate of 0.01, followed by full-network fine-tuning with a learning rate of 0.0001.|
| Training: epochs, batch size |Not stated|Section IV-A, Page 6| 3 epochs for head training, 7 epochs for fine-tuning; batch size 64|
| Augmentation and preprocessing | RGB images formed from bands B04, B03, B02. Further preprocessing and augmentation not stated| Section IV-A, Page 5| Tensor conversion and ImageNet mean/std normalization; no additional augmentation|
| Metric, and how it was averaged |Overall classification accuracy; averaging across seeds not stated |Section IV-A, Page 5; Table III, Page 5 | Test accuracy, macro-precision, macro-recall, and macro-F1; mean ± standard deviation across 3 seeds |

- **Setup:** The baseline was implemented using PyTorch in Google Colab with a CUDA T4 GPU. An ImageNet-pretrained ResNet-50 was used, with its final classification layer modified to predict the ten EuroSAT classes. The dataset was divided into fixed, stratified training, validation and test subsets using a 72/8/20 split. The validation set was used for model selection and checkpoint selection, while the test set was kept separate for final evaluation. The experiment was repeated using three random seeds: 42, 43 and 44. The same data split and training configuration were maintained across the runs to support a consistent comparison.
- **What the paper did not say,** and what you chose instead: The paper does not specify the optimizer, number of epochs, batch size, data augmentation, or validation set. We chose AdamW as the optimizer, trained the classification head for 3 epochs and the full network for 7 epochs, and used a batch size of 64. No additional data augmentation was applied. A validation subset was taken from the training data, and the test set was kept separate. These choices were kept fixed across the three runs.

| | Paper accuracy (%)| Ours: mean test accuracy ± SD (percentage points, n = 3) |
|---|---|---|
| Overall classification accuracy | 98.57% | 96.64% ± 0.27|

The paper reports an accuracy of 98.57% for the ImageNet-pretrained ResNet-50 model on EuroSAT RGB using an 80/20 class-wise training-test split (Table III, Page 5). Our baseline achieved a mean test accuracy of **96.64% ± 0.27 percentage points** across three runs. This represents a difference of **1.93 percentage points** compared with the paper's reported accuracy. The results differ, in part, because our experiment uses a stratified 72/8/20 train-validation-test split rather than the paper's 80/20 training-test split. The paper also leaves several training details unspecified, which required us to make and document our own implementation choices.

### Baseline checks following feedback

The following checks investigated possible causes of the initial gap. Each used seed 42 only; these results cannot establish consistency across seeds.

| Check | Change | Seed | Validation accuracy | Test accuracy | Time per run |
|---|---|---:|---|---|---|
| Longer training | 64 × 64; up to 30 fine-tuning epochs, early stopping with patience 5 | 42 | 97.87% | Unavailable | 10.95 min |
| Resizing | 224 × 224 with the longer-training procedure | 42 | 98.29% | Unavailable | 62.41 min |
| Full 80% training | Training and validation combined; 64 × 64; 3 head + 15 fine-tuning epochs | 42 | N/A: validation included in training | 97.02% | Not yet retrieved |

The test accuracies for the longer-training and resizing checks were unavailable because their checkpoints were not retained. Therefore, their validation results are reported separately and are not treated as test performance.

Resizing the images to 224 × 224 increased validation accuracy from 97.87% to 98.29%, a difference of 0.42 percentage points. However, the runtime increased from approximately 11 minutes to 62 minutes per run. Based on this trade-off, we retained the original 64 × 64 input resolution.

The full-80% training check achieved 97.02% test accuracy. This was an exploratory check rather than a replacement for the original baseline. The original baseline configuration was retained for the subsequent individual improvements.

### Deviations and remaining gap

| Deviation from the paper | Reason and limitation |
|---|---|
| 72/8/20 split instead of 80/20 | Reserves an independent validation set for checkpoint selection; fewer images are used for training. |
| AdamW | Provides adaptive updates and decoupled weight decay. The paper does not state its optimiser; no comparison with SGD was performed. |
| Three head epochs, seven fine-tuning epochs; batch size 64 | Implementation choices where the paper does not specify settings; retained as the fixed comparison budget. |
| Native 64 × 64 network inputs | Resizing is not clearly specified in the paper. Our resizing check increased computational cost substantially. |
| ImageNet normalisation without additional augmentation | Explicit implementation choices where the paper does not clearly specify the procedure. |

The full-80% run achieved 97.02% test accuracy, leaving a gap of **1.55 percentage points** to the paper’s 98.57%. Our checks suggest that training duration and input resolution can affect performance. They do not isolate all causes of the remaining gap, and validation scores cannot be compared directly with the paper’s test score.
## 4 · Your improvement (20 %)

### Part A: Effect of Pretraining with Limited Training Data

Since the paper's learning curve is based on training ResNet-50 from scratch, we investigate how ImageNet-pretrained ResNet-50 performs with limited EuroSAT training data. We expect pretraining to provide an advantage when only a small amount of labelled training data is available, with the performance difference becoming less pronounced at higher training-data proportions.

- **Idea:** Fine-tune ImageNet-pretrained ResNet-50 using five training-data proportions (10%, 30%, 50%, 70%, and 90%), keeping the validation and test sets fixed, and compare the results with the paper's scratch-trained learning curve.
- **Why it should help:** ImageNet pretraining provides useful visual features that may help the model recognise EuroSAT land-use and land-cover classes more effectively when only a limited amount of labelled training data is available.
- **Expected effect:** We expect the pretrained ResNet-50 to achieve strong classification accuracy with smaller training-data proportions, with accuracy improving as more training data becomes available. We also expect its performance to be closer to the paper's scratch-trained learning curve at higher training-data proportions. We evaluate the model using three random seeds.
- **Achieved result:** The ImageNet-pretrained ResNet-50 achieved a mean test accuracy of 90.07% with 10% of the training data, increasing to 96.31% with 90% of the training data, averaged across three random seeds.
  
### Part B — Evaluating ResNet-18 as a Smaller Alternative to ResNet-50 

Part B investigated whether a smaller network could retain comparable
classification accuracy on EuroSAT while reducing model size and training
time. ResNet-18 was selected because its smaller parameter count reduces
the size of the model, while ImageNet pretraining provides an established
starting point for transfer learning.

**Hypothesis:** Under the same data split and training settings,
ImageNet-pretrained ResNet-18 would achieve mean test accuracy no more
than one percentage point below the shared ResNet-50 baseline, while
using fewer parameters and taking less time to train.

**Implementation and evaluation:** adapted the shared
training pipeline to use ResNet-18 with a ten-class output layer. The
model was trained on the full training partition of 19,440 images using
seeds 42, 43 and 44. Both models used the same data split, preprocessing
and two-stage training schedule. For each seed, the checkpoint with the
highest validation accuracy was evaluated on the held-out test set.
The Part B contribution also included exporting per-seed metrics and
confusion matrices, comparing parameter counts, recording training times,
and producing the paired statistical comparison and figure.

**Achieved result:** ResNet-18 achieved 95.74% mean test accuracy compared
with 96.64% for ResNet-50. The observed decrease of 0.90 percentage
points fell within the one-point target. The parameter count decreased
from 23.53 million to 11.18 million, a reduction of 52.5%. ResNet-18
training took 2.74–2.87 minutes per run. Confirming a training speedup
requires baseline timings measured under comparable conditions.

[Part B notebook and results](Part%20B/)

## 5 · Experiments (part of 20 % and 15 %)

### Part A — Training-data proportion experiments

- **Design:** The effect of training-data size was evaluated using five proportions of the original training partition: 10%, 30%, 50%, 70% and 90%. The corresponding training sizes were 1,944, 5,832, 9,720, 13,608 and 17,496 images.

The same validation set (2,160 images) and test set (5,400 images) were maintained across all experiments. Each training-data proportion was evaluated using three random seeds (42, 43 and 44), resulting in 15 experiments. The same ImageNet-pretrained ResNet-50 and two-stage training procedure were applied across the five proportions.

| Training-data proportion | Mean test accuracy (%) | Standard deviation (percentage points) | Seeds |
|---|---:|---:|---:|
| 10% (1,944 images) | 90.07 | 0.43 | 3 |
| 30% (5,832 images) | 94.10 | 0.30 | 3 |
| 50% (9,720 images) | 95.02 | 0.32 | 3 |
| 70% (13,608 images) | 95.91 | 0.27 | 3 |
| 90% (17,496 images) | 96.31 | 0.17 | 3 |

- **Statistical comparison:** Mean test accuracy and standard deviation were calculated across three random seeds (42, 43 and 44) for each training-data proportion. The results were compared descriptively to examine how training-data availability affected classification performance. No inferential statistical test was performed for Part A.

- **Test set:** The same fixed test set of 5,400 images was used to evaluate each of the five training-data proportions and construct the learning curve. It was not used for training or checkpoint selection; the validation set was used for model selection.

  
### Part B — ResNet-18 Experiments 

Part B compared ImageNet-pretrained ResNet-18 with the shared ResNet-50
baseline. Both models used the same 19,440 training images, 2,160
validation images and 5,400 test images, with the same preprocessing
and two-stage training schedule. The comparison changed the network
architecture and its corresponding pretrained weights.

ResNet-18 was trained with seeds 42, 43 and 44. For each seed, the
checkpoint with the highest validation accuracy was evaluated on
the held-out test set. Each ResNet-18 result was paired with the
ResNet-50 result from the same training seed.

| Seed | ResNet-50 test accuracy (%) | ResNet-18 test accuracy (%) | Difference (pp) |
|---|---:|---:|---:|
| 42 | 96.4630 | 95.7222 | −0.7408 |
| 43 | 96.9444 | 95.7593 | −1.1851 |
| 44 | 96.5000 | 95.7407 | −0.7593 |
| **Mean** | **96.6358** | **95.7407** | **−0.8951** |

Differences are calculated as ResNet-18 minus ResNet-50. The sample
standard deviation across seeds was 0.27 percentage points (pp) for
ResNet-50 and 0.02 pp for ResNet-18.

**Statistical comparison.** A two-sided paired t-test gave
*t*(2) = −6.17 and *p* = 0.0253. The 95% confidence interval for
the mean difference was [−1.52, −0.27] pp. The exact Wilcoxon
signed-rank test gave *p* = 0.25. The observed mean loss of 0.90 pp
fell within the predefined one-point margin. With three matched
runs, the confidence interval still allows a loss larger than one point.

**Model size and training time.** ResNet-18 used 11.18 million
parameters compared with 23.53 million for ResNet-50, a reduction
of 52.5%. ResNet-18 training took 2.74–2.87 minutes per run.
Estimating a training speedup requires comparable ResNet-50 timings.

[Part B notebook, per-seed results and comparison figure](Part%20B/)

## 6 · Evaluation and error analysis (15 %)

### 6.1 Shared baseline – Evaluation and error analysis

**Metrics:** The shared baseline was evaluated using test accuracy, macro-precision, macro-recall and macro-F1. The ImageNet-pretrained ResNet-50 achieved a mean test accuracy of 96.64% ± 0.27 percentage points across three random seeds.

**Confusion matrix and errors:** The confusion matrix was examined to identify misclassifications between land-use and land-cover categories. Particular attention was given to confusion between River and Highway, which is also reported in the original EuroSAT paper (Figure 8). Three misclassified test images were selected to illustrate these errors.

### 6.2 Part A – Effect of Pretraining with Limited Training Data

**Metrics:** In addition to test accuracy, macro-precision, macro-recall and macro-F1 were calculated for each training-data proportion. The results were evaluated across three random seeds.

**Results:** The mean test accuracy increased from 90.07% ± 0.43 percentage points with 10% of the training data to 96.31% ± 0.17 percentage points with 90% of the training data. This demonstrates that the ImageNet-pretrained ResNet-50 achieved strong classification performance even with a limited number of labelled training images.

**Learning-curve comparison:** The results were compared with the scratch-trained ResNet-50 learning curve reported in Table II of the original paper. The pretrained model achieved substantially higher accuracy at the 10% training-data proportion, while the results were much closer at higher training-data proportions. However, this comparison is indicative rather than a controlled experiment measuring the effect of pretraining alone, because the experimental settings differ.

**Error analysis:** Confusion matrices and classification reports were generated for all five training-data proportions to examine class-wise performance and identify misclassifications.

**Limitations:** The experiments show how the ImageNet-pretrained ResNet-50 performs with different amounts of EuroSAT training data. However, the comparison with the paper's scratch-trained model does not isolate the effect of pretraining because the experimental setups differ. The results are also based on three random seeds and one fixed test split.

**6.3 Part B – ResNet-18 Evaluation and Error Analysis (Ishika Thakur)**

**Metrics and results.** ResNet-18 was evaluated on the same 5,400
held-out test images as the shared ResNet-50 baseline. Across seeds
42, 43 and 44, ResNet-18 achieved a mean test accuracy of
**95.74% ± 0.02 percentage points** (mean ± sample standard deviation),
compared with **96.64% ± 0.27 percentage points** for ResNet-50.
ResNet-18 achieved **95.65% macro precision**, **95.52% macro recall**
and **95.57% macro F1**, averaged across the three runs. The macro
metrics give equal weight to each of the ten classes.

**Error analysis.** A classification report and confusion matrix were
saved for each ResNet-18 run. The off-diagonal entries show which
true classes were predicted as other classes. These matrices allow
a direct check of whether the smaller model shares the baseline's
River–Highway confusion and whether the errors occur in both directions.

**Limitations.** The three runs used one fixed data split, so the
reported spread measures variation across training seeds. Performance
on geographically separate imagery remains untested. The observed mean
accuracy loss of 0.90 percentage points fell within the one-point
target, but the confidence interval in Section 5 includes losses
larger than one point.


## 7 · Conclusion

This project investigated deep-learning-based land-use and land-cover classification using the EuroSAT RGB dataset and reproduced a ResNet-50 classification baseline based on the approach presented by Helber et al. (2019). The reproduced baseline achieved a mean test accuracy of 96.64% ± 0.27 percentage points across three random seeds, compared with the 98.57% accuracy reported in the original paper.

The first improvement (Part A) investigated the effect of training-data availability using an ImageNet-pretrained ResNet-50. The mean test accuracy increased from 90.07% ± 0.43 percentage points  with 10% of the training data to 96.31% ± 0.17 percentage points with 90% of the training data. These results demonstrate that the pretrained model achieved strong classification performance even with a limited amount of labelled satellite imagery. The comparison with the paper's scratch-trained learning curve showed a substantial performance difference at 10% training data, while the results were much closer at higher training-data proportions. However, this comparison does not isolate the effect of pretraining alone because the experimental settings differ.

The second improvement (Part B) investigated ResNet-18 as a smaller alternative to ResNet-50. ResNet-18 achieved a mean test accuracy of 95.74% ± 0.02 percentage points, compared with 96.64% for ResNet-50. The observed accuracy difference was approximately 0.90 percentage points. ResNet-18 also reduced the number of model parameters from 23.53 million to 11.18 million, representing a reduction of 52.5%. Although the observed accuracy difference was within the predefined one-percentage-point target, the limited number of experimental runs does not establish statistical non-inferiority.

The evaluation also highlighted classification challenges, particularly confusion between River and Highway classes. This behaviour was also reported in the original EuroSAT paper, suggesting that certain land-cover categories have similar visual characteristics and can be difficult to distinguish.

Overall, the project provided practical experience in reproducing a published deep learning approach, designing experiments, evaluating classification performance and analysing model errors. The results highlighted the importance of training-data availability, transfer learning and model architecture when developing satellite image classification systems.

As future work, further experiments could investigate additional data augmentation techniques, alternative training strategies and improved image preprocessing. More random seeds and additional data splits could also provide stronger evidence about the consistency of the results. Further investigation into class-specific errors could help improve the distinction between visually similar land-cover categories while maintaining computational efficiency.

## 8 · Use of AI tools (5 %)
| Date | What I used ChatGPT for ( Part B) |
| --- | --- |
| 28 October 2026 | Helped formulate the research question and hypothesis for Part B, which compared an ImageNet-pretrained ResNet-18 with the shared ResNet-50 baseline. |
| 29 October 2026 | Helped clarify how to describe the baseline protocol, including the fixed training, validation, and test split, the use of random seeds, and checkpoint selection based on validation performance. |
| 7 November 2026 | Helped refine the Part B discussion using results from our experiments. We discussed how to report the trade-off between test accuracy and parameter count, and what timing evidence would be needed to claim a training speedup. |
ChatGPT (OpenAI) was used occasionally as a supplementary tool during project development, mainly to help troubleshoot coding issues. For example, it was used to understand Python and PyTorch error messages, investigate possible causes of implementation errors, and clarify concepts related to model training and fine-tuning.

When a coding issue occurred, ChatGPT suggestions were considered as possible solutions. Any relevant changes were checked manually, tested in the Jupyter notebooks, and verified by examining whether the code executed correctly and produced the expected outputs. The experimental setup, implementation, model training, evaluation, and interpretation of results were carried out by the project team.

ChatGPT was not used to generate model predictions, experimental results, or evaluation metrics. All reported results were obtained from the actual model experiments.

## 9 · Reproducibility (part of 25 %)

### Run it

The project was implemented in Python using PyTorch and Jupyter notebooks in Google Colab. The complete source code, experiment configurations, saved results and supporting files are available in the project GitHub repository:

https://github.com/Sushmanayak2000/EuroSAT_DeepLearning

To reproduce the experiments, the following steps should be followed:

1. Clone the GitHub repository.
2. Install the required dependencies using the provided `requirements.txt` file.
3. Download the EuroSAT RGB dataset from the official Zenodo source (https://zenodo.org/records/7711810) and make it available in the Google Colab environment.
4. Run the shared baseline notebook, `Baseline/EuroSAT_ResNet50_Baseline.ipynb`. This notebook prepares the dataset, creates the fixed stratified split, trains the ResNet-50 baseline and performs the evaluation.
5. Run `Part A/EuroSAT_Part_A_Label_Efficiency.ipynb` to conduct the training-data proportion experiments using the shared dataset split.
6. Run `Part B/PartB_ResNet18_Ishika_Final.ipynb` to conduct the ResNet-18 experiments.
7. Run the evaluation and analysis cells in the respective notebooks to generate the metrics, classification reports, confusion matrices and visualisations.

The shared dataset split and class mapping are stored in the `Baseline` folder as `eurosat_splits.csv` and `class_to_idx.json`. These files are reused across the experiments to maintain consistency.

### Seeds and hardware

The experiments were conducted using Google Colab with GPU acceleration on an NVIDIA Tesla T4 GPU. The implementation uses Python and PyTorch.

Three random seeds (42, 43 and 44) were used for the baseline and both individual experiments to examine the consistency of the results.

The shared ResNet-50 baseline and Part A use a two-stage training procedure: three epochs of classification-head training followed by seven epochs of full-network fine-tuning, with a batch size of 64.

Part A consists of 15 experimental runs, covering five training-data proportions (10%, 30%, 50%, 70% and 90%), each repeated with the three random seeds.

Part B evaluates ImageNet-pretrained ResNet-18 using the same three seeds. Its recorded training time was approximately 2.80–2.86 minutes per run. Training time can vary depending on the available Colab runtime and hardware conditions.

### Where the results come from

The main notebooks and their corresponding output files are organised into three folders.

| Experiment | Notebook | Saved results and outputs |
|---|---|---|
| Shared ResNet-50 baseline | `Baseline/EuroSAT_ResNet50_Baseline.ipynb` | `test_results.csv`, `training_history.csv`, `eurosat_splits.csv`, `class_to_idx.json` |
| Part A: Training-data proportions | `Part A/EuroSAT_Part_A_Label_Efficiency.ipynb` | `part_a_test_results.csv`, `part_a_results_summary.csv`, `part_a_validation_summary.csv`, `part_a_combined_training_history.csv` |
| Part A: Class-wise evaluation | Same Part A notebook | `Classification_Reports/` and `Confusion_Matrices/` |
| Part A: Learning-curve comparison | Same Part A notebook | `Improvement/part_a_learning_curve.png` |
| Part B: ResNet-18 experiments | `Part B/PartB_ResNet18_Ishika_Final.ipynb` | `resnet18_test_results.csv`, `resnet18_test_summary.csv`, `resnet18_validation_results.csv`, `resnet18_validation_summary.csv`, `resnet18_training_history.csv` |
| Part B: Class-wise evaluation | Same Part B notebook | Per-seed classification reports and confusion matrices |
| Part B: Training and validation curves | Same Part B notebook | `resnet18_training_validation_accuracy.png` and `resnet18_training_validation_loss.png` |
| Part B: Paired comparison | Same Part B notebook | `paired_runs-2.png` and `resnet18_baseline_results_summary.csv` |

The individual experiment outputs are saved as CSV, JSON and PNG files in the respective project folders. These files allow the reported metrics, training histories, class-wise results and comparison figures to be traced back to their experiments.

The repository also includes a `requirements.txt` file to document the software dependencies required to run the project.
## References

[1] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217–2226, 2019, doi: 10.1109/JSTARS.2019.2918242.

[2] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," Zenodo, 2019. [Online]. Available: https://zenodo.org/records/7711810.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[4] PyTorch, "PyTorch Documentation." [Online]. Available: https://pytorch.org/docs/stable/index.html.

[5] PyTorch, "Torchvision Models Documentation – ResNet-50." [Online]. Available: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html.

[6] PyTorch, "Torchvision Models Documentation – ResNet-18." [Online]. Available: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html.

[7] Google, "Google Colab." [Online]. Available: https://colab.research.google.com/.
