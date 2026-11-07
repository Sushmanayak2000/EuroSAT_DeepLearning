# Title: Reproducing EuroSAT Land-Use and Land-Cover Classification and Evaluating Label Efficiency

**Deep Learning · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Sushma, Ishika Thakur · **Repository:** https://github.com/Sushmanayak2000/EuroSAT_DeepLearning · **Date:** 20/10/2026

> How to use this template: copy it into your repository as `report.md` (or into LaTeX or Word with the same sections).
> Replace everything in [square brackets]. Keep the section titles: they follow the grading criteria (weights in brackets).
> Start now with section 2 (the data). There is no page limit; aim for clarity, not length: every number with its source, every claim with evidence.

---

## Abstract

Land use and land cover (LULC) classification from satellite imagery is an important task in remote sensing, enabling the automated identification of different geographical and environmental features. This project reproduces and evaluates a ResNet-50 transfer learning baseline based on Helber et al. (2019), who introduced the EuroSAT dataset and reported 98.57% classification accuracy using RGB satellite images. The experiments use the EuroSAT RGB dataset, consisting of 27,000 labelled images across ten land cover classes, with a fixed 72% training, 8% validation, and 20% test split and three random seeds to assess reproducibility. For Part A, an ImageNet-pretrained ResNet-50 is trained using a two-stage strategy involving classification head training followed by full-network fine-tuning; for Part B, ResNet-18 is evaluated as a smaller alternative to investigate the trade-off between classification performance and computational efficiency. The reproduced baseline and the ResNet-18 model are compared with the published result of 98.57%, using test accuracy, mean and standard deviation across random seeds, parameter count, and training time to assess their performance and efficiency.

## 1 · The paper and the task (research understanding, part of 15 %)

- **Paper:** Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242
- **Task:** The task is multiclass land-use and land-cover classification. The input is a 64 × 64 RGB satellite image, and the output is one of 10 land-use and land-cover classes.
- **Their method in your own words:** The authors introduce EuroSAT, a labelled satellite image dataset, and evaluate deep learning models for classifying land-use and land-cover categories. Their experiments include convolutional neural network architectures, including ResNet models, and investigate the use of pretrained models for image classification.
- **Their main result:** The authors report a classification accuracy of 98.57% using an ImageNet-pretrained ResNet-50 model on the RGB dataset (Table III).

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

The notebook also includes representative image samples from the ten classes and a class-distribution bar chart. Visual inspection shows that some land-cover categories can be visually similar, which may make classification more challenging. These observations provide context for the error analysis in Section 6.

## 3 · Reproducing the baseline (part of 15 %, and 25 % implementation)

**The recipe:** read the paper (or its code) like a recipe. One row per ingredient; leave nothing empty: if the paper does not say it, write "not stated" and what you chose.

The aim of this stage was to reproduce the ResNet-50 classification approach described by Helber et al. (2019) using the EuroSAT RGB dataset. The original paper was used as the reference for the model architecture, training procedure and evaluation. Where implementation details were not specified, we selected and documented our own settings.
| | The paper says | Where (page, table, code) | What we use |
|---|---|---|---|
| Data: version, split |EuroSAT RGB dataset, 80/20 training-test split applied class-wise| Section IV-A, Page 5|EuroSAT RGB, stratified 72/8/20 train-validation-test split |
| Model: layers, sizes, pretrained? |ResNet-50 pretrained on ILSVRC-2012 (ImageNet) | Table III, Page 5, Section IV-A, Page 6| ImageNet-pretrained ResNet-50, final fully connected layer replaced with 10 output classes|
| Training: optimiser, learning rate, schedule | First train the last layer with LR 0.01, then fine-tune the entire network with LR between 0.001 and 0.0001. Optimiser and schedule not stated| Section IV-A, Page 6| AdamW; Two-stage training: classification-head training with a LR of 0.01, followed by full-network fine-tuning with a LR of 0.0001.|
| Training: epochs, batch size |Not stated|Section IV-A, Page 6| 3 epochs for head training, 7 epochs for fine-tuning; batch size 64|
| Augmentation and preprocessing | RGB images formed from bands B04, B03, B02. Further preprocessing and augmentation not stated| Section IV-A, Page 5| Tensor conversion and ImageNet mean/std normalization; no additional augmentation|
| Metric, and how it was averaged |Overall classification accuracy; averaging across seeds not stated |Section IV-A, Page 5; Table III, Page 5 | Test accuracy, macro-precision, macro-recall, and macro-F1; mean ± standard deviation across 3 seeds |

- **Setup:** The baseline was implemented using PyTorch in Google Colab with a CUDA T4 GPU. An ImageNet-pretrained ResNet-50 was used, with its final classification layer modified to predict the ten EuroSAT classes.The dataset was divided into fixed, stratified training, validation and test subsets using a 72/8/20 split. The validation set was used for model selection and checkpoint selection, while the test set was kept separate for final evaluation.The experiment was repeated using three random seeds: 42, 43 and 44. The same data split and training configuration were maintained across the runs to support a consistent comparison.
- **What the paper did not say,** and what you chose instead: The paper does not specify the optimizer, number of epochs, batch size, data augmentation, or validation set. We chose AdamW as the optimizer, trained the classification head for 3 epochs and the full network for 7 epochs, and used a batch size of 64. No additional data augmentation was applied. A validation subset was taken from the training data, and the test set was kept separate. These choices were kept fixed across the three runs.

| | Paper | Ours (mean ± spread, n seeds) |
|---|---|---|
| Overall classification accuracy | 98.57% | 96.64% ± 0.27|

The paper reports an accuracy of 98.57% for the ImageNet-pretrained ResNet-50 model on EuroSAT RGB using an 80/20 class-wise training-test split (Table III, Page 5). Our baseline achieved a mean test accuracy of 96.32% ± 0.04% across three runs. The results differ, in part, because our experiment uses a stratified 72/8/20 train-validation-test split rather than the paper's 80/20 training-test split. The paper also leaves several training details unspecified, which required us to make and document our own implementation choices.

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

Since the paper's learning curve is based on training ResNet-50 from scratch, we investigate how ImageNet-pretrained ResNet-50 performs with limited EuroSAT training data. We expect pretraining to provide an advantage when only a small amount of labelled training data is available, with the performance difference becoming less pronounced as the training data increases.

- **Idea:** Fine-tune ImageNet-pretrained ResNet-50 using five training-data proportions (10%, 30%, 50%, 70%, and 90%), keeping the validation and test sets fixed, and compare the results with the paper's scratch-trained learning curve.
- **Why it should help:** ImageNet pretraining provides useful visual features that may help the model recognise EuroSAT land-use and land-cover classes more effectively when only a limited amount of labelled training data is available.
- **Expected effect:** We expect the pretrained ResNet-50 to achieve strong classification accuracy with smaller training-data proportions, with accuracy improving as more training data becomes available. We also expect the performance difference compared with the paper's scratch-trained learning curve to become smaller at higher training-data proportions. We evaluate the model using three random seeds.
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

- **Design:** The shared baseline used an ImageNet-pretrained ResNet-50 trained on the full training partition (19,440 images). For Part A, the effect of training-data size was evaluated using five proportions of the original training partition: 10%, 30%, 50%, 70% and 90%. The corresponding training sizes were 1,944, 5,832, 9,720, 13,608 and 17,496 images.

The same validation set (2,160 images) and test set (5,400 images) were maintained across the Part A experiments. Each training-data proportion was evaluated using three random seeds (42, 43 and 44), resulting in 15 experiments. The same two-stage training procedure was applied across the five proportions.
| Setting | [metric] mean | ± spread | seeds |
|---|---|---|---|
|Baseline (full training partition)|96.64|0.27|3|
|Part A (10% training data)|90.07|0.43|3|
|Part A (30% training data)|94.10| 0.30|3|
|Part A (50% training data)|95.02|0.32|3|
|Part A (70% training data)|95.91|0.27|3|
|Part A (90% training data)|96.31|0.17|3|

- **Statistical comparison :** Mean test accuracy and standard deviation were calculated across three random seeds (42, 43 and 44) for the baseline and each Part A training-data proportion. The results were compared descriptively to examine the effect of training-data size on classification performance.
- **Test set:** The same fixed test set of 5,400 images was used to evaluate each of the five Part A training-data proportions and construct the learning curve. It was not used for training or checkpoint selection; the validation set was used for model selection.

  
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

- [Metrics that fit the task (not only accuracy): confusion matrix, precision, recall, F1, IoU, …]
- [Where does the model fail? Examples of its most confident mistakes; which classes are confused, and why]
- [Limitations: what your results do **not** show]
  
**6.1 Shared baseline – Evaluation and error analysis**

Metrics: The shared baseline was evaluated using test accuracy, macro-precision, macro-recall and macro-F1. The ImageNet-pretrained ResNet-50 achieved a mean test accuracy of 96.64% ± 0.27 percentage points across three random seeds.

Confusion matrix and errors: The confusion matrix was examined to identify misclassifications between land-use and land-cover categories. Particular attention was given to confusion between River and Highway, which is also reported in the original EuroSAT paper (Figure 8). Three misclassified test images were selected to illustrate these errors.



**6.2 Part A – Effect of Pretraining with Limited Training Data**

Metrics: In addition to test accuracy, macro-precision, macro-recall and macro-F1 were calculated for each training-data proportion. The results were evaluated across three random seeds.

Results: The mean test accuracy increased from 90.07% ± 0.43 percentage points with 10% of the training data to 96.31% ± 0.17 percentage points with 90% of the training data. This demonstrates that the ImageNet-pretrained ResNet-50 achieved strong classification performance even with a limited number of labelled training images.

Error analysis: Confusion matrices and classification reports were generated for all five training-data proportions to examine class-wise performance and identify misclassifications.
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

[What you reproduced, what you improved, what you would try next.]

This project investigated deep-learning-based land-use and land-cover classification using the EuroSAT RGB dataset and reproduced a ResNet-50 classification baseline based on the approach presented by Helber et al. (2019). The reproduced baseline achieved a mean test accuracy of 96.64% ± 0.27% across three random seeds, compared with the 98.57% accuracy reported in the original paper.

The experiments further demonstrated the importance of training data availability for classification performance. In the experiments using ImageNet-pretrained ResNet-50, the mean test accuracy increased from 90.07% ± 0.43 percentage points with 10% of the training data to 96.31% ± 0.17 percentage points with 90% of the training data. This indicates that pretraining can provide a strong starting point even when only a limited number of labelled satellite images are available.

The evaluation also highlighted some remaining classification challenges, particularly the confusion between River and Highway classes. This behaviour was also reported in the original EuroSAT paper, suggesting that certain land-cover categories have similar visual characteristics and can be difficult to distinguish.

Overall, the project provided practical experience in reproducing a published deep learning approach, designing controlled experiments, evaluating classification performance and analysing model errors. Although the reproduced results were slightly below those reported in the original paper, the experiments helped identify possible reasons for the performance difference and highlighted the importance of consistent data splits, training configurations and evaluation procedures.

As a next step, further experiments could investigate alternative training strategies, improved image preprocessing and data augmentation techniques to improve classification performance while maintaining computational efficiency.

## 8 · Use of AI tools (5 %)

[Which tools you used, for what (code, text, debugging), and how you **checked** their output. Be specific.]

ChatGPT (OpenAI) was used as a supplementary tool during the development of this project, mainly for debugging assistance and discussing possible solutions to technical issues encountered during implementation. It was also used to clarify certain PyTorch concepts, understand error messages and explore possible improvements to the experimental setup.

The suggestions provided by ChatGPT were treated as guidance rather than directly accepted solutions. Relevant code changes were reviewed, tested in the notebook and checked against the actual execution results. Model training, experiment execution, result collection and interpretation were carried out and verified by the project team.

No AI tool was used to generate the experimental results or classification metrics.

## 9 · Reproducibility (part of 25 %)

- **Run it:** [the commands or notebook order; the environment, for example `requirements.txt`]
- **Seeds and hardware:** [seeds used; GPU or CPU; training time]
- **Where the results come from:** [which notebook/script makes which table and figure]

## References

[All papers, data sets, code and websites you used.]


[1] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217–2226, 2019. doi: 10.1109/JSTARS.2019.2918242.

[2] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," Zenodo, 2019. Available: https://zenodo.org/records/7711810.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[4] PyTorch, "PyTorch Documentation." Available: https://pytorch.org/docs/stable/index.html.

[5] PyTorch, "Torchvision Models Documentation – ResNet-50." Available: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html.
