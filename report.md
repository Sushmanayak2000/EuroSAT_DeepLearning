# Title: Reproducing EuroSAT Land-Use and Land-Cover Classification and Evaluating Label Efficiency

**Deep Learning · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Sushma, Ishika Thakur · **Repository:** https://github.com/Sushmanayak2000/EuroSAT_DeepLearning · **Date:** 20/10/2026

> How to use this template: copy it into your repository as `report.md` (or into LaTeX or Word with the same sections).
> Replace everything in [square brackets]. Keep the section titles: they follow the grading criteria (weights in brackets).
> Start now with section 2 (the data). There is no page limit; aim for clarity, not length: every number with its source, every claim with evidence.

---

## Abstract

[Four or five sentences: the task, the paper, the data, your reproduced result next to the published one, your improvement and its measured effect (mean ± spread).]

## 1 · The paper and the task (research understanding, part of 15 %)

- **Paper:** Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242
- **Task:** The task is multiclass land-use and land-cover classification. The input is a 64 × 64 RGB satellite image, and the output is one of 10 land-use and land-cover classes.
- **Their method in your own words:** The authors introduce EuroSAT, a labelled satellite image dataset, and evaluate deep learning models for classifying land-use and land-cover categories. Their experiments include convolutional neural network architectures, including ResNet models, and investigate the use of pretrained models for image classification.
- **Their main result:** The authors report a classification accuracy of 98.57% using an ImageNet-pretrained ResNet-50 model on the RGB dataset (Table III).

## 2 · The data 

| | |
|---|---|
| Source and version | Official EuroSAT RGB dataset from Zenodo: https://zenodo.org/records/7711810. Downloaded for this project |
| Licence | MIT Licence |
| Size | 27,000 images, each with dimensions 64 × 64 pixels and 3 RGB channels |
| Classes or targets | 10 land-use and land-cover classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. The class distribution is examined using a bar chart in the notebook |
| Splits | Stratified train-validation-test split: 19,440 training images (72%), 2,160 validation images (8%), and 5,400 test images (20%). The split preserves class proportions |
| Preprocessing | Images are loaded as RGB and converted to PyTorch tensors. ImageNet mean and standard deviation are used for channel-wise normalisation. Original image dimensions are retained. |
| Leakage checks | Checked for unreadable images, unexpected image dimensions or colour modes, and exact duplicate images. Verified that train, validation, and test splits have no overlapping image paths. The split was saved for reproducibility |
| Shared with Biomedical Image Analysis? | No |

The notebook also includes example images from each class and a class-distribution bar chart. Image inspection revealed potential visual challenges, including cloud cover and similarities between some land-cover categories. These observations will be considered during error analysis.

## 3 · Reproducing the baseline (part of 15 %, and 25 % implementation)

**The recipe:** read the paper (or its code) like a recipe. One row per ingredient; leave nothing empty: if the paper does not say it, write "not stated" and what you chose.

| | The paper says | Where (page, table, code) | What we use |
|---|---|---|---|
| Data: version, split |EuroSAT RGB dataset, 80/20 training-test split applied class-wise| Section IV-A, Page 5|EuroSAT RGB, stratified 72/8/20 train-validation-test split |
| Model: layers, sizes, pretrained? |ResNet-50 pretrained on ILSVRC-2012 (ImageNet) | Table III, Page 5, Section IV-A, Page 6| ImageNet-pretrained ResNet-50, final fully connected layer replaced with 10 output classes|
| Training: optimiser, learning rate, schedule | First train the last layer with LR 0.01, then fine-tune the entire network with LR between 0.001 and 0.0001. Optimiser and schedule not stated| Section IV-A, Page 6| AdamW; head LR 0.01, full-network fine-tuning LR 0.0001|
| Training: epochs, batch size |Not stated|Section IV-A, Page 6| 3 epochs for head training, 7 epochs for fine-tuning; batch size 64|
| Augmentation and preprocessing | RGB images formed from bands B04, B03, B02. Further preprocessing and augmentation not stated| Section IV-A, Page 5| Tensor conversion and ImageNet mean/std normalization; no additional augmentation|
| Metric, and how it was averaged |Overall classification accuracy; averaging across seeds not stated |Section IV-A, Page 5; Table III, Page 5 | Test accuracy, macro-precision, macro-recall, and macro-F1; mean ± standard deviation across 3 seeds |

- **Setup:** The experiment was implemented in PyTorch using Google Colab with a CUDA T4 GPU. The ResNet-50 model was initialized with ImageNet-pretrained weights. The dataset was divided into training, validation, and test subsets using a stratified 72/8/20 split. The validation set was used to select the best checkpoint, while the test set was reserved for final evaluation. The experiment was repeated with three random seeds: 42, 43, and 44
- **What the paper did not say,** and what you chose instead: The paper does not specify the optimizer, number of epochs, batch size, data augmentation, or validation set. We chose AdamW as the optimizer, trained the classification head for 3 epochs and the full network for 7 epochs, and used a batch size of 64. No additional data augmentation was applied. A validation subset was taken from the training data, and the test set was kept separate. These choices were kept fixed across the three runs.

| | Paper | Ours (mean ± spread, n seeds) |
|---|---|---|
| Overall classification accuracy | 98.57% | 96.32% ± 0.04% |

The paper reports an accuracy of 98.57% for the ImageNet-pretrained ResNet-50 model on EuroSAT RGB using an 80/20 class-wise training-test split (Table III, Page 5). Our baseline achieved a mean test accuracy of 96.32% ± 0.04% across three runs. The results differ, in part, because our experiment uses a stratified 72/8/20 train-validation-test split rather than the paper's 80/20 training-test split. The paper also leaves several training details unspecified, which required us to make and document our own implementation choices.
## 4 · Your improvement (20 %)

Because we observe that our ResNet-50 baseline achieves 96.32% accuracy but uses a relatively large network, we expect replacing it with ImageNet-pretrained ResNet-18 to improve model efficiency by reducing the parameter count by about 50%, while maintaining comparable classification accuracy.

- **Idea:** Replace ImageNet-pretrained ResNet-50 with ImageNet-pretrained ResNet-18, keeping the data split, preprocessing, and training settings fixed.
- **Why it should help:** ResNet-18 is a smaller network that requires fewer parameters and less computation. ImageNet pretraining provides useful visual features that may allow it to classify EuroSAT images accurately despite its smaller size.
- **Expected effect:** We expect fewer parameters and shorter training time while maintaining comparable test accuracy and macro-F1. We will evaluate both models across three random seeds.
## 5 · Experiments (part of 20 % and 15 %)

- **Design:** one change at a time; the same data splits, the same budget; at least three seeds per setting.

| Setting | [metric] mean | ± spread | seeds |
|---|---|---|---|
| baseline | | | |
| + [change 1] | | | |
| + [change 2] | | | |

- **Statistical comparison:** [test and result, for example a paired test over seeds or folds (Session 31)]
- **Test set:** used **once**, for the final models only.

## 6 · Evaluation and error analysis (15 %)

- [Metrics that fit the task (not only accuracy): confusion matrix, precision, recall, F1, IoU, …]
- [Where does the model fail? Examples of its most confident mistakes; which classes are confused, and why]
- [Limitations: what your results do **not** show]

## 7 · Conclusion

[What you reproduced, what you improved, what you would try next.]

## 8 · Use of AI tools (5 %)

[Which tools you used, for what (code, text, debugging), and how you **checked** their output. Be specific.]

## 9 · Reproducibility (part of 25 %)

- **Run it:** [the commands or notebook order; the environment, for example `requirements.txt`]
- **Seeds and hardware:** [seeds used; GPU or CPU; training time]
- **Where the results come from:** [which notebook/script makes which table and figure]

## References

[All papers, data sets, code and websites you used.]
