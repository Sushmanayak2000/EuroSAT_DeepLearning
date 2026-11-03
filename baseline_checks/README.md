Both notebooks in this folder investigate the shared
ResNet-50 baseline following the professor's feedback.

## Experiments

| Experiment | Evaluation | Accuracy | Seed |
|---|---|---:|---:|
| Longer training, 64 × 64 | Validation | 97.87% | 42 |
| Resizing to 224 × 224 | Validation | 98.29% | 42 |
| Full 80% training, 64 × 64 | Test | 97.02% | 42 |

The training-and-resizing notebook contains the first two
experiments. The full-training notebook contains the third.

Longer training used early stopping with patience 5 and
a maximum of 30 fine-tuning epochs.

The full-80% experiment used 3 head-training epochs and
15 fine-tuning epochs, selected from the earlier 64 × 64
validation experiment.

Resizing increased total runtime from 10.95 to 62.41 minutes.
We retained 64 × 64 inputs for computational efficiency.

The full-80% test result remains 1.55 percentage points below
the paper's reported 98.57%. Validation and test results
should not be compared directly.

## Error analysis

The full-80% model misclassified:
- 3 River images as Highway
- 8 Highway images as River

## Retained evidence

The 224 × 224 console output was retained, but its checkpoint
and original CSV files were lost when the Colab runtime ended.
