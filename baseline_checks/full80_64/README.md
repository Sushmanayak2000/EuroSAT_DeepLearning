ImageNet-pretrained ResNet-50 with 64 × 64 RGB inputs.
Training: 80% of EuroSAT; testing: the unchanged 20%.
Seed: 42. AdamW, batch size 64.
Three head-training epochs followed by 15 fine-tuning epochs.

Test accuracy: 97.02%.
Gap to the paper's 98.57%: 1.55 percentage points.

River/highway errors:
- River predicted as Highway: 3
- Highway predicted as River: 8
