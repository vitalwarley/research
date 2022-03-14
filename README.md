# research

## Automatic Kinship Recognition

- MXNet original implementation: [vuvko/fitw2020](https://github.com/vuvko/fitw2020)

### My Aims

- [x] Adapt Shadrikov code to PyTorch.
- [ ] Reproduce Shadrikov verification results.
  - [ ] Pretrain ResNet101 with ArcFace loss on cleaned MS-Celeb-1M.
  - [ ] Finetune pretrained model
    - [ ] pretrain + classification
    - [ ] pretrain + classification + normalization
    - [ ] pretrain + classification + normalization + different thresholds
- [ ] Experiment with transformers instead of the ResNet backbone on both MS-Celeb-1M pretraining and FIW finetuning.
