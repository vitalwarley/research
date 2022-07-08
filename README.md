# research

## Automatic Kinship Recognition

- MXNet original implementation: [vuvko/fitw2020](https://github.com/vuvko/fitw2020)

### My Aims

- [x] Adapt Shadrikov code to PyTorch.
- [ ] Reproduce Shadrikov verification results.
  - [x] Pretrain ResNet101 with ArcFace loss on cleaned MS-Celeb-1M.
      - [x] Results [here](https://github.com/vitalwarley/research/issues/9#issuecomment-1179432568).
  - [ ] Finetune pretrained model
    - [ ] pretrain + classification
    - [ ] pretrain + classification + normalization
    - [ ] pretrain + classification + normalization + different thresholds
- [ ] Experiment with transformers instead of the ResNet backbone on both MS-Celeb-1M pretraining and FIW finetuning.

### Results

#### Specs

- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- 48GB RAM
- 2x NVIDIA GeForce RTX 3090 24GB

#### CLI

##### Pretrain

TODO

#### Metrics

TODO
