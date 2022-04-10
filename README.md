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

### Results

#### Specs

- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- 48GB RAM
- 2x NVIDIA GeForce RTX 3090 24GB

#### CLI

``` 
python train.py --data-dir ../datasets/MS1M_v3/ --gpu 0,1 --strategy ddp --max_epochs 25 --model {resnet34,resnet101} --batch-size {2048,904} --task pretrain --lr 0.2 --momentum 0.9 --weight-decay 5e-4 --warmup 0 --cooldown 0 --arcface-s 64 --arcface-m 0.5 --precision 16 --loss arcface --scheduler poly
# train - total of 5127678 samples for n_classes=93430                                                                                                                                                                                                                          
# val - total of 51795 samples for n_classes=37221
``` 

#### Metrics

|Backbone|Set|Loss|Accuracy|
|-|-|-|-|
|ResNet34|Train|9.724|93.26%
|ResNet34|Val|12.42| 89.62%

- ResNet34 trained for 13 epochs, but could continue for more if a I haven't canceled by mistake.

