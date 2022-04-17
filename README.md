# research

## Automatic Kinship Recognition

- MXNet original implementation: [vuvko/fitw2020](https://github.com/vuvko/fitw2020)

### My Aims

- [x] Adapt Shadrikov code to PyTorch.
- [ ] Reproduce Shadrikov verification results.
  - [x] Pretrain ResNet101 with ArcFace loss on cleaned MS-Celeb-1M.
    - Convergence is possible, despite different [results](https://github.com/vitalwarley/research/issues/9#issuecomment-1100849905) -- I will fix this soon.
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

``` 
python train.py --data-dir ../datasets/MS1M_v3/ --devices 2 --max_epochs 50 --model resnet101 --batch-size 1024 --task pretrain --lr 0.2 --momentum 0.9 --weight-decay 5e-4 --warmup 0 --cooldown 0 --arcface-s 64 --arcface-m 0.5 --precision 16 --loss arcface --scheduler poly --accumulate_grad_batches=2
# train - total of 5127678 samples for n_classes=93430                                                                
# val - total of 51795 samples for n_classes=37221
``` 

#### Metrics

|Backbone|Train Loss|Train Accuracy|Val Loss|Val Accuracy|Date
|-|-|-|-|-|-|
|ResNet34|Train|9.724|93.26%|12.42| 89.62%|10-04-2022
|ResNet101|Train|9.734|96%|10.95| 96.23%|17-04-2022

- ResNet34 trained for 13 epochs, but could continue for more if a I haven't canceled by mistake.
- ResNet101 trained for 15 epochs, but could continue for more if the desktop didn't restarted. 
- Validation set is 1% of full dataset, therefore metrics aren't on pair with insightface.
- Scores are from last step instead of the best step.

