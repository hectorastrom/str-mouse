# 2.5 - Second attempt revision: swapping to `byclass` EMNIST dataset

In the second attempt (CNN attempt), I originally trained with the `letters`
split of the EMNIST dataset, which contained 124,000 training letters with
uppercase+lowercase drawings combined into case-agnostic labels.

This revision documents the process of going from the `letters` dataset to
`byclass`, and the results attained.

## Byclass EMNIST Split

For pretraining, we prefer to use the `byclass` split of the EMNIST dataset,
which contains labelled handwritten characters for `a-z`, `A-Z`, and `0-9` (62 classes).

There are ~698,000 unbalanced training samples (1x28x28 images), with
(min=1896, max=38374, mean=11257) samples/class.

## Results

### Pretraining

```
Checkpoint: rose-hill-1
Classes: 62
Val accuracy: 87.5% (epoch 4)
```

### Finetuning

After collecting 15 samples for each of the digits (the only new classes):

```
Checkpoint name:  zesty-sponge-3
Pretrain checkpoint:  rose-hill-1
Total training samples:  1205
Batch size:  128
Epochs:  200
Seed:  42
Data augmentation:  enabled
Class weighting: enabled (handles imbalanced data)
Final test accuracy: 80.9%
```
