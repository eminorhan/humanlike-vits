# Vision transfomers trained with human-like visual data

This repository contains the models described in the following paper:

Orhan AE (2023) [Scaling may be all you need for achieving human-level object recognition capacity with human-like visual experience](https://arxiv.org/abs/2308.xxxxx) arXiv:2308.xxxxx.

### Loading the models
The models are all hosted on [huggingface](https://huggingface.co/eminorhan/humanlike-vits). You will need the `huggingface_hub` library to download the models from huggingface (I have `huggingface-hub==0.14.1`). Model names are specified in the format `x_y_z`, where `x` is the model architecture, `y` is the fraction of the combined human-like video dataset used for self-supervised pretraining, and `z` is the seed:

* `x` can be one of `vits14`, `vitb14`, `vitl14`, `vith14`, `vith14@448`, `vith14@476`
* `y` can be one of `1.0`, `0.1`, `0.01`, `0.001`, `0.0001`
* `z` can be one of `1`, `2`, `3` 

Here, the model architectures are:
* `vits14` = ViT-S/14 
* `vitb14` = ViT-B/14
* `vitl14` = ViT-L/14
* `vith14` = ViT-H/14
* `vith14@448` = ViT-H/14@448 (trained with 448x448 images)
* `vith14@476` = ViT-H/14@476 (trained with 476x476 images)

and the data fractions are:
* `1.0` = full training data (~5000 hours) 
* `0.1` = 10% of the full data (~500 hours)
* `0.01` = 1% of the full data (~50 hours)
* `0.001` = 0.1% of the full data (~5 hours)
* `0.0001` = 0.01% of the full data (~0.5 hours)

When training on proper subsets of the full data (0.01%-10%), subset selection was repeated 3 times. The seed `z` corresponds to these repeats. Note that for the full training data (100%), there is only 1 possible dataset, so `z` can only be `1` for this case. 

Loading a pretrained model is then as easy as:

```python
from utils import load_model

model = load_model('vith14@476_1.0_1')
```

This will download the corresponding pretrained checkpoint, store it in cache, build the right model architecture, and load the pretrained weights onto the model, all in one go! When you load a model, you may get a warning message that says something like `_IncompatibleKeys(missing_keys=[], unexpected_keys=...)`. This is normal. This happens because we're not loading the decoder model used during MAE pretraining. We're only interested in the encoder backbone.

### Pretraining
The models were all trained with code from [this repository](https://github.com/eminorhan/mae), which is my personal copy of the excellent [MAE repository](https://github.com/facebookresearch/mae) from Meta AI. Pretraining and finetuning logs for all models can be found in the [`logs`](https://github.com/eminorhan/humanlike-vits/tree/master/logs) folder.

### Finetuning