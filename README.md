# Vision transfomers trained with human-like visual data

This repository contains the models described in the following paper:

Orhan AE (2023) [Scaling may be all you need for achieving human-level object recognition capacity with human-like visual experience.](https://arxiv.org/abs/2308.xxxxx) arXiv:2308.xxxxx.

## Loading the models
The models are all hosted on [huggingface](https://huggingface.co/eminorhan/humanlike-vits). You will need the [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/quick-start) library to download the models from huggingface (I have `huggingface-hub==0.14.1`). Model names are specified in the format `x_y_z`, where `x` is the model architecture, `y` is the fraction of the combined human-like video dataset used for self-supervised pretraining, and `z` is the seed:

* `x` can be one of `vits14`, `vitb14`, `vitl14`, `vith14`, `vith14@448`, `vith14@476`
* `y` can be one of `1.0`, `0.1`, `0.01`, `0.001`, `0.0001`
* `z` can be one of `1`, `2`, `3` 

Here, the model architectures (`x`) are:
* `vits14` = ViT-S/14 
* `vitb14` = ViT-B/14
* `vitl14` = ViT-L/14
* `vith14` = ViT-H/14
* `vith14@448` = ViT-H/14@448 (trained with 448x448 images)
* `vith14@476` = ViT-H/14@476 (trained with 476x476 images)

and the data fractions (`y`) are:
* `1.0` = full training data (~5000 hours) 
* `0.1` = 10% of the full data (~500 hours)
* `0.01` = 1% of the full data (~50 hours)
* `0.001` = 0.1% of the full data (~5 hours)
* `0.0001` = 0.01% of the full data (~0.5 hours)

When training on proper subsets of the full data (0.01%-10%), subset selection was repeated 3 times. The seed `z` corresponds to these three repeats. Note that for the full training data (100%), there is only 1 possible dataset, so `z` can only be `1` for this case. 

Loading a pretrained model is then as easy as:
```python
from utils import load_model

model = load_model('vith14@476_1.0_1')
```
where `'vith14@476_1.0_1'` is the model identifier. This will download the corresponding pretrained checkpoint, store it in cache, build the right model architecture, and load the pretrained weights onto the model, all in one go! 

When you load a pretrained model, you may get a warning message that says something like `_IncompatibleKeys(missing_keys=[], unexpected_keys=...)`. This is normal. This happens because we're not loading the decoder model used during MAE pretraining. We're only interested in the encoder backbone.

The above will just load the pretrained model that is not finetuned on ImageNet. If you instead want to load an ImageNet-finetuned version of the model, you just need to set `finetuned=True`:
```python
from utils import load_model

model = load_model('vith14@476_1.0_1', finetuned=True)
```
This will load the corresponding model that is finetuned with 2% of ImageNet training data (this is called the *permissive* finetuning condition in the paper). Unfortunately, I have not saved the models finetuned on ~1% of ImageNet (the *stringent* condition), so these are the only ImageNet-finetuned model I have for now, but please feel free to let me know if you might be interested in other finetuning settings too.

In the finetuned models, we use a classifier head that consists of a `BatchNorm1d` layer + a `Linear` layer.

## Pretraining
The models were all pretrained with code from [this repository](https://github.com/eminorhan/mae), which is my personal copy of the excellent [MAE repository](https://github.com/facebookresearch/mae) from Meta AI. In particular, I have used [this SLURM batch script](https://github.com/eminorhan/mae/blob/master/train_mae_sayavakepicutego4d.sh) to train all models (this script contains all training configuration details). Pretraining logs for all models can be found in the [`logs/pretraining_logs`](https://github.com/eminorhan/humanlike-vits/tree/master/logs/pretraining_logs) folder.

## Finetuning
The models were again finetuned with code from the same repository as the pretraining code. In particular, I have used [this SLURM batch script](https://github.com/eminorhan/mae/blob/master/eval_scripts/eval_finetune_imagenet.sh) to finetune all models (this script contains all relevant finetuning configuration details). All finetuning logs for the *permissive* condition can be found in the [`logs/finetuning_logs`](https://github.com/eminorhan/humanlike-vits/tree/master/logs/finetuning_logs) folder.

One important point to note is that during finetuning, I have not used the standard heavy data augmentations and regularizers used for MAE finetuning (*e.g.* cutmix and mixup). I have instead used very minimal data augmentations (just random resized crops and horizontal flips; see [here](https://github.com/eminorhan/mae/blob/63048899a4ac223d2db3bce0db7f5299193cebfd/eval_finetune.py#L72)). This is to make sure the finetuning data remain as "human-like" as possible. In my experience, it is possible to get a few percentage points better results (in absolute values)  with the more standard heavy agumentation and regularization pipeline for MAEs. There is a [separate branch](https://github.com/eminorhan/mae/tree/regularized) of my MAE repository that implements this more standard finetuning pipeline. You can use this branch if you would like to finetune the pretrained models in a more standard way.  