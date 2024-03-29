"""
utility functions for loading models from hf hub
"""
import os
import torch
from huggingface_hub import hf_hub_download


def load_model(model_name, finetuned=False):

    # parse identifier
    model_spec, data_frac, seed = model_name.split("_")

    # checks
    assert model_spec in ["vits14", "vitb14", "vitl14", "vith14", "vith14@448", "vith14@476"], "Unrecognized architecture!"
    assert data_frac in ["1.0", "0.1", "0.01", "0.001", "0.0001"], "Unrecognized data fraction!"
    assert seed in ["1", "2", "3"], "Unrecognized model seed!"

    # download checkpoint from hf
    if finetuned:
        checkpoint = hf_hub_download(repo_id="eminorhan/humanlike-vits", subfolder="finetuned_" + model_spec, filename=model_name + ".pth")
    else:
        checkpoint = hf_hub_download(repo_id="eminorhan/humanlike-vits", subfolder=model_spec, filename=model_name + ".pth")

    model_name_conversion_dict = {
        "vits14": "vit_small_patch14", 
        "vitb14": "vit_base_patch14",
        "vitl14": "vit_large_patch14",
        "vith14": "vit_huge_patch14",
        "vith14@448": "vit_huge_patch14_448",
        "vith14@476": "vit_huge_patch14_476",
    }

    model = build_mae(model_name_conversion_dict[model_spec], finetuned=finetuned)
    load_mae(model, checkpoint)

    return model


def build_mae(model_name, finetuned=False):
    import vit_models as vits
    if finetuned:
        model = vits.__dict__[model_name](num_classes=1000, global_pool=False)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    else:
        model = vits.__dict__[model_name](num_classes=0, global_pool=False)

    return model


def load_mae(model, pretrained_weights):
    if os.path.isfile(pretrained_weights):    
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint_model = checkpoint['model']

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")


def interpolate_pos_embed(model, checkpoint_model):
    '''
    Interpolate position embeddings for high-resolution. 
    Reference: https://github.com/facebookresearch/deit
    '''
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed