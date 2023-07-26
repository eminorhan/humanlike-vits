"""
Util functions
"""
import os
import torch
from huggingface_hub import hf_hub_download

def get_available_models():
    available_models = []

    return available_models

def load_model(model_name):

    # parse identifier
    model_spec, data_frac, seed = model_name.split("_")

    # checks
    assert model_spec in ["vits14", "vitb14", "vitl14", "vith14", "vith14@448", "vith14@476"], "Unrecognized architecture!"
    assert data_frac in ["1.0", "0.1", "0.01", "0.001", "0.0001"], "Unrecognized data fraction!"
    assert seed in ["1", "2", "3"], "Unrecognized model seed!"


    # download checkpoint from hf
    checkpoint = hf_hub_download(repo_id="eminorhan/"+model_name, filename=model_name+".pth")

    model = build_mae(arch, patch_size)
    load_mae(model, checkpoint)

    return model

def build_mae(arch, patch_size):
    import vision_transformer_mae as vits
    full_model_name = arch + "_patch" + str(patch_size)
    model = vits.__dict__[full_model_name](num_classes=0, global_pool=False)

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