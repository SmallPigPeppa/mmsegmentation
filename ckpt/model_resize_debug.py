# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader
from utils import to_2tuple, interpolate_resize_patch_embed, pi_resize_patch_embed, resize_abs_pos_embed


def resize_patch_embed(state_dict, new_patch_size, new_grid_size, old_grid_size, resize_type='pi'):
    # Adjust patch embedding
    if resize_type == "pi":
        state_dict["backbone.patch_embed.projection.weight"] = pi_resize_patch_embed(
            state_dict["backbone.patch_embed.projection.weight"],
            to_2tuple(new_patch_size),
        )
    elif resize_type == "interpolate":
        state_dict["backbone.patch_embed.projection.weight"] = interpolate_resize_patch_embed(
            state_dict["backbone.patch_embed.projection.weight"],
            to_2tuple(new_patch_size),
        )
    else:
        raise ValueError(
            f"{resize_type} is not a valid value for resize_type. Should be one of ['flexi', 'interpolate']"
        )

    # Adjust position embedding
    if "backbone.pos_embed" in state_dict.keys():
        state_dict["backbone.pos_embed"] = resize_abs_pos_embed(
            state_dict["backbone.pos_embed"], new_size=to_2tuple(new_grid_size), old_size=old_grid_size,
            num_prefix_tokens=1
        )

    return state_dict


def main():
# src_path=setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth
# dst_path=ms_ckpt/setr_naive_vit-large_8x1_384x384_80k_cityscapes_20211123_000505-20728e80_pi.pth
# python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 8 --new_grid_size 32 --old_grid_size 32 --resize_type pi
    new_patch_size = 8
    new_grid_size = 32
    old_grid_size = 32
    resize_type = 'pi'
    src = 'setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth'
    dst = 'ms_ckpt/setr_naive_vit-large_8x1_384x384_80k_cityscapes_20211123_000505-20728e80_pi.pth'


    checkpoint = CheckpointLoader.load_checkpoint(src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = resize_patch_embed(
        state_dict,
        new_patch_size=new_patch_size,
        new_grid_size=new_grid_size,
        old_grid_size=old_grid_size,
        resize_type=resize_type
    )
    mmengine.mkdir_or_exist(osp.dirname(dst))
    torch.save(weight, dst)


if __name__ == '__main__':
    main()
