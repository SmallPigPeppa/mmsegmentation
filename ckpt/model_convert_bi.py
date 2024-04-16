# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader
from resizer import to_2tuple, interpolate_resize_patch_embed, pi_resize_patch_embed, resize_abs_pos_embed


def resize_patch_embed(ckpt, new_patch_size, new_grid_size, old_grid_size, resize_type='pi'):
    # Adjust patch embedding
    if resize_type == "pi":
        ckpt["backbone.patch_embed.projection.weight"] = pi_resize_patch_embed(
            ckpt["backbone.patch_embed.projection.weight"],
            to_2tuple(new_patch_size),
        )
    elif resize_type == "interpolate":
        ckpt["backbone.patch_embed.projection.weight"] = interpolate_resize_patch_embed(
            ckpt["backbone.patch_embed.projection.weight"],
            to_2tuple(new_patch_size),
        )
    else:
        raise ValueError(
            f"{resize_type} is not a valid value for resize_type. Should be one of ['flexi', 'interpolate']"
        )

    # Adjust position embedding
    if "backbone.pos_embed" in ckpt.keys():
        ckpt["backbone.pos_embed"] = resize_abs_pos_embed(
            ckpt["backbone.pos_embed"], new_size=to_2tuple(new_grid_size), old_size=old_grid_size, num_prefix_tokens=1
        )

    return ckpt



def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
                    'MMSegmentation style.')
    parser.add_argument('--src', help='src model path or url',
                        default='setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('--dst', help='save path', default='debug.pth')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = resize_patch_embed(state_dict, new_patch_size=8, new_grid_size=32, old_grid_size=32, resize_type="pi")
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
