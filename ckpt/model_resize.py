# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

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
    parser = argparse.ArgumentParser(description="Resize model checkpoint embeddings.")
    parser.add_argument('--src', type=str, help='Source checkpoint file path')
    parser.add_argument('--dst', type=str, help='Destination checkpoint file path')
    parser.add_argument('--new_patch_size', type=int, help='New patch size for the embedding')
    parser.add_argument('--new_grid_size', type=int, help='New grid size for the position embedding')
    parser.add_argument('--old_grid_size', type=int, help='Old grid size of the position embedding')
    parser.add_argument('--resize_type', type=str, default='interpolate', choices=['pi', 'interpolate'],
                        help='Type of resizing to apply')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    resized_state_dict = resize_patch_embed(
        state_dict,
        new_patch_size=args.new_patch_size,
        new_grid_size=args.new_grid_size,
        old_grid_size=args.old_grid_size,
        resize_type=args.resize_type
    )

    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(resized_state_dict, args.dst)

if __name__ == '__main__':
    main()