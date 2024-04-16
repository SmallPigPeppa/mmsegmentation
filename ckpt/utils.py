import numpy as np
from einops import rearrange
from functorch import vmap
from torch import Tensor
import collections.abc
from itertools import repeat
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn.functional as F

def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))

def interpolate_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via interpolation

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    patch_embed = F.interpolate(
        patch_embed, new_patch_size, mode=interpolation, antialias=antialias
    )

    return patch_embed



def pi_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample patch embedding weights to a target resolution via pseudo-inverse
    resizing.

    Based on:
        https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py
        https://arxiv.org/abs/2212.08013

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)



def resize_abs_pos_embed(
    pos_embed: torch.Tensor,
    new_size: Tuple[int, int],
    old_size: Optional[Union[int, Tuple[int, int]]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    """Resize absolute position embeddings to a target resolution via interpolation

    Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py

    Args:
        pos_embed: Position embeddings tensor of size [b, n, d]
        new_size: Target [height, width] of embedding
        old_size: Original [height, width] of embedding
        num_prefix_tokens: Number of non-spatial prefix tokens (eg. cls)
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing
    Returns:
        Resized pos_embed of size [b, n', d]
    """

    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]

    if not old_size:
        old_size = int(math.sqrt(pos_embed.shape[1] - num_prefix_tokens))  # type:ignore
    old_size = to_2tuple(old_size)

    # Return if no resize necessary
    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, pos_embed = None, pos_embed

    # Interpolate position embedding
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=new_size, mode=interpolation, antialias=antialias
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    # Add back extra prefix tokens
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)

    return pos_embed