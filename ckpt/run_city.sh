src_path=setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth
dst_path=ms_ckpt/setr_naive_vit-large_8x1_384x384_80k_cityscapes_20211123_000505-20728e80_pi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 8 --new_grid_size 48 --old_grid_size 48 --resize_type pi



src_path=setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth
dst_path=ms_ckpt/setr_naive_vit-large_8x1_384x384_80k_cityscapes_20211123_000505-20728e80_bi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 8 --new_grid_size 48 --old_grid_size 48 --resize_type interpolate



src_path=setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth
dst_path=ms_ckpt/setr_naive_vit-large_8x1_192x192_80k_cityscapes_20211123_000505-20728e80_pi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 4 --new_grid_size 48 --old_grid_size 48 --resize_type pi



src_path=setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth
dst_path=ms_ckpt/setr_naive_vit-large_8x1_192x192_80k_cityscapes_20211123_000505-20728e80_bi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 4 --new_grid_size 48 --old_grid_size 48 --resize_type interpolate
