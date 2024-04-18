src_path=setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
dst_path=ms_ckpt/setr_naive_256x256_160k_b16_ade20k_20210619_191258-061f24f5_pi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 8 --new_grid_size 32 --old_grid_size 32 --resize_type pi


src_path=setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
dst_path=ms_ckpt/setr_naive_128x128_160k_b16_ade20k_20210619_191258-061f24f5_pi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 4 --new_grid_size 32 --old_grid_size 32 --resize_type pi



src_path=setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
dst_path=ms_ckpt/setr_naive_256x256_160k_b16_ade20k_20210619_191258-061f24f5_bi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 8 --new_grid_size 32 --old_grid_size 32 --resize_type interpolate


src_path=setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
dst_path=ms_ckpt/setr_naive_128x128_160k_b16_ade20k_20210619_191258-061f24f5_bi.pth
python model_resize.py --src ${src_path} --dst ${dst_path} --new_patch_size 4 --new_grid_size 32 --old_grid_size 32 --resize_type interpolate
