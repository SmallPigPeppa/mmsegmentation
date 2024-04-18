CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-512x512.py
CHECKPOINT_FILE=ckpt/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
