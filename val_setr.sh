CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-512x512.py
CHECKPOINT_FILE=ckpt/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
