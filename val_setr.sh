CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-512x512.py
CHECKPOINT_FILE=ckpt/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

#CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256.py
#CHECKPOINT_FILE=ckpt/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
#python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

#CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256_pi.py
#CHECKPOINT_FILE=ckpt/ms_ckpt/setr_naive_256x256_160k_b16_ade20k_20210619_191258-061f24f5_pi.pth
#python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

#CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256_bi.py
#CHECKPOINT_FILE=ckpt/ms_ckpt/setr_naive_256x256_160k_b16_ade20k_20210619_191258-061f24f5_bi.pth
#python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}


CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256_bipi.py
CHECKPOINT_FILE=ckpt/ms_ckpt/setr_naive_256x256_160k_b16_ade20k_20210619_191258-061f24f5_bi.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir work_dirs/configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256_bi


CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256_bipi.py
CHECKPOINT_FILE=ckpt/ms_ckpt/setr_naive_256x256_160k_b16_ade20k_20210619_191258-061f24f5_pi.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir work_dirs/configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256_pi


CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256.py
CHECKPOINT_FILE=ckpt/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir work_dirs/configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-256x256






CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-128x128_bipi.py
CHECKPOINT_FILE=ckpt/ms_ckpt/setr_naive_128x128_160k_b16_ade20k_20210619_191258-061f24f5_bi.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir work_dirs/configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-128x128_bi


CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-128x128_bipi.py
CHECKPOINT_FILE=ckpt/ms_ckpt/setr_naive_128x128_160k_b16_ade20k_20210619_191258-061f24f5_pi.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir work_dirs/configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-128x128_pi


CONFIG_FILE=configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-128x128.py
CHECKPOINT_FILE=ckpt/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --work-dir work_dirs/configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-128x128



