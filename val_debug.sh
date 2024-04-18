# IMG resize
#for size in 384x384
#do
#    # Set variables based on the current size
#    CONFIG_FILE="configs/setr/setr_vit-l_naive_8xb1-80k_cityscapes-${size}_imgresize.py"
#    CHECKPOINT_FILE="ckpt/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth"
#    WORK_DIR="work_dirs/setr-city/setr_vit-l_naive_8xb1-80k_cityscapes-${size}_imgresize"
#
#    # Execute the test command
#    echo "Running test for size $size"
#    python tools/test.py "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --work-dir "${WORK_DIR}"
#done


# IMG resize
for size in 256x256
do
    # Set variables based on the current size
    CONFIG_FILE="configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-${size}_imgresize.py"
    CHECKPOINT_FILE="ckpt/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth"
    WORK_DIR="work_dirs/setr-ade/setr_vit-l_naive_8xb2-160k_ade20k-${size}_imgresize"

    # Execute the test command
    echo "Running test for size $size"
    python tools/test.py "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --work-dir "${WORK_DIR}"
done