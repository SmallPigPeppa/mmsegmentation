# IMG resize
for size in 384x384
do
    # Set variables based on the current size
    CONFIG_FILE="configs/setr/setr_vit-l_naive_8xb1-80k_cityscapes-${size}_imgresize.py"
    CHECKPOINT_FILE="ckpt/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth"
    WORK_DIR="work_dirs/setr-city/setr_vit-l_naive_8xb1-80k_cityscapes-${size}_imgresize"

    # Execute the test command
    echo "Running test for size $size"
    python tools/test.py "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --work-dir "${WORK_DIR}"
done