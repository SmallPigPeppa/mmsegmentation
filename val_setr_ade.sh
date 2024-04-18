# Vanilla
for size in 512x512 256x256 128x128
do
    # Set variables based on the current size
    CONFIG_FILE="configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-${size}.py"
    CHECKPOINT_FILE="ckpt/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth"
    WORK_DIR="work_dirs/setr-ade/setr_vit-l_naive_8xb2-160k_ade20k-${size}_vanilla"

    # Execute the test command
    echo "Running test for size $size"
    python tools/test.py "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --work-dir "${WORK_DIR}"
done

# bi resize
for size in 256x256 128x128
do
    # Set variables based on the current size
    CONFIG_FILE="configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-${size}_bipi.py"
    CHECKPOINT_FILE="ckpt/ms_ckpt/setr_naive_${size}_160k_b16_ade20k_20210619_191258-061f24f5_bi.pth"
    WORK_DIR="work_dirs/setr-ade/setr_vit-l_naive_8xb2-160k_ade20k-${size}_bi"

    # Execute the test command
    echo "Running test for size $size"
    python tools/test.py "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --work-dir "${WORK_DIR}"
done


# pi resize
for size in 256x256 128x128
do
    # Set variables based on the current size
    CONFIG_FILE="configs/setr/setr_vit-l_naive_8xb2-160k_ade20k-${size}_bipi.py"
    CHECKPOINT_FILE="ckpt/ms_ckpt/setr_naive_${size}_160k_b16_ade20k_20210619_191258-061f24f5_pi.pth"
    WORK_DIR="work_dirs/setr-ade/setr_vit-l_naive_8xb2-160k_ade20k-${size}_pi"

    # Execute the test command
    echo "Running test for size $size"
    python tools/test.py "${CONFIG_FILE}" "${CHECKPOINT_FILE}" --work-dir "${WORK_DIR}"
done
