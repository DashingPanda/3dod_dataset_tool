IMG_FILE1="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/61831_fa2sd4a2West152_420_1625824638_1625825007_1_obstacle.jpg"
IMG_FILE2="/mnt/data_cfl/Projects/MonoUNI/lib/eval_tools/mask/2188.969252_mask.jpg"
OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/tmp"

python tools/tmp.py \
    --img_file1 $IMG_FILE1\
    --img_file2 $IMG_FILE2\
    --output_dir $OUTPUT_DIR