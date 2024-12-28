FILES_TXT='/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/train.txt'
DATASET_DIR="/mnt/data_cfl/Projects/Data/Rope3D_data"
OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/gt_visual/image/3dbbox"
DRAW_TYPE="3dbbox"

python tools/rope3d/show_tool/withtxt/show_rope3d_gt_image.py \
  --file_txt $FILES_TXT \
  --dataset_dir $DATASET_DIR \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE