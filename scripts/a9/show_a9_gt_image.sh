DATASET_DIR="/Users/panda/Documents/dataset/A9"
DATASETS='/Users/panda/Documents/dataset/A9/dataSets/all.txt'
OUTPUT_DIR="./output/a9/gt_visual/image/3dbbox"
DRAW_TYPE="3dbbox"


python tools/a9/show_tool/show_a9_gt_image.py \
  --dataset_dir $DATASET_DIR \
  --dataSets $DATASETS \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE