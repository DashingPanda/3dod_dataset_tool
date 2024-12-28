IMG_FILE="/mnt/data_cfl/Projects/Data/DAIR-V2X-I/single-infrastructure-side-image/000000.jpg"
OUTPUT_DIR="/mnt/data_cfl/Projects/rcooper-dataset-tools/output/dairv2x/gt_visual"
DRAW_TYPE="3dbbox"
# GT_TYPE="camera"
GT_TYPE="virtuallidar"
DATASET='dairv2x'

python tools/dairv2x/show_tool/show_dairv2x_gt_image.py \
  --img_file $IMG_FILE \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE \
  --gt_type $GT_TYPE \
  --dataset $DATASET


  


