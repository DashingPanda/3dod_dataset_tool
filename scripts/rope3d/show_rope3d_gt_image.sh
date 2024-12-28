DATASET_DIR="/mnt/data_cfl/Projects/Data/Rope3D_data"
IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.jpg"
GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.txt"

OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/gt_visual/image/3dbbox"
DRAW_TYPE="3dbbox"


python tools/rope3d/show_tool/show_rope3d_gt_image.py \
  --dataset_dir $DATASET_DIR \
  --img_file $IMG_FILE \
  --gt_file $GT_FILE \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE