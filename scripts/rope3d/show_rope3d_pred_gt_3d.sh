# DATASET_DIR="/mnt/data_cfl/Projects/Data/Rope3D_data"
# IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_customizeddatasetsplit_trial/data/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.txt"

DATASET_DIR="/mnt/data_cfl/Projects/Data/Rope3D_data"
IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1901_fa2sd4adatasetsj8fas14151_420_1629885634_1629885932_255_obstacle.jpg"
PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_customizeddatasetsplit_trial_epoch90/data/1901_fa2sd4adatasetsj8fas14151_420_1629885634_1629885932_255_obstacle.txt"
GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1901_fa2sd4adatasetsj8fas14151_420_1629885634_1629885932_255_obstacle.txt"


OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/pred_gt_visual/image/3dbox"
DRAW_TYPE="3dbbox"


python tools/rope3d/show_tool/show_rope3d_pred_gt_3d.py \
  --dataset_dir $DATASET_DIR \
  --img_file $IMG_FILE \
  --pred_txt_file $PRED_TXT_FILE \
  --gt_file $GT_FILE \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE \
  --draw_score_threshold 0.1