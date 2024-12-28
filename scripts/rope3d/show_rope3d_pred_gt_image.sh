# IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1901_fa2sd4adatasetsj8fas17151_420_1629891097_1629892096_134_obstacle.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_customizeddatasetsplit_trial/data/1901_fa2sd4adatasetsj8fas17151_420_1629891097_1629892096_134_obstacle.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1901_fa2sd4adatasetsj8fas17151_420_1629891097_1629892096_134_obstacle.txt"

# IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1901_fa2sd4adatasetsj8fas17151_420_1629891097_1629892096_134_obstacle.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_eval_official_customizeddatasetsplit/data/1901_fa2sd4adatasetsj8fas17151_420_1629891097_1629892096_134_obstacle.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1901_fa2sd4adatasetsj8fas17151_420_1629891097_1629892096_134_obstacle.txt"

# IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1679_fa2sd4adatasetNorth151_420_1616056928_1616057233_27_obstacle.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_customizeddatasetsplit_trial_epoch90/data/1679_fa2sd4adatasetNorth151_420_1616056928_1616057233_27_obstacle.txt"
# # PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_eval_official_customizeddatasetsplit/data/1679_fa2sd4adatasetNorth151_420_1616056928_1616057233_27_obstacle.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1679_fa2sd4adatasetNorth151_420_1616056928_1616057233_27_obstacle.txt"

# DATASET_DIR="/mnt/data_cfl/Projects/Data/Rope3D_data"
# IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_customizeddatasetsplit_trial_epoch90/data/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.txt"
# # PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_eval_official_customizeddatasetsplit/data/1679_fa2sd4adatasetNorth151_420_1616056928_1616057233_27_obstacle.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1901_fa2sd4adatasetf328h9k14camera151_420_1629885933_1629886230_177_obstacle.txt"

DATASET_DIR="/mnt/data_cfl/Projects/Data/Rope3D_data"
IMG_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2/1901_fa2sd4adatasetsj8fas17151_420_1629897100_1629898100_87_obstacle.jpg"
PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/eval/rope3d_customizeddatasetsplit_fa2sd4adatasetsj8fas17151/data/1901_fa2sd4adatasetsj8fas17151_420_1629897100_1629898100_87_obstacle.txt"
GT_FILE="/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1901_fa2sd4adatasetsj8fas17151_420_1629897100_1629898100_87_obstacle.txt"

OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/pred_gt_visual/image/3dbbox"
# OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/pred_gt_visual/image/3dbbox_Zcam"
DRAW_TYPE="3dbbox"
# DRAW_TYPE="3dbbox_Zcam"


python tools/rope3d/show_tool/show_rope3d_pred_gt_image.py \
  --dataset_dir $DATASET_DIR \
  --img_file $IMG_FILE \
  --pred_txt_file $PRED_TXT_FILE \
  --gt_file $GT_FILE \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE \
  --draw_score_threshold 0.1


