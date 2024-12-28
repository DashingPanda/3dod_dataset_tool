IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/136/seq-8/cam-0/1693909042.081970.jpg"
PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/rope3d_eval/data/136/seq-8/cam-0/1693909042.081970.txt"
GT_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/136/seq-8/1693909042.083451.json"

# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/137/seq-8/cam-0/1693909042.093420.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/rope3d_eval/data/137/seq-8/cam-0/1693909042.093420.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/137/seq-8/1693909042.138970.json"

# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/138/seq-8/cam-0/1693909042.067551.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/rope3d_eval/data/138/seq-8/cam-0/1693909042.067551.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/138/seq-8/1693909042.140799.json"

# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/139/seq-8/cam-0/1693909042.083600.jpg"
# PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/rope3d_eval/data/139/seq-8/cam-0/1693909042.083600.txt"
# GT_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/139/seq-8/1693909042.084974.json"

# OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/pred_gt_visual/image/center"
# DRAW_TYPE="3dbbox_center"

# python tools/rcooper/show_tool/show_rcooper_pred_gt_image.py \
#   --img_file $IMG_FILE \
#   --pred_txt_file $PRED_TXT_FILE \
#   --gt_file $GT_FILE \
#   --output_dir $OUTPUT_DIR \
#   --draw_type $DRAW_TYPE



IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/139/seq-10/cam-0/1693909077.683672.jpg"
PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/rope3d_eval_v3/data/139/seq-10/cam-0/1693909077.683672.txt"
GT_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/139/seq-10/1693909077.685052.json"

OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/pred_gt_visual/image/3dbox"
DRAW_TYPE="3dbbox"
# OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/pred_gt_visual/image/center"
# DRAW_TYPE="3dbbox_center"

python tools/rcooper/show_tool/show_rcooper_pred_gt_image.py \
  --img_file $IMG_FILE \
  --pred_txt_file $PRED_TXT_FILE \
  --gt_file $GT_FILE \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE \
  --draw_score_threshold 0.1