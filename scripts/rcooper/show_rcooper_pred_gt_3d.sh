IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/136/seq-8/cam-0/1693909042.081970.jpg"
PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/rope3d_eval_v3/data/136/seq-8/cam-0/1693909042.081970.txt"
GT_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/136/seq-8/1693909042.083451.json"
OUTPUT_DIR="/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_gt_visual/3d/center"
DRAW_TYPE="3dbbox_center"

python tools/rcooper/show_tool/show_rcooper_pred_gt_3d.py \
  --img_file $IMG_FILE \
  --pred_txt_file $PRED_TXT_FILE \
  --gt_file $GT_FILE \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE \
  --draw_score_threshold 0.1