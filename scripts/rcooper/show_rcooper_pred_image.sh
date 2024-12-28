IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/139/seq-10/cam-0/1693909077.683672.jpg"
PRED_TXT_FILE="/mnt/data_cfl/Projects/MonoUNI/output/rope3d_eval_v3/data/139/seq-10/cam-0/1693909077.683672.txt"
OUTPUT_DIR="/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_visual"
DRAW_TYPE="3dbbox"

python tools/rcooper/show_tool/show_rcooper_pred_image.py \
  --img_file $IMG_FILE \
  --pred_txt_file $PRED_TXT_FILE \
  --output_dir $OUTPUT_DIR \
  --draw_type $DRAW_TYPE \
  --draw_score_threshold 0.1