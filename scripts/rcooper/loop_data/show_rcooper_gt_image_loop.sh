DATA_JSON=""
OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/loop_data/gt_visual"
DRAW_TYPE="2dbbox"

python tools/rcooper/show_tool/loop_data/show_rcooper_gt_image_loop.py  \
    --data_json $DATA_JSON \
    --output_dir $OUTPUT_DIR \
    --draw_type $DRAW_TYPE