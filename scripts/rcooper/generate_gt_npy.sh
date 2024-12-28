LABEL_DIR="/mnt/data_cfl/Projects/Data/Rcooper_original/label"
OUTPUT_DIR="/mnt/data_cfl/Projects/rcooper-dataset-tools/output/gt_npy"

python dataprocess_tool/generate_gt_npy.py \
  --label_dir $LABEL_DIR \
  --output_dir $OUTPUT_DIR \