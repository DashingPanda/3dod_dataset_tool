LABEL_DIR="/mnt/data_cfl/Projects/Data/Rcooper/label"
OUTPUT_DIR="/mnt/data_cfl/Projects/rcooper-dataset-tools/output/rcooper/denorm"

python tools/rcooper/dataprocess_tool/generate_denorm.py \
  --label_dir $LABEL_DIR \
  --output_dir $OUTPUT_DIR \