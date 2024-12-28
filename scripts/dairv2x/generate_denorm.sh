LABEL_DIR="/mnt/data_cfl/Projects/Data/DAIR-V2X-I/single-infrastructure-side/label/virtuallidar"
OUTPUT_DIR="/mnt/data_cfl/Projects/rcooper-dataset-tools/output/dairv2x/denorm"
DATASET='dairv2x'

python tools/dairv2x/dataprocess_tool/generate_denorm.py \
  --label_dir $LABEL_DIR \
  --output_dir $OUTPUT_DIR \
  --dataset $DATASET