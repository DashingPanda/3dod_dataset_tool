LABEL_DIR="/mnt/data_cfl/Projects/Data/Rcooper/label"
IMAGE_DIR="/mnt/data_cfl/Projects/Data/Rcooper/data"
OUTPUT_DIR="/mnt/data_cfl/Projects/rcooper-dataset-tools/output/rcooper/cropped_images"

python tools/rcooper/dataprocess_tool/crop2dbox.py \
  --label_dir $LABEL_DIR \
  --image_dir $IMAGE_DIR \
  --output_dir $OUTPUT_DIR \