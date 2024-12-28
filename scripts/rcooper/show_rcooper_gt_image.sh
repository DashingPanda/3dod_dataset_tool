# # IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/106-105/105/seq-0/cam-0/1692685344.971904.jpg"
# # IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/106-105/106/seq-0/cam-0/1692685345.022704.jpg"
# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/136/seq-0/cam-0/1693908913.315240.jpg"
# # GT_NPY_FILE="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/gt_npy/106-105/105/seq-0/1692685344.949990.npy"
# # GT_NPY_FILE="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/gt_npy/106-105/106/seq-0/1692685344.949958.npy"
# GT_NPY_FILE="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/gt_npy/136-137-138-139/136/seq-0/1693908928.183244.npy"
# OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/gt_visual"

# python tools/rcooper/show_tool/show_rcooper_gt_image.py \
#   --img_file $IMG_FILE \
#   --gt_npy_file $GT_NPY_FILE \
#   --output_dir $OUTPUT_DIR 



# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/136/seq-0/cam-0/1693908913.315240.jpg"
# GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/136/seq-0/1693908913.283240.json"
# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/137/seq-0/cam-0/1693908913.326783.jpg"
# GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/137/seq-0/1693908913.339286.json"
# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/139/seq-8/cam-0/1693909042.083600.jpg"
# GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/139/seq-8/1693909042.084974.json"
# OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/gt_visual"
# GT_FILE_TYPE="json"
# DRAW_TYPE="3dbbox"

# python tools/rcooper/show_tool/show_rcooper_gt_image.py \
#   --img_file $IMG_FILE \
#   --gt_json_file $GT_JSON_FILE \
#   --output_dir $OUTPUT_DIR \
#   --gt_file_type $GT_FILE_TYPE \
#   --draw_type $DRAW_TYPE



# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/136/seq-0/cam-0/1693908913.315240.jpg"
# GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/136/seq-0/1693908913.283240.json"
# OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/gt_visual"
# GT_FILE_TYPE="json"
# DRAW_TYPE="3dbbox"

# python rcooper/show_tool/show_rcooper_gt_image.py \
#   --img_file $IMG_FILE \
#   --gt_json_file $GT_JSON_FILE \
#   --output_dir $OUTPUT_DIR \
#   --gt_file_type $GT_FILE_TYPE \
#   --draw_type $DRAW_TYPE






# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/136/seq-8/cam-0/1693909033.315322.jpg"
# GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/136/seq-8/1693909033.283361.json"

IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/137/seq-8/cam-0/1693909033.327045.jpg"
GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/137/seq-8/1693909033.339384.json"

# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/138/seq-8/cam-0/1693909033.334348.jpg"
# GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/138/seq-8/1693909033.341186.json"

# IMG_FILE="/mnt/data_cfl/Projects/Data/Rcooper/data/136-137-138-139/139/seq-8/cam-0/1693909033.317079.jpg"
# GT_JSON_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/136-137-138-139/139/seq-8/1693909033.284770.json"

OUTPUT_DIR="/mnt/data_cfl/Projects/3dod-dataset-tools/output/rcooper/gt_visual"
GT_FILE_TYPE="json"
# DRAW_TYPE="3dbbox"
# DRAW_TYPE="3dbbox_center"
DRAW_TYPE="2dbbox"

python tools/rcooper/show_tool/show_rcooper_gt_image.py \
  --img_file $IMG_FILE \
  --gt_json_file $GT_JSON_FILE \
  --output_dir $OUTPUT_DIR \
  --gt_file_type $GT_FILE_TYPE \
  --draw_type $DRAW_TYPE

