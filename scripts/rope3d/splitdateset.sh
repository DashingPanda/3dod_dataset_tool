FILE_DIR="/mnt/data_cfl/Projects/Data/Rope3D_data/image_2"
FILE_TYPE="jpg"

python tools/rope3d/dataprocess_tool/splitdateset.py \
  --file_dir $FILE_DIR \
  --file_type $FILE_TYPE