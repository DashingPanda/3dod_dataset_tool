DATASET_DIR='/mnt/data_cfl/Projects/Data/Rope3D_data/image_2'
TRAIN_IMAGE_TXT='/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/train.txt'
VAL_IMAGE_TXT='/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/val.txt'
OUTPUT_DIR='/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/vis_scenes'

python tools/rope3d/vis_scene_img.py \
  --dataset_dir $DATASET_DIR\
  --train_image_txt $TRAIN_IMAGE_TXT \
  --val_image_txt $VAL_IMAGE_TXT \
  --output_dir $OUTPUT_DIR