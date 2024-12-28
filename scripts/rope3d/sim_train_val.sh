DATASET_DIR='/mnt/data_cfl/Projects/Data/Rope3D_data/image_2'
TRAIN_IMAGE_TXT='/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/train.txt'
VAL_IMAGE_TXT='/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/val.txt'

python tools/rope3d/sim_train_val.py \
  --dataset_dir $DATASET_DIR\
  --train_image_txt $TRAIN_IMAGE_TXT \
  --val_image_txt $VAL_IMAGE_TXT