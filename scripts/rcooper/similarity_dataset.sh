DATASET1_TXT="/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/train.txt"
DATASET2_TXT="/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/val_original.txt"
LOG_FILE='/mnt/data_cfl/Projects/dairv2xi-dataset-tools/doc/log7.txt'

python tools/rcooper/dataprocess_tool/similarity_dataset.py \
  --dataset1_txt $DATASET1_TXT \
  --dataset2_txt $DATASET2_TXT \
  --log_file $LOG_FILE