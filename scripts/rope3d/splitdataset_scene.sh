DATASET_IMAGE_TXT='/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets/train.txt'
OUTPUT_DIR='/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/splitdataset_scene/train'

python tools/rope3d/dataprocess_tool/splitdataset_scene.py \
  --dataset_image_txt $DATASET_IMAGE_TXT \
  --output_dir $OUTPUT_DIR