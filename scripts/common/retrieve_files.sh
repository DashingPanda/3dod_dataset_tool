# FILE_FOLDER='/mnt/data_cfl/Projects/Data/Rope3D_data/label_2_4cls_for_train'
# DATASET='rope3d'

FILE_FOLDER='/gemini/data-1/Projects/Data/dg3d/3dobjectdetection/dgrope/version1/label'
DATASET='dg3d'

python tools/common/retrieve_files.py \
    --file_folder $FILE_FOLDER \
    --dataset $DATASET