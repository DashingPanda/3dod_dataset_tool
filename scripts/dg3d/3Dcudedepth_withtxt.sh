FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/dg3d/label_0.txt'
DATASET_DIR='/gemini/data-1/Projects/Data/dg3d/3dobjectdetection/dgrope/version1'

python tools/dg3d/dataprocess_tool/3Dcudedepth_withtxt.py \
    --files_txt $FILES_TXT \
    --dataset_dir $DATASET_DIR