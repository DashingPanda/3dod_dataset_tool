# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/lable_0.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/label_0_tmp1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/lable_1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/lable_2.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/label_2_tmp1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/lable_3.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/label_3_tmp1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/lable_4.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/label_4_tmp1.txt'

# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_0.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_0_1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_1_1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_2.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_2_1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_3.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_3_1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_4.txt'
FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_4_1.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_5.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_6.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_7.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_8.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2_9.txt'
# FILES_TXT='/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/label_tmp.txt'

DATASET_DIR='/mnt/data_cfl/Projects/Data/Rope3D_data'

python tools/rope3d/dataprocess_tool/3Dcudedepth_withtxt.py \
    --files_txt $FILES_TXT \
    --dataset_dir $DATASET_DIR