LABEL_INFO_FILE="/mnt/data_cfl/Projects/Data/Rcooper/label/infos/136-137-138-139_8-info.json"
FRAME_ID=29

python tools/rcooper/dataprocess_tool/verify_calib.py \
  --label_info_file $LABEL_INFO_FILE \
  --frame_id $FRAME_ID