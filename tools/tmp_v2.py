import os

# txt_file = '/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets_v1/train.txt'
txt_file = '/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets_v1/val_original.txt'

scene_id_set = set()

with open(txt_file, 'r') as f:
    for line in f:
        line_split_list = line.strip().split('_')
        scene_id = '_'.join(line_split_list[1:2])
        scene_id_set.add(scene_id)

print(len(scene_id_set))
print(scene_id_set)


image_folder = '/mnt/data_cfl/Projects/Data/Rope3D_data/image_2'

# retrieve all the images in the image folder
image_list = list()
for file in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file)
    image_list.append(file_path)

print(len(image_list))

# scene2image_dict = {}

# for file in image_list:
#     file_name = os.path.basename(file)
#     line_split_list = file_name.split('_')
#     scene_id = '_'.join(line_split_list[0:5])
#     if scene_id not in scene2image_dict:
#         scene2image_dict[scene_id] = file

# print(len(scene2image_dict))

# for scene_id in scene2image_dict:
#     file = scene2image_dict[scene_id]
#     file_name = os.path.basename(file)
#     target_file = os.path.join('/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/scenes', file_name)
#     # copy the image to the target directory
#     os.system(f'cp {file} {target_file}')
