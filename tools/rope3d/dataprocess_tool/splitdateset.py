import argparse
import os

import sys
sys.path.append('/gemini/data-1/Projects/3dod-dataset-tools')

from lib.utils import common_utils



def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--file_dir', type=str, required=True,
                        help='The directory will be searched to retrieve the files')
    parser.add_argument('--file_type', type=str, required=True, default=None,
                        help='The file type to be searched for in the directory')
    opt = parser.parse_args()
    return opt

def main():
    """
    The script is designed to split a dataset into training and validation sets, 
    with an emphasis on maintaining dissimilarity between the feature distributions of the two sets.
    """
    opt = args_parser()

    file_dir = opt.file_dir
    file_type = opt.file_type

    file_list = common_utils.retrieve_files(file_dir, file_type)
    print(len(file_list))

    scene_id_set = set()
    scene_num_dict = {} # key: scene_id, value: number of files in the scene
    for file in file_list:
        file_name = os.path.basename(file)
        # scene_id = '_'.join(file_name.split('_')[0:2])
        # scene_id = '_'.join(file_name.split('_')[0:1])
        scene_id = '_'.join(file_name.split('_')[1:2])
        # if scene_id not in scene_id_set:
        #     print(file)
        scene_id_set.add(scene_id)
        if scene_id not in scene_num_dict:
            scene_num_dict[scene_id] = 1
            print(file)
        else:
            scene_num_dict[scene_id] += 1

    for scene_id in scene_num_dict:
        print(scene_id, scene_num_dict[scene_id])

    print(len(scene_id_set))    


    # file_list.sort()
    # file_txt1 = '/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets_v2/train.txt'
    # file_txt2 = '/mnt/data_cfl/Projects/Data/Rope3D_data/ImageSets_v2/val.txt'
    # f1 = open(file_txt1, 'w')
    # f2 = open(file_txt2, 'w')
    # for i, file in enumerate(file_list):
    #     file_name = os.path.basename(file).replace('.jpg', '')
    #     if i < 31506:
    #         f1.write(file_name + '\n')
    #     else:
    #         f2.write(file_name + '\n')
    # f1.close()
    # f2.close()
    print('Done')


if __name__ == '__main__':
    main()