import argparse
import numpy as np
import json
import os
from tqdm import tqdm

import sys
sys.path.append('/gemini/data-1/Projects/rcooper-dataset-tools')

from lib.data_utils.post_processor import box_utils
from lib.utils.common_utils import retrieve_label_files, get_gt_location_dimensions_rotation




def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--label_dir', type=str, required=True,
                        help='the directory of the label of the rcooper dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    opt = parser.parse_args()
    return opt


def save_npy(output_dir, npy_file_name, npy_data):
    output_path = os.path.join(output_dir, npy_file_name.split('label/')[-1].replace('.json', '.npy'))
    # check if the output_path directory exists, if not, create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    np.save(output_path, npy_data)


def generate_gt_npy(opt):
    label_dir = opt.label_dir
    output_dir = opt.output_dir
    label_files = retrieve_label_files(label_dir)
    for label_file in tqdm(label_files):

        # get the object location, dimensions, and rotation from the label file
        gt_location_dimensions_rotation = get_gt_location_dimensions_rotation(label_file)
        if gt_location_dimensions_rotation is None:
            continue
        
        # convert the object location, dimensions, and rotation to 3D corners
        corners3d = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation, order='whl')
        save_npy(output_dir, label_file, corners3d)
        


def main():
    opt = args_parser()
    generate_gt_npy(opt)

if __name__ == '__main__':
    main()