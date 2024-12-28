import argparse
import os

import sys
sys.path.append('/mnt/data_cfl/Projects/3dod-dataset-tools')

from lib.data_utils.dataset.rope3d import datahelper




def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--dataset_image_txt', type=str, required=True,
                        help='The Txt file path of the image list of the rope3d dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='The directory of the visualized scene images')
    opt = parser.parse_args()
    return opt


def main():
    """
    This script is used to visualize the scene images of the rope3d dataset.
    """
    # ======================== parse arguments ========================
    opt = args_parser()
    dataset_image_txt = opt.dataset_image_txt
    output_dir = opt.output_dir

    print('======================== parse arguments ========================')
    print('dataset_image_txt:', dataset_image_txt)
    print('output_dir:', output_dir)

    # ======================== visualize scene images ========================
    print('======================== visualize scene images ========================')
    image_list = list() # list of image paths for training
    for line in open(dataset_image_txt, 'r'):
        line = line.strip()
        image_list.append(line)

    print('Number of images:', len(image_list))
    datahelper.split_dataset_scene(image_list, output_dir)



if __name__ == '__main__':
    main()