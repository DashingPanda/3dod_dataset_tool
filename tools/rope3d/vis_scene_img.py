import argparse
import os

import sys
sys.path.append('/mnt/data_cfl/Projects/3dod-dataset-tools')

from lib.data_utils.dataset.rope3d import datahelper




def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='The directory of the rope3d dataset')
    parser.add_argument('--train_image_txt', type=str, required=True,
                        help='The Txt file path of the image list of the rope3d dataset for training')
    parser.add_argument('--val_image_txt', type=str, required=True,
                        help='The Txt file path of the image list of the rope3d dataset for validation')
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
    dataset_dir = opt.dataset_dir
    train_image_txt = opt.train_image_txt
    val_image_txt = opt.val_image_txt
    output_dir = opt.output_dir

    print('======================== parse arguments ========================')
    print('dataset_dir:', dataset_dir)
    print('train_image_txt:', train_image_txt)
    print('val_image_txt:', val_image_txt)
    print('output_dir:', output_dir)

    # ======================== visualize scene images ========================
    print('======================== visualize scene images ========================')
    train_image_list = list() # list of image paths for training
    for line in open(train_image_txt, 'r'):
        path = os.path.join(dataset_dir, line.strip() + '.jpg')
        train_image_list.append(path)
    val_image_list = list() # list of image paths for validation
    for line in open(val_image_txt, 'r'):
        path = os.path.join(dataset_dir, line.strip() + '.jpg')
        val_image_list.append(path)

    vis_output_dir = os.path.join(output_dir, 'train')
    datahelper.show_scene_img(train_image_list, vis_output_dir)
    vis_output_dir = os.path.join(output_dir, 'val')
    datahelper.show_scene_img(val_image_list, vis_output_dir)



if __name__ == '__main__':
    main()