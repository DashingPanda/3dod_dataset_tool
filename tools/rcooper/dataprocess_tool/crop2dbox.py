import os
import sys
import cv2
import argparse

from tqdm import tqdm


from lib.utils import common_utils




def args_parse():
    parser = argparse.ArgumentParser(description='crop image according to 2d box')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='the directory of the label of the rcooper dataset')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='the directory of the image of the rcooper dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset cropped image')
    args = parser.parse_args()
    return args


def crop_image_2dbox(label_dir: str,image_dir: str, output_dir: str):
    """
    This function is used to crop the image according to the 2d box.

    Parameters
    ----------
    label_dir : str
        The directory of the label of the rcooper dataset
    image_dir : str
        The directory of the image of the rcooper dataset
    output_dir : str
        The directory of the output folder of the rcooper dataset cropped image
    """
    label_files = common_utils.retrieve_label_files(label_dir, 'rcooper')
    label_files.sort()
    image_files = common_utils.retrieve_image_files(image_dir, 'rcooper')
    image_files.sort()
    index = 0   
    print(len(label_files))
    print(len(image_files))
    for (label_file, image_file) in tqdm(zip(label_files, image_files), desc='crop image according to 2d box', colour='green'):
        if index == 10:
            break   
        # check if the label file is a coop label file
        if 'coop' in label_file.split('/'):
            continue
        print(f'label_file: {label_file}')  
        print(f'image_file: {image_file}')
        index += 1


def main():
    args = args_parse()
    label_dir = args.label_dir
    image_dir = args.image_dir
    output_dir = args.output_dir

    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'label_dir: {label_dir}')
    print(f'image_dir: {image_dir}')
    print(f'output_dir: {output_dir}')
    # ======================== start cropping ========================
    print('======================== start cropping ========================')
    print('================================================================')
    crop_image_2dbox(label_dir, image_dir, output_dir)


if __name__ == '__main__':
    main()