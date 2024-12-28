import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Show the ground truth image of the RCooper dataset')
    parser.add_argument('--data_json', type=str, required=True, 
                        help='The path to a json file containing the image file names and the corresponding labels')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--draw_type', type=str, required=True, choices=['3dbbox', '3dbbox_center', '2dbbox'],
                        help='the type of the visualization of the gt, including 3dbbox, 3dbbox_center, 2dbbox')
    return parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()
