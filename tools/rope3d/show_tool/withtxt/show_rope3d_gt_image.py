import os
import argparse

from tools.rope3d.show_tool.show_rope3d_gt_image import visualize_gt



def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--file_txt', type=str, required=True,
                        help='the path to the txt file containing the dataset')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='the path to the directory containing the dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--draw_type', type=str, required=True, choices=['3dbbox', '3dbbox_center', '3dbbox_Zcam'],
                        help='the type of the visualization of the predicted result, including 3dbbox, 3dbbox_center, 3dbbox_Zcam')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    file_txt = args.file_txt
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    draw_type = args.draw_type

    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'dataset_dir: {dataset_dir}')
    print(f'output_dir: {output_dir}')
    print(f'draw_type: {draw_type}')

    # ======================== start visualization ========================
    print('======================== start visualization ========================')
    print('=====================================================================')
    file_list = list()
    with open(file_txt, 'r') as f:
        for line in f:
            file_list.append(line.strip())
    for file in file_list:
        print(f'file: {file}')
        img_file = os.path.join(dataset_dir, 'image_2', file + '.jpg')
        gt_file = os.path.join(dataset_dir, 'label_2', file + '.txt')
        print(f'img_file: {img_file}')
        print(f'gt_file: {gt_file}')
        visualize_gt(dataset_dir, img_file, gt_file, output_dir, draw_type)
        print('==================================')


if __name__ == '__main__':
    main()