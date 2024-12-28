import cv2
import numpy as np
import os
import argparse

import sys
sys.path.append('/mnt/data_cfl/Projects/3dod-dataset-tools')

from lib.utils import common_utils, transformation
from lib.data_utils.data import Data
from lib.data_utils.post_processor import box_utils
from lib.draw_utils import draw3d
from lib.data_utils.dataset.rope3d import labelloader as rope3d_labelloader


def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='the path to the directory containing the dataset')
    parser.add_argument('--img_file', type=str, required=True,
                        default='/mnt/data_cfl/Projects/Data/Rcooper_original/data/106-105/105/seq-0/cam-0/1692685344.971904.jpg',
                        help='the directory of the image of the rcooper dataset')
    parser.add_argument('--pred_txt_file', type=str, required=False, 
                        help='the directory of a predict result txt file')
    parser.add_argument('--gt_file', type=str, required=False, 
                        help='the directory of a predict result file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--draw_type', type=str, required=True,
                        help='the type of the visualization of the predicted result, including 3dbbox, 3dbbox_center')
    parser.add_argument('--draw_score_threshold', type=float, required=True, default=0.1,
                        help='show the predicted result with score greater than the threshold')
    opt = parser.parse_args()
    return opt


def visualize_pred_gt(dataset_dir, img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold):
    """
    Paratemers
    ----------
    dataset_dir : str
        the path to the directory containing the dataset
    img_file : str
        the path of the image of the rcooper dataset
    """

    filename_woe = os.path.basename(img_file).split('.')[0] # filename without extension

    cam_intri, cam2world_extri, world2cam_extri = rope3d_labelloader.get_transformation_matrix(filename_woe, dataset_dir)

    # print(f'cam_intri: {cam_intri}')
    # exit()

    img = cv2.imread(img_file)
    
    if draw_type == '3dbbox_center':
        pass

    elif draw_type == '3dbbox':
        # =================== get the predicted 3d bounding box in world coordinate system ===================
        pred_result_lines = open(pred_txt_file, 'r').readlines()
        print(f'the number of pred_result_lines: {len(pred_result_lines)}')
        # load label from the predicted result file
        objs = [] # list of valid objects in the predicted result file
        for pred_result_line in pred_result_lines:
            pred_result_line = pred_result_line.strip()
            pred_data = Data()
            rope3d_labelloader.get_label_data(pred_data, pred_result_line)
            if pred_data.score < draw_score_threshold:
                continue
            objs.append(pred_data)
        # get the pred_location_dimensions_rotation_cam in camera coordinate system
        pred_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)
        print(f'the number of predicted objects after filtering: {len(objs)}')
        # transform the pred_location_dimensions_rotation_cam to lidar coordinate system from camera coordinate system
        pred_location_cam = pred_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        pred_location_cam = common_utils.covert2homogeneous(pred_location_cam) # (N, 4), homogeneous coordinates
        pred_location_world = transformation.cam2wor(pred_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
        pred_location_world = pred_location_world[:, :3]  # (N, 3), location in world coordinate systems
        pred_rotation_cam = pred_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
        pred_rotation_world = transformation.cam2wor_rotation(pred_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
        pred_location_dimensions_rotation_world = np.concatenate([pred_location_world, pred_location_dimensions_rotation_cam[:, 3:6], pred_rotation_world], axis=1)  # (N, 7)
        # get the 8 corners of a 3d bounding box in world coordinate system
        pred_corners_world = box_utils.boxes_to_corners_3d(pred_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)

        # =================== get the gt 3d bounding box in world coordinate system ===================
        gt_result_lines = open(gt_file, 'r').readlines()
        # load label from the predicted result file
        objs = [] # list of valid objects in the predicted result file
        for gt_result_line in gt_result_lines:
            gt_result_line = gt_result_line.strip()
            gt_data = Data()
            rope3d_labelloader.get_label_data(gt_data, gt_result_line)
            if gt_data.obj_type not in ['car', 'van', 'bus', 'cyclist']: # only show 3d bounding box for car, van, bus, cyclist
                continue
            if gt_data.X == 0 and gt_data.Y == 0 and gt_data.Z == 0: # No 3d bounding box
                continue
            objs.append(gt_data)
        print(f'the number of gt objects: {len(objs)}')
        # get the gt_location_dimensions_rotation in lidar coordinate system
        gt_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)
        # transform the pred_location_dimensions_rotation_cam to lidar coordinate system from camera coordinate system
        gt_location_cam = gt_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        gt_location_cam = common_utils.covert2homogeneous(gt_location_cam) # (N, 4), homogeneous coordinates
        gt_location_world = transformation.cam2wor(gt_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
        gt_location_world = gt_location_world[:, :3]  # (N, 3), location in world coordinate systems
        gt_rotation_cam = gt_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
        gt_rotation_world = transformation.cam2wor_rotation(gt_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
        gt_location_dimensions_rotation_world = np.concatenate([gt_location_world, gt_location_dimensions_rotation_cam[:, 3:6], gt_rotation_world], axis=1)  # (N, 7)
        # get the 8 corners of a 3d bounding box in world coordinate system
        gt_corners_world = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)
        
        # =================== draw the predicted and gt 3d bounding box in world coordinate system ===================
        draw3d.draw_3dbox_pred_gt(pred_corners_world, gt_corners_world)


        

def main():
    opt = args_parser()
    dataset_dir = opt.dataset_dir
    img_file = opt.img_file
    pred_txt_file = opt.pred_txt_file
    gt_file = opt.gt_file
    output_dir = opt.output_dir
    draw_type = opt.draw_type
    draw_score_threshold = opt.draw_score_threshold


    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'dataset_dir: {dataset_dir}')
    print(f'img_file: {img_file}')
    print(f'pred_txt_file: {pred_txt_file}')
    print(f'gt_file: {gt_file}')
    print(f'output_dir: {output_dir}')
    print(f'draw_type: {draw_type}')
    print(f'draw_score_threshold: {draw_score_threshold}')

    # ======================== start visualization ========================
    print('======================== start visualization ========================')
    print('=====================================================================')
    visualize_pred_gt(dataset_dir, img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold)
    print('======================== end visualization ========================')
    print('===================================================================')
    print('\n')



if __name__ == '__main__':
    main()