import cv2
import numpy as np
import json
import argparse
import os

import sys
sys.path.append('/mnt/data_cfl/Projects/rcooper-dataset-tools')

from lib.utils import common_utils, transformation
from lib.data_utils.post_processor import box_utils
from lib.draw_utils import draw2d
from lib.dataset.dairv2x import labelloader


def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--img_file', type=str, required=True,
                        default='/mnt/data_cfl/Projects/Data/Rcooper_original/data/106-105/105/seq-0/cam-0/1692685344.971904.jpg',
                        help='the directory of the image of the rcooper dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--gt_type', type=str, required=True, choices=['camera', 'virtuallidar'],
                        help='Labeled data in Infrastructure Virtual LiDAR Coordinate System fitting objects \
                              in image based on image frame time, or point cloud based on point cloud frame time.')
    parser.add_argument('--draw_type', type=str, required=True,
                        help='the type of the visualization of the gt, including 3dbbox, 3dbbox_center')
    parser.add_argument('--dataset', type=str, required=True, choices=['rcooper', 'dairv2x'],
                        help='The type of the dataset, including rcooper and dairv2x')
    opt = parser.parse_args()
    return opt



def draw_gt_3dbox(img, gt, lid2cam_extri, lid2cam_intri):
    """
    Parameters
    ----------
    img : np.ndarray

    gt : 3D bbox corner points in lidar coordinate system
        np.ndarray
        (N, 8, 3)
    """
    # conection relationship of the 3D bbox vertex
    connection = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    print(f'the number of gt: {len(gt)}')

    # draw the 3D bbox on the image
    for i, box_obj in enumerate(gt):
        box_obj = np.hstack((box_obj, np.ones((8,1))))
        box_obj_cam = (lid2cam_extri @ box_obj.T).T
        box_obj_cam = box_obj_cam[:, :3]
        box_obj_img = (lid2cam_intri @ box_obj_cam.T).T
        box_obj_img = box_obj_img[:, :2] / box_obj_img[:, 2:]
        # draw the 2D bbox on the image
        for connect in connection:
            cv2.line(img, (int(box_obj_img[connect[0]][0]), int(box_obj_img[connect[0]][1])), 
                    (int(box_obj_img[connect[1]][0]), int(box_obj_img[connect[1]][1])), (0, 0, 255), 2)
    return img


def draw_gt_3dboxcenter(img, gt, lid2cam_extri, lid2cam_intri):
    """
    Parameters
    ----------
    img : np.ndarray

    gt : 3D bbox center points in lidar coordinate system
        np.ndarray
        (N, 3)
    """
    print(f'lid2cam_extri:\n{lid2cam_extri}')
    print(f'lid2cam_intri:\n{lid2cam_intri}')

    # draw the 3D bbox center on the image
    for i, center in enumerate(gt):
        center_cam = (lid2cam_extri @ np.hstack((center, 1)).T).T
        print(f'center_cam: {center_cam}')
        center_cam = center_cam[:3]
        center_img = (lid2cam_intri @ center_cam).T
        print(f'center_img: {center_img}')
        center_img = center_img[:2] / center_img[2]
        print(f'center_img: {center_img}')
        cv2.circle(img, (int(center_img[0]), int(center_img[1])), 5, (0, 0, 255), -1)
    return img


def visualize_gt(img_file, gt_file, output_dir, draw_type, dataset):

    lid2cam_intri, lid2cam_extri, cam2lid_extri = labelloader.get_transformation_matrix(img_file)

    img = cv2.imread(img_file)


    if draw_type == '3dbbox':
        # get the object location, dimensions, and rotation from the label file
        gt_location_dimensions_rotation = labelloader.get_label_location_dimensions_rotation(gt_file) # (N, 7)
        if gt_location_dimensions_rotation is None:
            print(f'No gt found in {gt_file}')
            return
        # convert the object location, dimensions, and rotation to 3D corners
        corners3d_lid = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation, order='hwl') # (N, 8, 3)
        corners3d_lid = corners3d_lid.reshape(-1, 3)  # (N*8, 3)
        corners3d_lid = common_utils.covert2homogeneous(corners3d_lid)  # (N, 4)
        # transform the 3D corners to the camera coordinate system
        corners3d_cam = transformation.lid2cam(corners3d_lid, lid2cam_extri)  # (N, 4)
        # transform the 3D corners to the image coordinate system
        corners3d_img = transformation.cam2img(corners3d_cam, lid2cam_intri)  # (N, 2)
        corners3d_img = corners3d_img.reshape(-1, 8, 2)  # (N, 8, 2)
        gt = corners3d_img
        print(f'the number of gt: {len(gt)}')
        img_visual = draw2d.draw_3dbox(img, gt)

    elif draw_type == '3dbbox_center':
        gt = labelloader.get_label_location_dimensions_rotation(gt_file)
        print(f'the number of gt: {len(gt)}')
        gt = gt[:, :3] # only keep the center points
        img_visual = draw_gt_3dboxcenter(img, gt, lid2cam_extri, lid2cam_intri)

    common_utils.save_img(img_visual, output_dir, img_file, dataset)


def main():
    opt = args_parser()
    img_file = opt.img_file
    output_dir = opt.output_dir
    gt_type = opt.gt_type
    draw_type = opt.draw_type
    dataset = opt.dataset

    img_file_split = img_file.split('/single-infrastructure-side-image/')
    gt_file = os.path.join(img_file_split[0], 'single-infrastructure-side/label', gt_type, img_file_split[1].replace('.jpg', '.json'))

    

    # ============ visualization parameters ============
    print('============ visualization parameters ============')
    print(f'img_file: {img_file}')
    print(f'gt_file: {gt_file}')
    print(f'output_dir: {output_dir}')
    print(f'gt_type: {gt_type}')
    print(f'draw_type: {draw_type}')

    # ============ start visualization ============
    print('============ start visualization ============')
    visualize_gt(img_file, gt_file, output_dir, draw_type, dataset)
    print('============ end visualization ============')
    print('\n')



if __name__ == '__main__':
    main()