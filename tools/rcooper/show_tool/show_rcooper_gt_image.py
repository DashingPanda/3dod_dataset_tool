import cv2
import numpy as np
import json
import argparse
import os

from lib.utils import common_utils, transformation
from lib.data_utils.data import Data
from lib.data_utils.post_processor import box_utils
from lib.data_utils.dataset.rcooper import labelloader as rcooper_labelloader
from lib.draw_utils import draw2d




def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--img_file', type=str, required=True,
                        default='/mnt/data_cfl/Projects/Data/Rcooper_original/data/106-105/105/seq-0/cam-0/1692685344.971904.jpg',
                        help='the directory of the image of the rcooper dataset')
    parser.add_argument('--gt_npy_file', type=str, required=False, 
                        default='/mnt/data_cfl/Projects/rcooper-dataset-tools/output/gt_npy/106-105/105/seq-0/1692685344.949990.npy',
                        help='the directory of a gt npy filer of the rcooper dataset label npy file')
    parser.add_argument('--gt_json_file', type=str, required=False, 
                        default='/mnt/data_cfl/Projects/rcooper-dataset-tools/output/gt_npy/106-105/105/seq-0/1692685344.949990.npy',
                        help='the directory of a gt json filer of the rcooper dataset label json file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--gt_file_type', type=str, required=True,
                        help='the type of the file of the gt, including npy and json')
    parser.add_argument('--draw_type', type=str, required=True, choices=['3dbbox', '3dbbox_center', '2dbbox'],
                        help='the type of the visualization of the gt, including 3dbbox, 3dbbox_center, 2dbbox')
    args = parser.parse_args()
    return args



def draw_gt_3dbox(img, gt, lid2cam_extri, cam_intri):
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
        box_obj_img = (cam_intri @ box_obj_cam.T).T
        box_obj_img = box_obj_img[:, :2] / box_obj_img[:, 2:]
        # draw the 2D bbox on the image
        for connect in connection:
            cv2.line(img, (int(box_obj_img[connect[0]][0]), int(box_obj_img[connect[0]][1])), 
                    (int(box_obj_img[connect[1]][0]), int(box_obj_img[connect[1]][1])), (0, 0, 255), 2)
    return img


def compute_2d_bounding_boxes_top_left_bottom_right(corners3d_img):
    """
    Compute 2D bounding boxes for N 3D bounding boxes projected onto a image,
    represented by the top-left and bottom-right points.

    Parameters:
    corners3d_img : np.ndarray, shape (N, 8, 2)
        An array represents the 8 corner points of N 3D bounding boxes in image coordinates.

    Returns:
    bounding_boxes_2d : np.ndarray, shape (N, 4, 2)
        An array contains the 4 corner points of the 2D bounding boxes for each 3D bounding box. 
    The order of the corners is: [top-left, top-right, bottom-right, bottom-left]

    Notes
    -----
    
        4 -------- 5           0 ---------- 1
       /|         /|           |            |
      7 -------- 6 .           |            |
      | |        | |   --->    |            |
      . 0 -------- 1           |            |
      |/         |/            |            |
      3 -------- 2             3 ---------- 2
    """
    # Compute min and max for each box
    x_min = np.min(corners3d_img[:, :, 0], axis=1)  # Minimum x-coordinate
    y_min = np.min(corners3d_img[:, :, 1], axis=1)  # Minimum y-coordinate
    x_max = np.max(corners3d_img[:, :, 0], axis=1)  # Maximum x-coordinate
    y_max = np.max(corners3d_img[:, :, 1], axis=1)  # Maximum y-coordinate

    # Create the 2D bounding boxes
    bounding_boxes_2d = np.stack([
        np.stack([x_min, y_min], axis=1),  # Top-left
        np.stack([x_max, y_min], axis=1),  # Top-right
        np.stack([x_max, y_max], axis=1),  # Bottom-right
        np.stack([x_min, y_max], axis=1)   # Bottom-left
    ], axis=1)

    return bounding_boxes_2d


def get_all_object_label_list(label_list: list) -> list:
    """
    Parameters
    ----------
    label_list : list, len(label_list) = N, where N is the number of objects in the label file
        Each label in label_list is a dictinary containing object data in the rcooper dataset label format.

    Notes
    -----
        The format of the label dict is: {'type': 'car', 'occluded_state': 0, 'truncated_state': 0, 'crowding': 0,
    'ignore': 0, 'track_id': 11, '3d_location': {'x': -36.54881450515726, 'y': 28.69501520664547,
    'z': -4.49969242101686}, '3d_dimensions': {'w': 2.0999991362500112, 'h': 1.4499998637499885,
    'l': 4.679112000351287}, 'rotation': 0.08186919162720498}
    """
    objs = [] # list of objects in the gt label file
    for label in label_list:
        data = Data()
        rcooper_labelloader.get_label_data(data, label)
        objs.append(data)
    return objs


def visualize_gt(img_file, gt_file, output_dir, draw_type, gt_file_type=None):
    scene_id = img_file.split('/')[8]
    camera_id = img_file.split('/')[10]
    print(f'scene_id: {scene_id}')
    print(f'camera_id: {camera_id}')
    assert scene_id in ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139']
    assert camera_id in ['cam-0', 'cam-1']

    cam_intri, lid2cam_extri, cam2lid_extri, lid2world_extri, world2lid_extri = rcooper_labelloader.get_transformation_matrix(scene_id, camera_id)

    img = cv2.imread(img_file)

    # =================================================== 3dbbox ===================================================

    if gt_file_type == 'npy' and draw_type == '3dbbox':
        gt = np.load(gt_file)
        print(f'the number of gt: {len(gt)}')
        img_visual = draw_gt_3dbox(img, gt, lid2cam_extri, cam_intri)

    elif gt_file_type == 'json' and draw_type == '3dbbox':
        gt_label_list = json.load(open(gt_file, 'r'))
        print(f'the number of gt: {len(gt_label_list)}')
        # load label from the gt label file
        objs = get_all_object_label_list(gt_label_list)# list of objects in the gt label file
        # check if there is no object in the image
        if len(objs) == 0:
            print(f'There is no valid object in the image: {img_file}')
            return
        # get the gt_location_dimensions_rotation in lidar coordinate system
        gt_location_dimensions_rotation = common_utils.get_location_dimensions_rotation(objs)
        # convert the object location, dimensions, and rotation to 3D corners
        corners3d_lid = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation, order='whl') # (N, 8, 3)
        corners3d_lid = corners3d_lid.reshape(-1, 3)  # (N*8, 3)
        corners3d_lid = common_utils.covert2homogeneous(corners3d_lid)  # (N, 4)
        # transform the 3D corners to the camera coordinate system
        corners3d_cam = transformation.lid2cam(corners3d_lid, lid2cam_extri)  # (N, 4)
        # transform the 3D corners to the image coordinate system
        corners3d_img = transformation.cam2img(corners3d_cam, cam_intri)  # (N, 2)
        corners3d_img = corners3d_img.reshape(-1, 8, 2)  # (N, 8, 2)
        gt = corners3d_img
        print(f'the number of gt: {len(gt)}')
        img_visual = draw2d.draw_3dbox(img, gt)

    # =================================================== 3dbbox_center ===================================================

    elif gt_file_type == 'json' and draw_type == '3dbbox_center':
        gt_label_list = json.load(open(gt_file, 'r'))
        print(f'the number of gt: {len(gt_label_list)}')
        # load label from the gt label file
        objs = get_all_object_label_list(gt_label_list)# list of objects in the gt label file
        # check if there is no object in the image
        if len(objs) == 0:
            print(f'There is no valid object in the image: {img_file}')
            return
        # get the gt_location_dimensions_rotation in lidar coordinate system
        gt_location_dimensions_rotation = common_utils.get_location_dimensions_rotation(objs)
        gt_location_lid = gt_location_dimensions_rotation[:, :3]  # (N, 3), location in lidar coordinate system
        # convert the gt_location_lid to homogeneous coordinates
        gt_location_lid = common_utils.covert2homogeneous(gt_location_lid)  # (N, 4)
        # transform the gt_location_lid to camera coordinate system from lidar coordinate system
        gt_location_cam = transformation.lid2cam(gt_location_lid, lid2cam_extri)  # (N, 4)
        # transform the gt_location_cam to image coordinate system from camera coordinate system
        gt_location_img = transformation.cam2img(gt_location_cam, cam_intri)  # (N, 2)
        img_visual = draw2d.draw_3dboxcenter_with_distence(img, gt_location_img, gt_location_lid)

    # =================================================== 2dbbox ===================================================

    elif draw_type == '2dbbox':
        gt_label_list = json.load(open(gt_file, 'r'))
        print(f'the number of gt: {len(gt_label_list)}')
        # load label from the gt label file
        objs = get_all_object_label_list(gt_label_list)# list of objects in the gt label file
        # check if there is no object in the image
        if len(objs) == 0:
            print(f'There is no valid object in the image: {img_file}')
            return
        # get the gt_location_dimensions_rotation in lidar coordinate system
        gt_location_dimensions_rotation = common_utils.get_location_dimensions_rotation(objs)
        # convert the object location, dimensions, and rotation to 3D corners
        corners3d_lid = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation, order='whl') # (N, 8, 3)
        corners3d_lid = corners3d_lid.reshape(-1, 3)  # (N*8, 3)
        corners3d_lid = common_utils.covert2homogeneous(corners3d_lid)  # (N, 4)
        # transform the 3D corners to the camera coordinate system
        corners3d_cam = transformation.lid2cam(corners3d_lid, lid2cam_extri)  # (N, 4)
        # transform the 3D corners to the image coordinate system
        corners3d_img = transformation.cam2img(corners3d_cam, cam_intri)  # (N, 2)
        corners3d_img = corners3d_img.reshape(-1, 8, 2)  # (N, 8, 2)
        corners2d_img = compute_2d_bounding_boxes_top_left_bottom_right(corners3d_img)
        img_visual = draw2d.draw_2dbox(img, corners2d_img)

    # =================================================== save visualised image ===================================================

    output_dir = os.path.join(output_dir, draw_type)
    common_utils.save_img(img_visual, output_dir, img_file, dataset='rcooper')


def main():
    args = args_parser()
    img_file = args.img_file
    gt_file_type = args.gt_file_type
    output_dir = args.output_dir
    draw_type = args.draw_type
    if gt_file_type == 'npy':
        gt_file = args.gt_npy_file
    elif gt_file_type == 'json':
        gt_file = args.gt_json_file
    else:
        raise ValueError('gt_file_type should be npy or json')

    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'img_file: {img_file}')
    print(f'gt_file: {gt_file}')
    print(f'output_dir: {output_dir}')
    print(f'gt_file_type: {gt_file_type}')
    print(f'draw_type: {draw_type}')

    # ======================== start visualization ========================
    print('======================== start visualization ========================')
    print('=====================================================================')
    visualize_gt(img_file, gt_file, output_dir, draw_type, gt_file_type)
    print('======================== end visualization ========================')
    print('===================================================================')
    print('\n')



if __name__ == '__main__':
    main()