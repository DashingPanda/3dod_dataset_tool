import os
import cv2
import numpy as np
import argparse

from torch.utils.data import DataLoader

from lib.utils import common_utils, transformation
from lib.data_utils.dataset.a9.a9 import A9
from lib.data_utils.data import Data
from lib.data_utils.post_processor import box_utils
from lib.draw_utils import draw2d
from lib.data_utils.dataset.rope3d import labelloader as rope3d_labelloader


def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='the path to the directory containing the dataset')
    parser.add_argument('--dataSets', type=str, required=True,
                        help='a txt file containing the list of the data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--draw_type', type=str, required=True, choices=['3dbbox', '3dbbox_center', '3dbbox_Zcam'],
                        help='the type of the visualization of the predicted result, including 3dbbox, 3dbbox_center, 3dbbox_Zcam')
    args = parser.parse_args()
    return args


def calculate_distance(locations: np.ndarray, denorm: np.ndarray):
    print(f'locations: {locations.shape}') # (N, 4)
    print(f'denorm: {denorm.shape}') # (4, )

    a = np.dot(locations, denorm)
    print(f'a: {a.shape}')
    print(f'a: {a}')

    # calculate denorm[0] ** 2 + denorm[1] ** 2 + denorm[2] ** 2
    denominator = np.sum(denorm[:-1] ** 2)

    t = -denorm[-1] / denominator
    project_point = denorm[:-1] * t
    distance_camera = np.sqrt(np.sum(project_point ** 2))

    t = -(np.sum(a)) / denominator
    project_points = (denorm * t)[:-1] + locations[:, :-1] # (N, 3)
    distance_objs = np.sqrt(np.sum((project_points) ** 2, axis=1))

    distance = distance_objs - distance_camera
    print(f't: {t}')
    print(f'denorm: {denorm}')
    # print(f'project_points: {project_points}') 

    print(f'distance_camera: {distance_camera}')
    print(f'distance_objs: {distance_objs}')
    print(f'distance: {distance}')
    
    return distance
    


def visualize_gt(dataset_dir, dataSets, output_dir, draw_type):
    """
    Paratemers
    ----------
    dataset_dir : str
        the path to the directory containing the dataset
    img_file : str
        the path of the image of the rcooper dataset
    """

    # filename_woe = os.path.basename(img_file).split('.')[0] # filename without extension

    # cam_intri, cam2world_extri, world2cam_extri = rope3d_labelloader.get_transformation_matrix(filename_woe, dataset_dir)

    # print(f'cam_intri: {cam_intri}')
    # exit()

    # img = cv2.imread(img_file)
    
    if draw_type == '3dbbox_center':
        pass

    elif draw_type == '3dbbox':
        data_set = A9(root_dir=dataset_dir, dataset_txt=dataSets)


    # elif draw_type == '3dbbox':
    #     # ================ draw the gt 3d bounding box in image coordinate system ================
    #     gt_result_lines = open(gt_file, 'r').readlines()
    #     # load label from the predicted result file
    #     objs = [] # list of valid objects in the predicted result file
    #     for gt_result_line in gt_result_lines:
    #         gt_result_line = gt_result_line.strip()
            # gt_data = Data()
    #         rope3d_labelloader.get_label_data(gt_data, gt_result_line)
    #         if gt_data.obj_type not in ['car', 'van', 'bus', 'cyclist']: # only show 3d bounding box for car, van, bus, cyclist
    #             continue
    #         if gt_data.X == 0 and gt_data.Y == 0 and gt_data.Z == 0: # No 3d bounding box
    #             continue
    #         objs.append(gt_data)
    #     print(f'the number of gt objects: {len(objs)}')
    #     # get the gt_location_dimensions_rotation in camera coordinate system
    #     gt_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)

    #     gt_location_cam = gt_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
    #     gt_location_cam = common_utils.covert2homogeneous(gt_location_cam) # (N, 4), homogeneous coordinates
    #     gt_location_world = transformation.cam2wor(gt_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
    #     gt_location_world = gt_location_world[:, :3]  # (N, 3), location in world coordinate systems
    #     gt_rotation_cam = gt_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
    #     gt_rotation_world = transformation.cam2wor_rotation(gt_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
    #     gt_location_dimensions_rotation_world = np.concatenate([gt_location_world, gt_location_dimensions_rotation_cam[:, 3:6], gt_rotation_world], axis=1)  # (N, 7)
    #     # get the 8 corners of a 3d bounding box in world coordinate system
    #     gt_corners_world = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)
    #     gt_corners_world = gt_corners_world.reshape(-1, 3)  # (N*8, 3) in world coordinate system
    #     # convert the pred_corners_world to homogeneous coordinates
    #     gt_corners_world = common_utils.covert2homogeneous(gt_corners_world)  # (N*8, 4)
    #     # transform the pred_corners_worlde to camera coordinate system from world coordinate system
    #     gt_corners_cam = transformation.wor2cam(gt_corners_world, world2cam_extri)  # (N*8, 4)
    #     # transform the pred_corners_cam to image coordinate system from camera coordinate system
    #     gt_corners_img = transformation.cam2img(gt_corners_cam, cam_intri)  # (N*8, 2)
    #     gt_corners_img = gt_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
    #     # draw the predicted 3d bounding box in image coordinate system
    #     img_visual = draw2d.draw_3dbox(img, gt_corners_img)

    #     # ================ save the image with predicted and gt 3d bounding box ================
    #     common_utils.save_img(img_visual, output_dir, img_file, dataset='rope3d')

    
def main():
    args = args_parser()
    dataset_dir = args.dataset_dir
    dataSets = args.dataSets
    output_dir = args.output_dir
    draw_type = args.draw_type


    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'dataset_dir: {dataset_dir}')
    print(f'dataSets: {dataSets}')
    print(f'output_dir: {output_dir}')
    print(f'draw_type: {draw_type}')

    # ======================== start visualization ========================
    print('======================== start visualization ========================')
    print('=====================================================================')
    visualize_gt(dataset_dir, dataSets, output_dir, draw_type)
    print('======================== end visualization ========================')
    print('===================================================================')
    print('\n')



if __name__ == '__main__':
    main()