import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.append('/gemini/data-1/Projects/rcooper-dataset-tools')

from lib.utils.common_utils import retrieve_label_files
from lib.utils import common_utils
from lib.data_utils.post_processor import box_utils
from lib.dataset.dairv2x import labelloader





def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--label_dir', type=str, required=True,
                        help='the directory of the label of the dairv2x dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the dairv2x dataset lael npy file')
    parser.add_argument('--dataset', type=str, required=True, choices=['rcooper', 'dairv2x'],
                        help='The type of the dataset, including rcooper and dairv2x')
    opt = parser.parse_args()
    return opt


def get_sceneid2filelist_mapping(filename2sceneid_mapping: dict) -> dict:
    """
    This function is used to get the mapping from sceneid to the filelist of the dairv2x dataset.

    Parameter
    ---------
    filename2sceneid_mapping: dict, the mapping from filename to sceneid

    Return
    ------
    sceneid2filelist_mapping: dict, the mapping from sceneid to the filelist of the dairv2x dataset
    """
    sceneid2filelist_mapping = dict()
    for filename in filename2sceneid_mapping:
        sceneid = filename2sceneid_mapping[filename]
        if sceneid not in sceneid2filelist_mapping:
            sceneid2filelist_mapping[sceneid] = []
        sceneid2filelist_mapping[sceneid].append(filename)
    return sceneid2filelist_mapping




def fit_plane_least_squares(points: np.ndarray) -> np.ndarray:
    """
    This function is used to fit a plane using the Least Squares Method. \
    The plane equation is: ax + by + (-1)z + d = 0, significantly the coefficient before z is -1.

    Args:
        points: (N, -1), the N points (x, y, z, ...) in the plane
    
    Returns:
        coeff: coefficients of the plane equation (a, b, d)
    """
    # construct the matrix A
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    # construct the vector B
    B = points[:, 2]
    # solve the least squares problem
    coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return coeff


def calculate_ground_plane_cam(bottom_points_lidar: np.ndarray, transform_matrix: np.ndarray) -> tuple:
    """
    This function is used to calculate the ground plane equation using the Least Squares Method.
    
    Args:
        bottom_points_lidar: (N, 4, 3), the N object bottom points in the ground in the lidar coordinate
        transform_matrix: (4, 4), the transformation matrix from the lidar coordinate to the camera coordinate
    
    Returns:
        a tuple of the ground plane equation (a, b, c, d)
    """
    bottom_points_lidar = bottom_points_lidar.reshape((-1, 3)) # (N*4, 3)
    # calculate the points in the camera coordinate
    ones = np.ones((bottom_points_lidar.shape[0], 1))

    print(f'the shape of bottom_points_lidar is {bottom_points_lidar.shape}')
    print(f'the shape of transform_matrix is {transform_matrix.shape}')

    bottom_points_camera = np.concatenate((bottom_points_lidar, ones), axis=1) # (N*4, 4)
    bottom_points_camera = np.dot(transform_matrix, bottom_points_camera.T).T # (N*4, 4)

    coeffs = fit_plane_least_squares(bottom_points_camera)
    a, b, d = coeffs
    c = -1
    # transform the coefficients into float64
    a, b, c, d = map(float, [a, b, c, d])

    # ========== debug ==========
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # for point in bottom_points_lidar:
    #     x, y, z = point
    #     ax.scatter(x, y, z, color='blue', marker='o', alpha=0.7)
    # plt.savefig('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/3D/3d_lidar.png')
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # for point in bottom_points_camera:
    #     x, y, z, _ = point
    #     ax.scatter(x, y, z, color='blue', marker='o', alpha=0.7)
    # plt.savefig('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/3D/3d_camera.png')
    # plt.close()
    
    # coeffs = fit_plane_least_squares(bottom_points_camera)
    # a, b, d = coeffs
    # c = -1

    # # draw the ground plane in the camera coordinate
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # for point in bottom_points_camera:
    #     x, y, z, _ = point
    #     ax.scatter(x, y, z, color='blue', marker='o', alpha=0.7)
    # x = np.arange(-10, 10, 0.1)
    # y = np.arange(-10, 10, 0.1)
    # X, Y = np.meshgrid(x, y)
    # Z = a*X + b*Y + d
    # ax.plot_surface(X, Y, Z, alpha=0.2)
    # plt.savefig('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/3D/3d_plane.png')
    # plt.close()

    # exit()

    return a, b, c, d


def generate_save_ground_plane_equation_cam(object_bottom_coners_in_a_scene: np.ndarray, output_dir: str, img_file: str, cam_intri_file_list: list, dataset: str):
    _, lid2cam_extri, _, = labelloader.get_transformation_matrix(img_file)
    coefficients = calculate_ground_plane_cam(object_bottom_coners_in_a_scene, lid2cam_extri)
    # save the ground plane equation in the output directory
    for cam_intri_file in cam_intri_file_list:
        common_utils.save_ground_plane_equation_cam(coefficients, output_dir, cam_intri_file=cam_intri_file, dataset=dataset)




def generate_denorm(opt: argparse.Namespace):
    """
    This script is used to generate the ground plane equations of the cameras in the rcooper dataset.
    To this end, need to retrieve the label of the rcooper dataset to get the location in Lidar coordinate 
    to calcuate the points in the ground, and transform them to the camera coordinate with the transformation matrix
    provided in the dataset folder. Then use these ground points, at least 50 objects bottom points, 
    to calculate the ground plane equation using the Least Squares Method.
    Finally, save the ground plane equation in the output directory.

    There are two kinds of scenes in the rcooper dataset, 'intersection' and 'corridor'.
    Significantly, the 'intersection' scenes have one camera respectively, while the 'corridor' scenes have two cameras respectively.
    """
    label_dir = opt.label_dir
    output_dir = opt.output_dir
    dataset = opt.dataset
    label_files = retrieve_label_files(label_dir)

    cam_intri_dir = os.path.join(label_dir.split('/label')[0], 'calib', 'camera_intrinsic')

    print('============ visualization parameters ============')
    print(f'label_dir: {label_dir}')
    print(f'cam_intri_dir: {cam_intri_dir}')
    print(f'output_dir: {output_dir}')
    print(f'dataset: {dataset}')


    filename2sceneid_mapping = labelloader.get_filename2sceneid_mapping(cam_intri_dir)

    sceneid2filelist_mapping = get_sceneid2filelist_mapping(filename2sceneid_mapping) # get the mapping from sceneid to the filelist of the dairv2x dataset


    scene_id_last = None
    object_bottom_coners_in_a_scene = np.empty((0, 4, 3)) # (N, 4, 3), N is the number of objects in a scene
    scene_processed = set() # set of scene ids that have been processed

    for scene_id in sceneid2filelist_mapping:
        if scene_id in scene_processed:
            continue
        for file_name in sceneid2filelist_mapping[scene_id]:
            label_file = os.path.join(label_dir, file_name)
            img_file = os.path.join(label_dir.split('/label')[0] + '-image', file_name.replace('.json', '.jpg'))
            # check if the scene has been processed
            if scene_id in scene_processed:
                # print(f'skip scene: {scene_id}')
                break
            # check if there are enough objects in a scene
            if object_bottom_coners_in_a_scene.shape[0] >= 50:
                print(f'enough objects in a scene: {scene_id}')
                cam_intri_file_list = sceneid2filelist_mapping[scene_id]
                generate_save_ground_plane_equation_cam(object_bottom_coners_in_a_scene, output_dir, img_file, cam_intri_file_list, dataset)
                object_bottom_coners_in_a_scene = np.empty((0, 4, 3))
                scene_processed.add(scene_id_last)
            # check if the scene has changed
            if scene_id_last is not None and scene_id_last!= scene_id:
                print(f'change to a new scene: {scene_id}')
                cam_intri_file_list = sceneid2filelist_mapping[scene_id]
                generate_save_ground_plane_equation_cam(object_bottom_coners_in_a_scene, output_dir, img_file, cam_intri_file_list, dataset)
                object_bottom_coners_in_a_scene = np.empty((0, 4, 3))
                scene_processed.add(scene_id_last)
            scene_id_last = scene_id
            # get the object location, dimensions, and rotation from the label file
            gt_location_dimensions_rotation = labelloader.get_label_location_dimensions_rotation(label_file)
            if gt_location_dimensions_rotation is None:
                continue
            # convert the object location, dimensions, and rotation to 3D corners
            corners3d = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation, order='whl')
            # get the corners in the box bottoms
            bottom_corners = corners3d[:, :4, :]
            object_bottom_coners_in_a_scene = np.concatenate((object_bottom_coners_in_a_scene, bottom_corners), axis=0)

    # print(f'object_bottom_coners_in_a_scene: {object_bottom_coners_in_a_scene}')
    # for label_file in tqdm(label_files):
    #     print(f'label_file: {label_file}')
    #     scene_id = labelloader.get_sceneid_from_filename(label_file, filename2sceneid_mapping)
    #     # check if the scene has been processed
    #     if scene_id in scene_processed:
    #         # print(f'skip scene: {scene_id}')
    #         continue
    #     # check if there are enough objects in a scene
    #     if object_bottom_coners_in_a_scene.shape[0] >= 50:
    #         print(f'enough objects in a scene: {scene_id}')
    #         generate_save_ground_plane_equation_cam(object_bottom_coners_in_a_scene, output_dir)
    #         object_bottom_coners_in_a_scene = np.empty((0, 4, 3))
    #         scene_processed.add(scene_id_last)
    #     # check if the scene has changed
    #     if scene_id_last is not None and scene_id_last!= scene_id:
    #         print(f'change to a new scene: {scene_id}')
    #         # TODO: calculate the ground plane equation using the Least Squares Method
    #         # TODO: save the ground plane equation in the output directory
    #         object_bottom_coners_in_a_scene = np.empty((0, 4, 3))
    #         scene_processed.add(scene_id_last)
    #     scene_id_last = scene_id
    #     # get the object location, dimensions, and rotation from the label file
    #     gt_location_dimensions_rotation = labelloader.get_label_location_dimensions_rotation(label_file)
    #     if gt_location_dimensions_rotation is None:
    #         continue
    #     # convert the object location, dimensions, and rotation to 3D corners
    #     corners3d = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation, order='whl')
    #     # get the corners in the box bottoms
    #     bottom_corners = corners3d[:, :4, :]
    #     object_bottom_coners_in_a_scene = np.concatenate((object_bottom_coners_in_a_scene, bottom_corners), axis=0)

def main():
    opt = args_parser()
    generate_denorm(opt)

if __name__ == '__main__':
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # exit()
    main()