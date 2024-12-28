import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.append('/gemini/data-1/Projects/3dod-dataset-tools')

from lib.utils.common_utils import retrieve_label_files
from lib.utils import common_utils
from lib.data_utils.post_processor import box_utils
from lib.dataset.rcooper import labelloader





def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--label_dir', type=str, required=True,
                        help='the directory of the label of the rcooper dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    args = parser.parse_args()
    return args

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


def generate_save_ground_plane_equation_cam(object_bottom_coners_in_a_scene: np.ndarray, scene_id: str, 
                                            scene_id_intersection: list, scene_id_corridor: list, output_dir: str):
    if scene_id in scene_id_intersection:
        camera_id = 'cam-0'
        _, lid2cam_extri, _, _, _ = labelloader.get_transformation_matrix(scene_id, camera_id)
        coefficients = calculate_ground_plane_cam(object_bottom_coners_in_a_scene, lid2cam_extri)
        # save the ground plane equation in the output directory
        common_utils.save_ground_plane_equation_cam(coefficients, scene_id, camera_id, output_dir)

    elif scene_id in scene_id_corridor:
        for camera_id in ['cam-0', 'cam-1']:
            _, lid2cam_extri, _, _, _ = labelloader.get_transformation_matrix(scene_id, camera_id)
            coefficients = calculate_ground_plane_cam(object_bottom_coners_in_a_scene, lid2cam_extri)
            # save the ground plane equation in the output directory
            common_utils.save_ground_plane_equation_cam(coefficients, scene_id, camera_id, output_dir)
    else:
        print(f'unknown scene: {scene_id}')
        exit()
    # TODO: save the ground plane equation in the output directory


def generate_denorm(args: argparse.Namespace):
    """
    This script is used to generate the ground plane equations of the cameras in the rcooper dataset.
    To this end, need to retrieve the label of the rcooper dataset to get the location in Lidar coordinate 
    to calcuate the points in the ground, and transform them to the camera coordinate with the transformation matrix
    provided in the dataset folder. Then use these ground points, at least 50 objects bottom points, 
    to calculate the ground plane equation using the Least Squares Method.
    Finally, save the ground plane equation in the output directory.

    There are two kinds of scenes in the rcooper dataset, 'intersection' and 'corridor'.
    The 'intersection' scene includes scene id '117', '118', '119', '120', '136', '137', '138', and '139'.
    The 'corridor' scene involves scene id '105', '106', '115', and '116'. 
    Significantly, the 'intersection' scenes have one camera respectively, while the 'corridor' scenes have two cameras respectively.
    """
    label_dir = args.label_dir
    output_dir = args.output_dir
    label_files = retrieve_label_files(label_dir)

    scene_id_intersection = list(['117', '118', '119', '120', '136', '137', '138', '139'])
    scene_id_corridor = list(['105', '106', '115', '116'])

    scene_id_last = None
    object_bottom_coners_in_a_scene = np.empty((0, 4, 3)) # (N, 4, 3), N is the number of objects in a scene
    scene_processed = set() # set of scene ids that have been processed
    print(f'object_bottom_coners_in_a_scene: {object_bottom_coners_in_a_scene}')
    for label_file in tqdm(label_files):
        # check if the label file is a coop label file
        if 'coop' in label_file.split('/'):
            continue
        scene_id = labelloader.get_sceneid_from_filename(label_file, index_default=8)
        # check if the scene has been processed
        if scene_id in scene_processed:
            # print(f'skip scene: {scene_id}')
            continue
        # check if there are enough objects in a scene
        if object_bottom_coners_in_a_scene.shape[0] >= 50:
            print(f'enough objects in a scene: {scene_id}')
            generate_save_ground_plane_equation_cam(object_bottom_coners_in_a_scene, scene_id, 
                                                    scene_id_intersection, scene_id_corridor, output_dir)
            object_bottom_coners_in_a_scene = np.empty((0, 4, 3))
            scene_processed.add(scene_id_last)
        # check if the scene has changed
        if scene_id_last is not None and scene_id_last!= scene_id:
            print(f'change to a new scene: {scene_id}')
            # TODO: calculate the ground plane equation using the Least Squares Method
            # TODO: save the ground plane equation in the output directory
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

def main():
    args = args_parser()
    generate_denorm(args)

if __name__ == '__main__':
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.savefig('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/3D/3d_bbox.png')
    # exit()
    main()