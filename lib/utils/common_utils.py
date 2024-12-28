import torch
import numpy as np
import os
import cv2
import warnings

# Customize the warnings format
def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{category.__name__}: {message}\n"

warnings.formatwarning = custom_warning_format

from tqdm import tqdm

import sys
sys.path.append('/mnt/data_cfl/Projects/3dod-dataset-tools')

from lib.data_utils.data import Data

# Disable scientific notation
torch.set_printoptions(sci_mode=False, precision=8)
np.set_printoptions(suppress=True)





def save_ground_plane_equation_cam(coefficients: tuple, output_dir: str, scene_id=None, camera_id=None, cam_intri_file=None, dataset=None):
    if dataset is None:
        output_path = os.path.join(output_dir.split('/output')[0], 'output/tmp', 'tmp.txt')
    elif dataset == 'rcooper':
        file_path = os.path.join(output_dir, f'{scene_id}_{camera_id}.txt')
    elif dataset == 'dairv2x':
        file_path = os.path.join(output_dir, os.path.basename(cam_intri_file).replace('.json', '.txt'))
    print(f'output_dir: {output_dir}')
    print(f'file_path: {file_path}')
    # chech the directory of the output file
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # save the ground plane equation in the output file
    with open(file_path, 'w') as f:
        f.write(f'{coefficients[0]} {coefficients[1]} {coefficients[2]} {coefficients[3]}')


def save_img(img, output_dir:str, img_file:str, dataset:str=None):
    """
    Save the image to the specified directory.

    This function saves the image to the specified directory, creating any
    necessary subdirectories named after the dataset. 
    If the dataset is not given, the image will be saved to the 'tmp' directory.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be saved.
    output_dir : str
        The path to the directory where the image will be saved.
    img_file : str
        The path to the image file.
    dataset : str, optional
        The name of the dataset. If not given, the image will be saved to the 'tmp' directory. 
        Supported options include: 
        - 'dairv2x'
        - 'rcooper'
        - 'rope3d'
    """
    if dataset is None:
        warnings.warn("(Warning from save_img function) Dataset is not specified. The image will be saved to the 'tmp' directory.", UserWarning)
        img_name = os.path.basename(img_file)
        output_path = os.path.join(output_dir.split('/output')[0], 'output/tmp', img_name)
    elif dataset == 'dairv2x':
        output_path = os.path.join(output_dir, img_file.split('single-infrastructure-side-image/')[-1])
    elif dataset == 'rcooper':
        output_path = os.path.join(output_dir, img_file.split('/data/')[-1])
    elif dataset == 'rope3d':
        output_path = os.path.join(output_dir, img_file.split('image_2/')[-1])

    print(f'output_path: {output_path}')
    # check if the output directory exists, if not, create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    cv2.imwrite(output_path, img)


def retrieve_label_files(label_dir: str, dataset: str) -> list:
    """
    Retrieve all label files from a specified directory.

    This function scans the specified directory and its subdirectories to
    collect all label files with a '.json' extension, excluding those that
    contain 'info' in their filenames. It is designed to work with the RCooper
    dataset or similar datasets where label files are organized in a nested directory structure.

    Parameters
    ----------
    label_dir : str
        The path to the directory containing label files. This directory can include
        subdirectories, and all will be recursively searched for valid label files.

    dataset : str
        The name of the dataset. Supported options include: 'rope3d', 'rcooper', 'dairv2x'.

    Returns
    -------
    list of str
        A list of paths to the label files found in the directory. Each entry in the list
        represents the full path to a '.json' file that does not contain 'info' in its name.

    Notes
    -----
    Rope3D:
        This function scans the specified directory and its subdirectories to
    collect all label files with a '.txt' extension. It is designed to work with the Rope3D
    dataset or similar datasets where label files are organized in a nested directory structure.

    RCooper:
        This function scans the specified directory and its subdirectories to
    collect all label files with a '.json' extension, excluding those that
    contain 'info' in their filenames. It is designed to work with the RCooper
    dataset or similar datasets where label files are organized in a nested directory structure.
    """

    assert dataset in ['rope3d', 'rcooper', 'dairv2x', 'dg3d'], f"Dataset {dataset} is not supported."

    if dataset == 'rope3d':
        label_files = list()
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.txt'):
                    label_files.append(os.path.join(root, file))
        return label_files

    elif dataset == 'rcooper':
        label_files = list()
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.json'):
                    if 'info' in file:
                        continue
                    label_files.append(os.path.join(root, file))
        return label_files

    elif dataset == 'dg3d':
        label_files = list()
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.json'):
                    label_files.append(os.path.join(root, file))
        return label_files


def retrieve_image_files(label_dir: str, dataset: str) -> list:
    """
    Retrieve all label files from a specified directory.

    This function scans the specified directory and its subdirectories to
    collect all label files with a '.json' extension, excluding those that
    contain 'info' in their filenames. It is designed to work with the RCooper
    dataset or similar datasets where label files are organized in a nested directory structure.

    Parameters
    ----------
    label_dir : str
        The path to the directory containing label files. This directory can include
        subdirectories, and all will be recursively searched for valid label files.

    dataset : str
        The name of the dataset. Supported options include: 'rope3d', 'rcooper', 'dairv2x'.

    Returns
    -------
    list of str
        A list of paths to the label files found in the directory. Each entry in the list
        represents the full path to a '.json' file that does not contain 'info' in its name.

    Notes
    -----
    Rope3D:
        This function scans the specified directory and its subdirectories to
    collect all label files with a '.txt' extension. It is designed to work with the Rope3D
    dataset or similar datasets where label files are organized in a nested directory structure.

    RCooper:
        This function scans the specified directory and its subdirectories to
    collect all label files with a '.json' extension, excluding those that
    contain 'info' in their filenames. It is designed to work with the RCooper
    dataset or similar datasets where label files are organized in a nested directory structure.
    """

    assert dataset in ['rope3d', 'rcooper', 'dairv2x'], f"Dataset {dataset} is not supported."

    if dataset == 'rope3d':
        pass

    elif dataset == 'rcooper':
        image_files = list()
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_files.append(os.path.join(root, file))
        return image_files



def retrieve_files(file_dir: str, file_type=None) -> list:
    """
    Retrieve all files from a specified directory.

    This function scans the specified directory and its subdirectories to
    collect all files.

    Parameters
    ----------
    file_dir : str
        The path to the directory containing files. This directory can include
        subdirectories, and all will be recursively searched for valid files.
    file_type : str
        The type of file to retrieve, specified as a string. For example, 'png', 'jpg' or 'json'.

    Returns
    -------
    list of str
        A list of paths to the files found in the directory. Each entry in the list
        represents the full path to a file of the specified type (if specified).
    """
    file_llist = list()
    for root, dirs, files in tqdm(os.walk(file_dir), desc="Retrieving files", colour='green'):
        for file in files:
            if file_type is not None:
                if file.endswith(f'.{file_type}'):
                    if 'info' in file:
                        continue
                    file_llist.append(os.path.join(root, file))
            else:
                file_llist.append(os.path.join(root, file))
    return file_llist


def get_location_dimensions_rotation(objs: list):
    """
    get the object location, dimension, and rotation from the predicted result file.

    Parameters
    ----------
    objs: list, (N)
        A list of objects predicted by the model or labelled by annotators.

    Returns
    -------
    ldr_numpy : (N, 7), the numpy array of the object location in the camera coordinate, dimension, and rotation.
        - location : (x, y, z)
        - dimension : (h, w, l)
        - rotation

    score : (N), the score of the predicted object.
    """

    ldr_numpy = list()
    for obj in objs:
        location = [obj.X, obj.Y, obj.Z]
        dimension = [obj.h, obj.w, obj.l]
        rotation = [obj.yaw]
        ldr_numpy.append(location + dimension + rotation)
    ldr_numpy = np.array(ldr_numpy)
    return ldr_numpy


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).double(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Rotate 3D points along the z-axis by a specified angle.

    This function applies a rotation transformation to a set of 3D points
    around the z-axis. The rotation angle is specified in radians, and the
    rotation is performed for each batch of points independently.

    Parameters
    ----------
    points : torch.Tensor or np.ndarray, shape (B, N, 3 + C)
        A batch of points to be rotated. B is the batch size, N is the number
        of points per batch, and C represents any additional features associated
        with each point. The first three dimensions (x, y, z) are rotated.
    angle : torch.Tensor or np.ndarray, shape (B,)
        A batch of rotation angles in radians. Each angle specifies the rotation
        for the corresponding batch of points. The angle increases in the
        counter-clockwise direction, following the right-hand rule (x => y).

    Returns
    -------
    points_rot : torch.Tensor or np.ndarray, shape (B, N, 3 + C)
        The rotated points. The shape is the same as the input `points`, with the
        (x, y, z) coordinates rotated along the z-axis. Any additional features
        (C) remain unchanged.

    Notes
    -----
    The rotation transformation is defined by the following rotation matrix:

        [ cos(angle)  sin(angle)  0 ] \n
        [ -sin(angle) cos(angle)  0 ] \n
        [ 0           0           1 ] \n

    This matrix rotates points in the xy-plane by the specified angle, leaving
    the z-coordinate unchanged.

    Example
    -------
    >>> points = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # shape (1, 2, 3)
    >>> angle = torch.tensor([np.pi / 2])  # 90 degrees in radians
    >>> rotate_points_along_z(points, angle)
    tensor([[[0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0]]])
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).double()
    points_rot = torch.matmul(points[:, :, 0:3].double(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def covert2homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Convert points from non-homogeneous coordinates to homogeneous coordinates.

    This function adds an additional dimension to the input points, setting the 
    last coordinate to 1, thereby converting them into homogeneous coordinates.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        N points in non-homogeneous coordinates, represented as (x, y, z).

    Returns
    -------
    points_homogeneous : np.ndarray, shape (N, 4)
        N points in homogeneous coordinates, represented as (x, y, z, 1).

    Notes
    -----
    This transformation is commonly used in 3D geometry and computer vision to
    facilitate affine transformations (translation, rotation, scaling) using
    matrix multiplication.

    The transformation is as follows:
    
    Non-homogeneous coordinates:   [x, y, z]
    Homogeneous coordinates:       [x, y, z, 1]

    Example
    -------
    >>> points = np.array([[1, 2, 3], [4, 5, 6]])
    >>> convert2homogeneous(points)
    array([[1., 2., 3., 1.],
           [4., 5., 6., 1.]])
    """

    points_homogeneous = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return points_homogeneous
