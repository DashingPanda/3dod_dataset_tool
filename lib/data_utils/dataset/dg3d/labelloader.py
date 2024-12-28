import os
import numpy as np
import yaml
from pyquaternion import Quaternion

import sys
sys.path.append('/mnt/data_cfl/Projects/3dod-dataset-tools')

from lib.data_utils.data import Data

np.set_printoptions(suppress=True)





def get_label_data(data: Data, label: str):
    """
    Extract object data from a label line.

    Parameters
    ----------
    data : Data (from lib.data_utils.data import Data)
        An instance of the Data class, which will be updated with object data.
    label : str
        A string containing object data in the rope3d dataset label format.

    Notes
    -----
    The format of the label line is: class, truncation, occlusion, alpha, 
    box2d (left_top_x, left_top_y, right_bottom_x, right_bottom_y),  
    3d_dimensions (h, w, l), 3d_location in camera coordinates (x, y, z), yaw, score (optional)
    """
    label_split_list = label.split(' ')
    fields = label_split_list
    data.obj_type = fields[0].lower()  # object type [car, pedestrian, cyclist, ...]
    data.truncation = float(fields[1])  # truncation [0,1,2]
    data.occlusion = int(float(fields[2]))  # occlusion  [0,1,2]
    data.x1 = int(float(fields[4]))  # left   [px] (2d box)
    data.y1 = int(float(fields[5]))  # top    [px] (2d box)
    data.x2 = int(float(fields[6]))  # right  [px] (2d box)
    data.y2 = int(float(fields[7]))  # bottom [px] (2d box)
    data.h = float(fields[8])  # height [m]
    data.w = float(fields[9])  # width  [m]
    data.l = float(fields[10])  # length [m]
    data.X = float(fields[11])  # X [m] (3d box center)
    data.Y = float(fields[12])  # Y [m] (3d box center)
    data.Z = float(fields[13])  # Z [m] (3d box center)
    data.yaw = float(fields[14])  # yaw angle [rad]
    if len(fields) >= 16:
        data.score = float(fields[15])  # detection score
    else:
        data.score = 1


def get_intrinsics_matrix(filename_woe: str, dataset_dir: str) -> tuple:
    """
    Retrieve the camera's intrinsic matrix from the calibration file.

    Parameters
    ----------
    filename_woe : str
        The base name without the extension of the data file, 
        e.g. '1632_fa2sd4a11North_420_1612431546_1612432197_1_obstacle'
    
    dataset_dir : str
        The path to the directory containing the dataset
    
    Returns
    -------
    tuple
        (fx, fy, cx, cy, s)
        - fx: float, the focal length in the x-direction.
        - fy: float, the focal length in the y-direction.
        - cx: float, the x-coordinate of the principal point.
        - cy: float, the y-coordinate of the principal point.
        - s: float, the skew factor.
    """
    calib_file_path = os.path.join(dataset_dir, 'calib', filename_woe + '.txt')
    cam_intri_line = open(calib_file_path, 'r').readlines()[0].strip()
    ccam_intri_line = cam_intri_line.split(' ')
    cam_intri = [float(e) for e in ccam_intri_line[1:]]
    fx, s, cx, _, _, fy, cy, _, _, _, _, _, = cam_intri
    return fx, fy, cx, cy, s


def get_transformation_matrix(filename_woe: str, dataset_dir: str):
    """
    Retrieve transformation matrices between camera, and world coordinate systems for a given image file.

    Parameters
    ----------
    filename_woe : str
        Filename without extension, e.g. '1632_fa2sd4a11North_420_1612431546_1612432197_1_obstacle'
    dataset_dir : str
        The path to the directory containing the dataset

    Returns
    -------
    tuple
        (cam_intri, cam2world_extri, world2cam_extri)
        - cam_intri: (3, 3) numpy array, the camera's intrinsic matrix.
        - cam2world_extri: (4, 4) numpy array, the transformation matrix from the camera to the world.
        - world2cam_extri: (4, 4) numpy array, the transformation matrix from the world to the camera.

    Notes
    -----
    The definition of a camera's intrinsic matrix (cam_intri):

    [[fx, s, cx], \n
     [0, fy, cy], \n
     [0, 0, 1]] \n

        - fx: float, the focal length in the x-direction.
        - fy: float, the focal length in the y-direction.
        - cx: float, the x-coordinate of the principal point.
        - cy: float, the y-coordinate of the principal point.
        - s: float, the skew factor.
    """
    calib_dir = os.path.join(dataset_dir, 'calib')
    extrinsics_dir = os.path.join(dataset_dir, 'extrinsics')
    calibe_file = os.path.join(calib_dir, filename_woe + '.txt')
    extrinsics_file = os.path.join(extrinsics_dir, filename_woe + '.yaml')
    print(f'calibe_file: {calibe_file}')
    print(f'extrinsics_file: {extrinsics_file}')

    # ====== camera intrinsic matrix ======
    cam_intri_line = open(calibe_file, 'r').readlines()[0].strip()
    ccam_intri_line = cam_intri_line.split(' ')
    cam_intri = [float(e) for e in ccam_intri_line[1:]]
    cam_intri = np.array(cam_intri).reshape(3, 4)
    cam_intri = cam_intri[:3, :3]

    # ====== camera to world transformation matrix ======
    # x = yaml.safe_load(open(extrinsics_file, 'r'))
    text_file = open(extrinsics_file, 'r')
    cont = text_file.read()
    x = yaml.safe_load(cont)
    r = x['transform']['rotation']
    t = x['transform']['translation']
    q = Quaternion(r['w'], r['x'], r['y'], r['z'])
    m = q.rotation_matrix
    m = np.matrix(m).reshape((3, 3))
    t = np.matrix([t['x'], t['y'], t['z']]).T
    cam2world_extri = np.vstack((np.hstack((m, t)), np.array([0, 0, 0, 1])))

    # print(f'cam2world_extri: {cam2world_extri.shape}')
    # cam2world_extri = np.array(cam2world_extri.I)
    world2cam_extri = np.array(cam2world_extri.I)
    # print(f'cam2world_extri: {cam2world_extri.shape}')

    # ====== world to camera transformation matrix ======
    # world2cam_extri = np.linalg.inv(cam2world_extri)
    cam2world_extri = np.linalg.inv(world2cam_extri)


    return cam_intri, cam2world_extri, world2cam_extri
