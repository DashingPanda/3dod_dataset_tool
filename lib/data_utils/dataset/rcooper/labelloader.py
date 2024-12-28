import numpy as np
import json

import sys
sys.path.append('/mnt/data_cfl/Projects/3dod-dataset-tools')

from lib.data_utils.data import Data



def get_label_data(data: Data, label: dict):
    """
    Extract object data from a label dict.

    Parameters
    ----------
    data : Data
        An instance of the Data class, which will be updated with object data.
    label : dict
        A dictinary containing object data in the rcooper dataset label format.

    Notes
    -----
    The format of the label dict is: {'type': 'car', 'occluded_state': 0, 'truncated_state': 0, 'crowding': 0, 
    'ignore': 0, 'track_id': 11, '3d_location': {'x': -36.54881450515726, 'y': 28.69501520664547, 
    'z': -4.49969242101686}, '3d_dimensions': {'w': 2.0999991362500112, 'h': 1.4499998637499885, 
    'l': 4.679112000351287}, 'rotation': 0.08186919162720498}
    """
    
    fields = label
    data.obj_type = fields['type'].lower()  # object type [car, pedestrian, cyclist, ...]
    data.truncation = float(fields['truncated_state'])  # truncation [0..1]
    data.occlusion = int(float(fields['occluded_state']))  # occlusion  [0,1,2]
    data.h = float(fields['3d_dimensions']['h'])  # height [m]
    data.w = float(fields['3d_dimensions']['w'])  # width  [m]
    data.l = float(fields['3d_dimensions']['l'])  # length [m]
    data.X = float(fields['3d_location']['x'])  # X [m] (3d box center)
    data.Y = float(fields['3d_location']['y'])  # Y [m] (3d box center)
    data.Z = float(fields['3d_location']['z'])  # Z [m] (3d box center)
    data.yaw = float(fields['rotation'])  # yaw angle [rad]



def get_sceneid_from_filename(filename: str, index_default: int = 0) -> str:
    """
    Extract the scene ID from a given filename.

    This function searches through a provided filename to identify and return the 
    scene ID. The scene ID corresponds to one of the predefined options from the 
    RCooper dataset. If the scene ID is not found at the specified default index, 
    the function searches through the entire filename to locate a valid scene ID.

    Parameters
    ----------
    filename : str
        The full path or name of the file, containing the scene ID within its structure.
    index_default : int, optional
        The default index to look for the scene ID in the split components of the filename.
        If not found at this index, the function searches the entire filename. Default is 0.

    Returns
    -------
    scene_id : str
        The extracted scene ID, which is one of the following:
        ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139'].

    Notes
    -----
    The filename is expected to be structured in a way where one of the components 
    (obtained by splitting the filename by '/') matches one of the predefined scene IDs.

    Example
    -------
    >>> get_sceneid_from_filename("path/to/label/136-137-138-139/136/seq-0/1693908913.283240.json", index_default=4)
    '136'

    In this example, the function identifies '136' as the scene ID from the third 
    component of the filename.
    """
    option_scene_id = ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139']
    index_scene_id = index_default
    filename_split = filename.split('/')
    scene_id = filename_split[index_scene_id]
    if scene_id in option_scene_id:
        return scene_id
    else:
        scene_id = [filename_split[index_scene_id] for index_scene_id in range(len(filename_split)) if filename_split[index_scene_id] in option_scene_id][0]
        return scene_id


def get_label_location_dimensions_rotation(json_file: str, return_type_trackid=False) -> np.ndarray:
    """
    Extract object location, dimensions, and rotation from a JSON label file.

    This function reads a label file in JSON format, extracts the 3D location, dimensions, 
    and rotation for each object, and returns them in a structured numpy array format.

    Parameters
    ----------
    json_file : str
        The path to the label file containing object data in JSON format.

    return_type_trackid : bool, optional
        If True, returns an additional numpy array with object types and track IDs. Default is False.


    Returns
    -------
    gt_location_dimensions_rotation : np.ndarray, shape (N, 7)
        A numpy array where each row corresponds to an object, and the columns represent:
        - (x, y, z) : 3D location in the LiDAR coordinate system.
        - (width, height, length) : Dimensions of the object.
        - rotation : The rotation angle around the vertical axis.
        
    gt_type_trackid : np.ndarray, shape (N, 2), optional
        If `return_type_trackid` is True, an additional array with object types and track IDs is returned.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    if len(data) == 0:
        return None

    gt_location_dimensions_rotation = list()
    gt_type_trackid = list()
    for object in data:
        # print(object)
        location = object['3d_location']
        dimensions = object['3d_dimensions']
        rotation = object['rotation']
        type = object['type']
        trackid = object['track_id']
        # add the all values of the dict of the locaton, dimensions, rotation, type and trackid to a list
        location = list(location.values())
        dimensions = list(dimensions.values())
        rotation = [rotation]
        type = [type]
        trackid = [trackid]
        gt_location_dimensions_rotation.append(location + dimensions + rotation)
        gt_type_trackid.append(type + trackid)
    gt_location_dimensions_rotation = np.array(gt_location_dimensions_rotation)
    gt_type_trackid = np.array(gt_type_trackid)
    if return_type_trackid:
        return gt_location_dimensions_rotation, gt_type_trackid
    return gt_location_dimensions_rotation


def get_transformation_matrix(scene_id: str, camera_id: str=None):
    """
    Retrieve transformation matrices between LiDAR, camera, and world coordinate systems for a given scene.

    This function returns the transformation matrices necessary for converting coordinates between the 
    LiDAR, camera, and world coordinate systems based on specified scene and optional camera IDs. The function
    supports scenes from the RCooper dataset and provides both intrinsic and extrinsic matrices based on
    calibration data files.

    Parameters
    ----------
    scene_id : str
        The scene ID from the RCooper dataset. Valid options are:
        ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139'].
    camera_id : str, optional
        The camera ID from the RCooper dataset, which is required for retrieving camera-related matrices. 
        Valid options are: ['cam-0', 'cam-1']. If not provided, only LiDAR to world transformations are returned.

    Returns
    -------
    tuple
        If `camera_id` is provided, returns a tuple:
            (cam_intri, lid2cam_extri, cam2lid_extri, lid2world_extri, world2lid_extri)
        - cam_intri: (3, 3) numpy array, the camera's intrinsic matrix.
        - lid2cam_extri: (4, 4) numpy array, the transformation matrix from the LiDAR to the camera.
        - cam2lid_extri: (4, 4) numpy array, the transformation matrix from the camera to the LiDAR.
        - lid2world_extri: (4, 4) numpy array, the transformation matrix from the LiDAR to the world.
        - world2lid_extri: (4, 4) numpy array, the transformation matrix from the world to the LiDAR.
        
        If `camera_id` is not provided, returns a tuple:
            (lid2world_extri, world2lid_extri)
        - lid2world_extri: (4, 4) numpy array, the transformation matrix from the LiDAR to the world.
        - world2lid_extri: (4, 4) numpy array, the transformation matrix from the world to the LiDAR.

    Raises
    ------
    AssertionError
        If the `scene_id` or `camera_id` is invalid.
    """

    assert scene_id in ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139'], "Invalid scene ID. Valid options are: ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139']"

    # lidar to world
    json_file = f'/mnt/data_cfl/Projects/Data/Rcooper/calib/lidar2world/{scene_id}.json'
    lid2world_calib = json.load(open(json_file, 'r'))
    lid2world_calib_R = lid2world_calib['rotation']
    lid2world_calib_R = np.array(lid2world_calib_R)
    lid2world_calib_T = lid2world_calib['translation']
    lid2world_calib_T = np.array(lid2world_calib_T)
    lid2world_extri = np.hstack((lid2world_calib_R, lid2world_calib_T.reshape(3,1)))
    row = np.array([0., 0., 0., 1.])
    lid2world_extri = np.vstack((lid2world_extri, row))
    # world to lidar
    world2lid_extri = np.linalg.inv(lid2world_extri)

    if not camera_id is None:
        assert camera_id in ['cam-0', 'cam-1'], "Invalid camera ID. Valid options are: ['cam-0', 'cam-1']"
        # lidar to camera
        json_file = f'/mnt/data_cfl/Projects/Data/Rcooper/calib/lidar2cam/{scene_id}.json'
        lid2cam_calib = json.load(open(json_file, 'r'))
        lid2cam_calib = lid2cam_calib[camera_id.replace('-', '_')]
        lid2cam_extri = lid2cam_calib['extrinsic']
        cam_intri = lid2cam_calib['intrinsic']
        lid2cam_extri = np.array(lid2cam_extri)
        cam_intri = np.array(cam_intri)
        # camera to lidar
        cam2lid_extri = np.linalg.inv(lid2cam_extri)

        return cam_intri, lid2cam_extri, cam2lid_extri, lid2world_extri, world2lid_extri
    
    else:
        return lid2world_extri, world2lid_extri