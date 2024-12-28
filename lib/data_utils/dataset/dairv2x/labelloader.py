import numpy as np
import json
import os




def get_sceneid_from_filename(filename: str, filename2sceneid_mapping: dict) -> str:
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
    filename = os.path.basename(filename)
    scene_id = filename2sceneid_mapping[filename]
    print(f'scene_id: {scene_id}')
    return scene_id
    

def get_filename2sceneid_mapping(cam_intri_dir: str) -> dict:
    """
    Generate a mapping of filenames to scene IDs from camera intrinsic calibration files.

    This function scans the specified directory and its subdirectories to find camera 
    intrinsic calibration files in JSON format. It extracts the scene ID from each file 
    and creates a dictionary mapping the filename to its corresponding scene ID.

    Parameters
    ----------
    cam_intri_dir : str
        The path to the directory containing the camera intrinsic calibration JSON files.

    Returns
    -------
    filename2sceneid_mapping : dict
        A dictionary where the keys are the filenames (ending in '.json') and the values 
        are the corresponding scene IDs extracted from the 'cameraID' field in each JSON file.
    """
    filename2sceneid_mapping = dict()
    for root, dirs, files in os.walk(cam_intri_dir):
        for file in files:
            if file.endswith('.json'):
                scene_id = json.load(open(os.path.join(root, file), 'r'))['cameraID']
                filename2sceneid_mapping[file] = scene_id
    return filename2sceneid_mapping
                
    



def get_label_location_dimensions_rotation(json_file: str, return_type=False) -> np.ndarray:
    """
    Extract object location, dimensions, and rotation from a JSON label file.

    This function reads a label file in JSON format, extracts the 3D location, dimensions, 
    and rotation for each object, and returns them in a structured numpy array format.

    Parameters
    ----------
    json_file : str
        The path to the label file containing object data in JSON format.

    return_type : bool, optional
        If True, returns an additional numpy array with object types. Default is False.


    Returns
    -------
    gt_location_dimensions_rotation : np.ndarray, shape (N, 7)
        A numpy array where each row corresponds to an object, and the columns represent:
        - (x, y, z) : 3D location in the LiDAR coordinate system.
        - (height, width, length) : Dimensions of the object.
        - rotation : The rotation angle around the vertical axis.
        
    gt_type_trackid : np.ndarray, shape (N, 1), optional
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
        # add the all values of the dict of the locaton, dimensions, rotation, type and trackid to a list
        location = list(location.values())
        location = [float(i) for i in location]
        dimensions = list(dimensions.values())
        dimensions = [float(i) for i in dimensions]
        rotation = [rotation]
        rotation = [float(i) for i in rotation]
        type = [type]
        gt_location_dimensions_rotation.append(location + dimensions + rotation)
        gt_type_trackid.append(type)
    gt_location_dimensions_rotation = np.array(gt_location_dimensions_rotation)
    gt_type_trackid = np.array(gt_type_trackid)
    if return_type:
        return gt_location_dimensions_rotation, gt_type_trackid
    return gt_location_dimensions_rotation


def get_transformation_matrix(file_path: str):
    """
    Retrieve the camera intrinsic matrix and the transformation matrices between LiDAR and camera coordinate systems.

    This function extracts the camera's intrinsic matrix and computes the transformation matrices 
    for converting coordinates between the LiDAR and camera coordinate systems. The information is 
    retrieved from corresponding calibration JSON files based on the provided image file path.

    Parameters
    ----------
    file_path : str
        The path to the image file. This path is used to locate the associated calibration files.

    Returns
    -------
    tuple
        (cam_intri, lid2cam_extri, cam2lid_extri)
        - cam_intri: (3, 3) numpy array, the camera's intrinsic matrix.
        - lid2cam_extri: (4, 4) numpy array, the transformation matrix from the LiDAR to the camera.
        - cam2lid_extri: (4, 4) numpy array, the transformation matrix from the camera to the LiDAR.
    """
    
    # camera intrinsic matrix
    file_path_split = file_path.split('/single-infrastructure-side-image/')
    json_file = os.path.join(file_path_split[0], 'single-infrastructure-side/calib/camera_intrinsic', file_path_split[1].replace('.jpg', '.json'))
    intrinx_json = json.load(open(json_file, 'r'))
    intrinx = intrinx_json['cam_K']
    intrinx = np.array(intrinx)
    cam_intri = intrinx.reshape(3, 3)

    # lidar to camera transformation matrix
    json_file = os.path.join(file_path_split[0], 'single-infrastructure-side/calib/virtuallidar_to_camera', file_path_split[1].replace('.jpg', '.json'))
    lid2cam_calib = json.load(open(json_file, 'r'))
    lid2cam_calib_R = lid2cam_calib['rotation']
    lid2cam_calib_R = np.array(lid2cam_calib_R)
    lid2cam_calib_T = lid2cam_calib['translation']
    lid2cam_calib_T = np.array(lid2cam_calib_T)
    lid2cam_extri = np.hstack((lid2cam_calib_R, lid2cam_calib_T.reshape(3,1)))
    row = np.array([0., 0., 0., 1.])
    lid2cam_extri = np.vstack((lid2cam_extri, row))
    # camera to lidar transformation matrix
    cam2lid_extri = np.linalg.inv(lid2cam_extri)
 
    return cam_intri, lid2cam_extri, cam2lid_extri