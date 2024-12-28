import sys
sys.path.append('/gemini/data-1/Projects/rcooper-dataset-tools')

from lib.utils import common_utils

import numpy as np
import torch

# Disable scientific notation
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)




def boxes_to_corners_3d(boxes3d: np.ndarray, order: str, isCenter: bool = True) -> np.ndarray:
    """
    Convert 3D bounding boxes to their corner points.

    This function takes 3D bounding boxes defined by their center, dimensions, and
    heading (rotation around the z-axis) and computes the coordinates of the 8 corners
    of each bounding box. The order of dimensions can be specified to match different
    conventions.

    Parameters
    ----------
    boxes3d : np.ndarray, shape (N, 7)
        An array of 3D bounding boxes where each box is defined by:
        - x, y, z : Coordinates of the box center in the LiDA coordinates system.
        - dx, dy, dz : Dimensions of the box along the x, y, and z axes.
        - heading : Rotation angle around the z-axis in radians.
        
    order : str
        Specifies the order of dimensions in the input array. Can be:
        - 'whl': width, height, length
        - 'lwh': length, width, height
        - 'hwl': height, width, length

    Returns
    -------
    corners3d : np.ndarray, shape (N, 8, 3)
        The coordinates of the 8 corners for each bounding box. Each box is represented
        by 8 corner points in 3D space, corresponding to the vertices of the box.

    Notes
    -----
    The corners are returned in the following order for each box:
    
        5 -------- 4
       /|         /|
      6 -------- 7 .
      | |        | |
      . 1 -------- 0
      |/         |/
      2 -------- 3

    - The function adjusts the dimensions according to the specified `order`.
    - It computes the corners by first creating a template based on the box dimensions
      and then applying the rotation and translation to align with the box's position
      and orientation.
    
    Example
    -------
    >>> boxes3d = np.array([[0, 0, 0, 2, 1, 1, np.pi/4]])
    >>> corners = boxes_to_corners_3d(boxes3d, 'lwh')
    >>> print(corners)
    array([[[ 0.707, -0.707, -0.5],
            [ 1.707,  0.293, -0.5],
            [-0.293,  1.707, -0.5],
            [-1.293,  0.707, -0.5],
            [ 0.707, -0.707,  0.5],
            [ 1.707,  0.293,  0.5],
            [-0.293,  1.707,  0.5],
            [-1.293,  0.707,  0.5]]])
    """
    # ^ z
    # |
    # |
    # | . x
    # |/
    # +-------> y
    
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    if order == 'hwl':
        boxes3d[:, 3:6] = boxes3d[:, [5, 4, 3]] # wlh to lwh

    if order == 'whl':
        boxes3d[:, 3:6] = boxes3d[:, [5, 3, 4]] # whl to lwh

    if isCenter:
        template = boxes3d.new_tensor((
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
            [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
        )) / 2
    
    else:#bottom center, ground: z axis is up
        template = boxes3d.new_tensor((
            [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0],
            [1, -1, 2], [1, 1, 2], [-1, 1, 2], [-1, -1, 2],
        )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3),
                                                   boxes3d[:, 6]).view(-1, 8, 3)

    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d