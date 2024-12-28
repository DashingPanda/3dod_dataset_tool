# === author: chenfenglian ===
# This file contains functions to transform points between different coordinate systems

import numpy as np
import math

np.set_printoptions(suppress=True)


# =========================== Coordinate Systems Transformations ===========================

def corA2corB(points_A: np.ndarray, T_AB: np.ndarray) -> np.ndarray:
    """
    Transform points from the coordinate system A to the coordinate system B.

    This function applies a rigid transformation to convert 3D points from the
    coordinate system A into the coordinate system B using the extrinsic matrix.

    Parameters
    ----------    
    points_A : np.ndarray, (N, 4), N is the number of points
        Points in the LiDAR coordinate system in homogeneous coordinates, represented as
        (X_A, Y_A, Z_A, 1)^T, where N is the number of points.
    T_AB : np.ndarray, (4, 4)
        The extrinsic matrix, which describes the rigid transformation from the LiDAR
        coordinate system to the camera coordinate system. It is typically defined as:

        [ R  t ] \n
        [ 0  1 ]


    Returns
    -------
    points_B : np.ndarray, (N, 4), N is the number of points
        Points in camera coordinate system, homogeneous coordinates, (x, y, z, 1)

    Notes
    -----
    The transformation from the coordinate system A (X_A, Y_A, Z_A) to the coordinate
    system B (X_B, Y_B, Z_B) is done using the following equation:

       [ X_B ]     [ R  t ] [ X_A ] \n
       [ Y_B ]  =  [ 0  1 ] [ Y_A ] \n
       [ Z_B ]              [ Z_A ] \n
       [ 1   ]              [ 1   ] \n

    Where:
    - (X_A, Y_A, Z_A) are the coordinates in the coordinate system A.
    - (X_B, Y_B, Z_B) are the transformed coordinates in the coordinate system B.
    - R is the 3x3 rotation matrix, and t is the 3x1 translation vector from the extrinsic matrix.
    """
    # transform points from A to B
    points_B = np.dot(T_AB, points_A.T).T # (N, 4)
    return points_B


def cam2lid(points_cam: np.ndarray, Transformation_cam2lid: np.ndarray) -> np.ndarray:
    """
    Transform points from the camera coordinate system to the LiDAR coordinate system.

    This function applies a rigid transformation to convert 3D points from the camera
    coordinate system into the LiDAR coordinate system using the Transformation matrix.

    Parameters
    ----------    
    points_cam : np.ndarray, (N, 4), N is the number of points
        Points in the camera coordinate system in homogeneous coordinates, represented as
        (X_c, Y_c, Z_c, 1)^T, where N is the number of points.
    Transformation_cam2lid : np.ndarray, (4, 4)
        The Transformation matrix, which describes the rigid transformation from the camera
        coordinate system to the LiDAR coordinate system. It is typically defined as:

        [ R  t ] \n
        [ 0  1 ]


    Returns
    -------
    points_lid : np.ndarray, (N, 4), N is the number of points
        Points in LiDAR coordinate system, homogeneous coordinates, (x, y, z, 1)

    Notes
    -----
    The transformation from the camera coordinate system (X_c, Y_c, Z_c) to the LiDAR
    coordinate system (X_l, Y_l, Z_l) is done using the following equation:

       [ X_l ]     [ R  t ] [ X_c ] \n
       [ Y_l ]  =  [ 0  1 ] [ Y_c ] \n
       [ Z_l ]              [ Z_c ] \n
       [ 1   ]              [ 1   ] \n

    Where:
    - (X_c, Y_c, Z_c) are the coordinates in the camera coordinate system.
    - (X_l, Y_l, Z_l) are the transformed coordinates in the LiDAR coordinate system.
    - R is the 3x3 rotation matrix, and t is the 3x1 translation vector from the transformation matrix.
    """
    # # transform points from camera to LiDAR
    return corA2corB(points_cam, Transformation_cam2lid)


def cam2wor(points_cam: np.ndarray, Transformation_cam2wor: np.ndarray) -> np.ndarray:
    """
    Transform points from the camera coordinate system to the world coordinate system.

    This function applies a rigid transformation to convert 3D points from the camera
    coordinate system into the world coordinate system using the transformation matrix.

    Parameters
    ----------    
    points_cam : np.ndarray, (N, 4), N is the number of points
        Points in the camera coordinate system in homogeneous coordinates, represented as
        (X_c, Y_c, Z_c, 1)^T, where N is the number of points.
    Transformation_cam2wor : np.ndarray, (4, 4)
        The transformation matrix, which describes the rigid transformation from the camera
        coordinate system to the world coordinate system. It is typically defined as:

        [ R  t ] \n
        [ 0  1 ]


    Returns
    -------
    points_wor : np.ndarray, (N, 4), N is the number of points
        Points in world coordinate system, homogeneous coordinates, (x, y, z, 1)

    Notes
    -----
    The transformation from the camera coordinate system (X_c, Y_c, Z_c) to the world
    coordinate system (X_w, Y_w, Z_w) is done using the following equation:

       [ X_w ]     [ R  t ] [ X_c ] \n
       [ Y_w ]  =  [ 0  1 ] [ Y_c ] \n
       [ Z_w ]              [ Z_c ] \n
       [ 1   ]              [ 1   ] \n

    Where:
    - (X_c, Y_c, Z_c) are the coordinates in the camera coordinate system.
    - (X_w, Y_w, Z_w) are the transformed coordinates in the world coordinate system.
    - R is the 3x3 rotation matrix, and t is the 3x1 translation vector from the transformation matrix.
    """
    # # transform points from camera to world
    return corA2corB(points_cam, Transformation_cam2wor)


def lid2cam(points_lid: np.ndarray, Transformation_lid2cam: np.ndarray) -> np.ndarray:
    """
    Transform points from the LiDAR coordinate system to the camera coordinate system.

    This function applies a rigid transformation to convert 3D points from the LiDAR
    coordinate system into the camera coordinate system using the transformation matrix.

    Parameters
    ----------    
    points_lid : np.ndarray, (N, 4), N is the number of points
        Points in the LiDAR coordinate system in homogeneous coordinates, represented as
        (X_l, Y_l, Z_l, 1)^T, where N is the number of points.
    Transformation_lid2cam : np.ndarray, (4, 4)
        The transformation matrix, which describes the rigid transformation from the LiDAR
        coordinate system to the camera coordinate system. It is typically defined as:

        [ R  t ] \n
        [ 0  1 ]


    Returns
    -------
    points_cam : np.ndarray, (N, 4), N is the number of points
        Points in camera coordinate system, homogeneous coordinates, (x, y, z, 1)

    Notes
    -----
    The transformation from the LiDAR coordinate system (X_l, Y_l, Z_l) to the camera
    coordinate system (X_c, Y_c, Z_c) is done using the following equation:

       [ X_c ]     [ R  t ] [ X_l ] \n
       [ Y_c ]  =  [ 0  1 ] [ Y_l ] \n
       [ Z_c ]              [ Z_l ] \n
       [ 1   ]              [ 1   ] \n

    Where:
    - (X_l, Y_l, Z_l) are the coordinates in the LiDAR coordinate system.
    - (X_c, Y_c, Z_c) are the transformed coordinates in the camera coordinate system.
    - R is the 3x3 rotation matrix, and t is the 3x1 translation vector from the transformation matrix.
    """
    # transform points from LiDAR to camera
    return corA2corB(points_lid, Transformation_lid2cam)


def lid2wor(points_lid: np.ndarray, Transformation_lid2wor: np.ndarray) -> np.ndarray:
    """
    Transform points from the LiDAR coordinate system to the world coordinate system.

    This function applies a rigid transformation to convert 3D points from the LiDAR
    coordinate system into the world coordinate system using the transformation matrix.

    Parameters
    ----------    
    points_lid : np.ndarray, (N, 4), N is the number of points
        Points in the LiDAR coordinate system in homogeneous coordinates, represented as
        (X_w, Y_w, Z_w, 1)^T, where N is the number of points.
    Transformation_lid2wor : np.ndarray, (4, 4)
        The Transformation matrix, which describes the rigid transformation from the LiDAR
        coordinate system to the world coordinate system. It is typically defined as:

        [ R  t ] \n
        [ 0  1 ]


    Returns
    -------
    points_wor : np.ndarray, (N, 4), N is the number of points
        Points in world coordinate system, homogeneous coordinates, (x, y, z, 1)

    Notes
    -----
    The transformation from the LiDAR coordinate system (X_l, Y_l, Z_l) to the world
    coordinate system (X_w, Y_w, Z_w) is done using the following equation:

       [ X_w ]     [ R  t ] [ X_l ] \n
       [ Y_w ]  =  [ 0  1 ] [ Y_l ] \n
       [ Z_w ]              [ Z_l ] \n
       [ 1   ]              [ 1   ] \n

    Where:
    - (X_l, Y_l, Z_l) are the coordinates in the LiDAR coordinate system.
    - (X_w, Y_w, Z_w) are the transformed coordinates in the world coordinate system.
    - R is the 3x3 rotation matrix, and t is the 3x1 translation vector from the transformation matrix.
    """
    # transform points from LiDAR to world
    return corA2corB(points_lid, Transformation_lid2wor)


def wor2cam(points_wor: np.ndarray, Transformation_wor2cam: np.ndarray) -> np.ndarray:
    """
    Transform points from the worlde coordinate system to the camera coordinate system.

    This function applies a rigid transformation to convert 3D points from the world
    coordinate system into the camera coordinate system using the transformation matrix.

    Parameters
    ----------    
    points_wor : np.ndarray, (N, 4), N is the number of points
        Points in the world coordinate system in homogeneous coordinates, represented as
        (X_w, Y_w, Z_w, 1)^T, where N is the number of points.
    Transformation_wor2cam : np.ndarray, (4, 4)
        The Transformation matrix, which describes the rigid transformation from the world
        coordinate system to the camera coordinate system. It is typically defined as:

        [ R  t ] \n
        [ 0  1 ]


    Returns
    -------
    points_cam : np.ndarray, (N, 4), N is the number of points
        Points in camera coordinate system, homogeneous coordinates, (x, y, z, 1)

    Notes
    -----
    The transformation from the world coordinate system (X_w, Y_w, Z_w) to the camera
    coordinate system (X_c, Y_c, Z_c) is done using the following equation:

       [ X_c ]     [ R  t ] [ X_w ] \n
       [ Y_c ]  =  [ 0  1 ] [ Y_w ] \n
       [ Z_c ]              [ Z_w ] \n
       [ 1   ]              [ 1   ] \n

    Where:
    - (X_w, Y_w, Z_w) are the coordinates in the world coordinate system.
    - (X_c, Y_c, Z_c) are the transformed coordinates in the camera coordinate system.
    - R is the 3x3 rotation matrix, and t is the 3x1 translation vector from the transformation matrix.
    """
    # transform points from world to camera
    return corA2corB(points_wor, Transformation_wor2cam)



# =========================== to image coordinate system ===========================

def cam2img(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Transform points from the camera coordinate system to the image coordinate system.

    This function projects 3D points from the camera coordinate system onto a 2D image plane
    using the camera's intrinsic matrix.

    Parameters
    ----------
    points_cam : np.ndarray, (N, 4), N is the number of points
         Points in the camera coordinate system in homogeneous coordinates, represented as
        (X_c, Y_c, Z_c, 1)^T, where N is the number of points.
    intrinsics : np.ndarray, (3, 3)
        The camera's intrinsic matrix, defined as:
        
        [ fx   0  cx ] \n
        [  0  fy  cy ] \n
        [  0   0   1 ] \n

        where fx and fy are the focal lengths in the x and y directions, and \
        cx and cy are the coordinates of the principal point in the image.


    Returns
    -------
    points_img : np.ndarray, (N, 2), N is the number of points
        Points in image coordinate system, (u, v)

    Notes
    -----
    The projection of a 3D point (X_c, Y_c, Z_c) from the camera coordinate system to the
    2D image plane (u, v) is done in two steps:

    1. Compute the image plane coordinates in homogeneous form:
    
       [ u ]     [ fx   0  cx  0 ] [ X_c ] \n
       [ v ]  =  [  0  fy  cy  0 ] [ Y_c ] \n
       [ w ]     [  0   0   1  0 ] [ Z_c ] \n
                                   [  1  ] \n

    2. Normalize the homogeneous coordinates to get the pixel coordinates:
    
       u = u / w \n
       v = v / w

    This gives the final image coordinates (u, v).
    """

    # convert to homogeneous coordinates
    intrinsics = np.hstack((intrinsics, np.array([[0], [0], [0]]))) # (3, 4)
    # transform points from camera to image
    points_img = np.dot(intrinsics, points_cam.T).T # (N, 3)
    # normalize points to image size
    points_img[:, 0] /= points_img[:, 2]
    points_img[:, 1] /= points_img[:, 2]
    # remove homogeneous coordinates
    points_img = points_img[:, :2]
    return points_img



# =========================== angle transformation ===========================

def cam2lid_rotation(rotation_cam: np.ndarray, Transformation_cam2lid: np.ndarray) -> np.ndarray:
    """
    Transform rotation degrees from camera coordinate system along y-axis to LiDAR coordinate system along z-axis.

    Parameters
    ----------
    rotation_cam : np.ndarray, (N, 1), N is the number of objects
        Rotation degrees in camera coordinate system along y-axis
    Transformation_cam2lid : np.ndarray, (4, 4)
        transformation matrix of the camera to lidar

    Returns
    -------
    rotation_lid : np.ndarray, (N, 1), N is the number of objects
        Rotation degrees in LiDAR coordinate system along z-axis
    """
    rotation_lid_list = []
    for ry3d in rotation_cam:
        theta = np.matrix([math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
        theta0 = Transformation_cam2lid[:3, :3] * theta  #first column
        yaw_world_res = math.atan2(theta0[1], theta0[0])
        rotation_lid_list.append(yaw_world_res)
    rotation_lid = np.array(rotation_lid_list).reshape(-1, 1)
    return rotation_lid


def cam2wor_rotation(rotation_cam: np.ndarray, Transformation_cam2wor: np.ndarray) -> np.ndarray:
    """
    Transform rotation degrees from camera coordinate system along y-axis to world coordinate system along z-axis.

    Parameters
    ----------
    rotation_cam : np.ndarray, (N, 1), N is the number of objects
        Rotation degrees in camera coordinate system along y-axis
    Transformation_cam2wor : np.ndarray, (4, 4)
        transformation matrix of the camera to world

    Returns
    -------
    rotation_wor : np.ndarray, (N, 1), N is the number of objects
        Rotation degrees in world coordinate system along z-axis
    """

    rotation_wor_list = []
    for ry3d in rotation_cam:
        theta = np.matrix([math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
        theta0 = Transformation_cam2wor[:3, :3] * theta  #first column
        yaw_world_res = math.atan2(theta0[1], theta0[0])
        rotation_wor_list.append(yaw_world_res)
    rotation_wor = np.array(rotation_wor_list).reshape(-1, 1)
    return rotation_wor