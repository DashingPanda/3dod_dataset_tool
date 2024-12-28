# author: ChenFenglian

import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from functools import wraps
import time

from lib.utils import common_utils, transformation
from lib.data_utils.post_processor import box_utils
from lib.data_utils.data import Data
from lib.data_utils.dataset.dg3d import labelloader as dg3d_labelloader



def args_parser():
    parser = argparse.ArgumentParser(description='3Dcudedepth')
    parser.add_argument('--files_txt', type=str, help='the path to a .txt file the contain the path of the label files')
    parser.add_argument('--dataset_dir', type=str, help='the path to the directory containing the dataset')
    args = parser.parse_args()
    return args



def timer(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        begin_time = time.perf_counter()
        result = func(*args, **kwargs)
        start_time = time.perf_counter()
        print('=' * 50)
        print('func:%r took: %2.4f sec' % (func.__name__, start_time - begin_time))
        return result
    return wrap



# @timer
def get_3dbox_corners_img_cam(objs: list, cam_intri: np.ndarray, cam2world_extri: np.ndarray, world2cam_extri: np.ndarray) -> np.ndarray:
    # get the obj_location_dimensions_rotation in camera coordinate system
    obj_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)  # (N, 7)
    obj_location_cam = obj_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
    obj_location_cam = common_utils.covert2homogeneous(obj_location_cam) # (N, 4), homogeneous coordinates
    obj_location_world = transformation.cam2wor(obj_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
    obj_location_world = obj_location_world[:, :3]  # (N, 3), location in world coordinate systems
    obj_rotation_cam = obj_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
    obj_rotation_world = transformation.cam2wor_rotation(obj_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
    obj_location_dimensions_rotation_world = np.concatenate([obj_location_world, obj_location_dimensions_rotation_cam[:, 3:6], obj_rotation_world], axis=1)  # (N, 7)
    # get the 8 corners of a 3d bounding box in world coordinate system
    obj_corners_world = box_utils.boxes_to_corners_3d(obj_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)
    obj_corners_world = obj_corners_world.reshape(-1, 3)  # (N*8, 3) in world coordinate system
    # convert the obj_corners_world to homogeneous coordinates
    obj_corners_world = common_utils.covert2homogeneous(obj_corners_world)  # (N*8, 4)
    # transform the obj_corners_worlde to camera coordinate system from world coordinate system
    obj_corners_cam = transformation.wor2cam(obj_corners_world, world2cam_extri)  # (N*8, 4)
    # transform the obj_corners_cam to image coordinate system from camera coordinate system
    obj_corners_img = transformation.cam2img(obj_corners_cam, cam_intri)  # (N*8, 2)
    obj_corners_img = obj_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
    obj_corners_cam = obj_corners_cam[:,:3].reshape(-1, 8, 3)  # (N, 8, 3) in camera coordinate system
    return obj_corners_img, obj_corners_cam



# @timer
def mask_3dbox_img(o_corners_img : np.ndarray):
    """
    Draw the silhouette of a 3d bounding box in a image

    Parameters
    ----------
    o_corners_img : np.ndarray, shape (8, 2)
        The 8 corners of a 3d bounding box in pixel coordinate system

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

    Draw the silhouette of a 3d bounding box in a image:
    
        5 -------- 4                    5 -------- 4
       /|         /|                   /           |
      6 -------- 7 .                  6            |
      | |        | |     ------->     |            |   
      . 1 -------- 0                  |            0
      |/         |/                   |           /
      2 -------- 3                    2 -------- 3

    """

    # create a image with zeros of size 1920x1080
    cude_mask_obj = np.zeros((1080, 1920), dtype=np.uint16)
    # convert the o_corners_img to int
    o_corners_img = o_corners_img.astype(np.int32)
    # get the 6 planes of the 3d bounding box
    o_plane1_img = o_corners_img[0:4]  # (4, 2) # points of the first plane (p0, p1, p2, p3)
    o_plane2_img = o_corners_img[4:8]  # (4, 2) # points of the second plane (p4, p5, p6, p7)
    o_plane3_img = np.concatenate((o_corners_img[2:4], o_corners_img[7:8], o_corners_img[6:7]))  # (4, 2) # points of the third plane (p2, p3, p7, p6)
    o_plane4_img = np.concatenate((o_corners_img[1:3], o_corners_img[6:7], o_corners_img[5:6]))  # (4, 2) # points of the fourth plane (p1, p2, p6, p5)
    o_plane5_img = np.concatenate((o_corners_img[0:2], o_corners_img[5:6], o_corners_img[4:5]))  # (4, 2) # points of the fifth plane (p0, p1, p5, p4)
    o_plane6_img = np.concatenate((o_corners_img[0:1], o_corners_img[3:4], o_corners_img[7:8], o_corners_img[4:5]))  # (4, 2) # points of the sixth plane (p0, p3, p7, p4)
    # Draw the closed plane area
    cv2.fillPoly(cude_mask_obj, [o_plane1_img], color=(255, 255, 255))
    cv2.fillPoly(cude_mask_obj, [o_plane2_img], color=(255, 255, 255))
    cv2.fillPoly(cude_mask_obj, [o_plane3_img], color=(255, 255, 255))
    cv2.fillPoly(cude_mask_obj, [o_plane4_img], color=(255, 255, 255))
    cv2.fillPoly(cude_mask_obj, [o_plane5_img], color=(255, 255, 255))
    cv2.fillPoly(cude_mask_obj, [o_plane6_img], color=(255, 255, 255))
    
    return cude_mask_obj


# # @timer
def mask_3dboxplaen_img(p_corners_img):
    """
    Draw the silhouette of a 3d bounding box plane in a image

    Parameters
    ----------
    p_corners_img : np.ndarray, shape (4, 2)
        The 4 corners of a 3d bounding box plane in pixel coordinate system

    """

    # create a image with zeros of size 1920x1080
    plane_mask_obj = np.zeros((1080, 1920), dtype=np.uint16)
    # convert the o_corners_img to int
    p_corners_img = p_corners_img.astype(np.int32)
    # Draw the closed plane area
    cv2.fillPoly(plane_mask_obj, [p_corners_img], color=(255, 255, 255))
    return plane_mask_obj


# # @timer
def calculate_coefficient_plane(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """
    Calculate the coefficients of a plane given 3 points on the plane

    Parameters
    ----------
    p1 : np.ndarray, shape (3,)
        The first point on the plane
    p2 : np.ndarray, shape (3,)
        The second point on the plane
    p3 : np.ndarray, shape (3,)
        The third point on the plane
    """
    # calculate the normal vector of the plane
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    # calculate the coefficients of the plane
    a, b, c = normal
    d = -np.dot(normal, p1)
    return np.array([a, b, c, d])


# # @timer
def calculate_coefficient_6planes(o_corners_cam: np.ndarray):
    """

    Parameters
    ----------
    o_corners_cam : np.ndarray, shape (8, 3)
        The 8 corners of a 3d bounding box in camera coordinate system

    Returns
    -------
    coefficient_6planes : np.ndarray, shape (6, 4)
        The coefficients of the 6 planes of the 3d bounding box
        
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

    The 6 planes of a 3d bounding box:
    - plane1: p0, p1, p2, p3
    - plane2: p4, p5, p6, p7
    - plane3: p2, p3, p6, p7
    - plane4: p1, p2, p5, p6
    - plane5: p0, p1, p4, p5
    - plane6: p0, p3, p4, p7

    """

    def check(p1, p2, p3, p4, coefficient_plane):
        a, b, c, d = coefficient_plane
        print(f'===================== check the coefficient_plane is correct =====================')
        print(f'check the coefficient_plane is correct: {a*p1[0] + b*p1[1] + c*p1[2] + d}')
        print(f'check the coefficient_plane is correct: {a*p2[0] + b*p2[1] + c*p2[2] + d}')
        print(f'check the coefficient_plane is correct: {a*p3[0] + b*p3[1] + c*p3[2] + d}')
        print(f'check the coefficient_plane is correct: {a*p4[0] + b*p4[1] + c*p4[2] + d}')

    coefficient_6planes = np.zeros((6, 4), dtype=np.double)
    coefficient_plane1 = calculate_coefficient_plane(o_corners_cam[0], o_corners_cam[1], o_corners_cam[2])
    coefficient_plane2 = calculate_coefficient_plane(o_corners_cam[4], o_corners_cam[5], o_corners_cam[6])
    coefficient_plane3 = calculate_coefficient_plane(o_corners_cam[2], o_corners_cam[3], o_corners_cam[6])
    coefficient_plane4 = calculate_coefficient_plane(o_corners_cam[1], o_corners_cam[2], o_corners_cam[6])
    coefficient_plane5 = calculate_coefficient_plane(o_corners_cam[0], o_corners_cam[1], o_corners_cam[4])
    coefficient_plane6 = calculate_coefficient_plane(o_corners_cam[0], o_corners_cam[3], o_corners_cam[4])

    # check(o_corners_cam[0], o_corners_cam[1], o_corners_cam[2], o_corners_cam[3], coefficient_plane1)
    # check(o_corners_cam[4], o_corners_cam[5], o_corners_cam[6], o_corners_cam[7], coefficient_plane2)
    # check(o_corners_cam[2], o_corners_cam[3], o_corners_cam[6], o_corners_cam[7], coefficient_plane3)
    # check(o_corners_cam[1], o_corners_cam[2], o_corners_cam[6], o_corners_cam[5], coefficient_plane4)
    # check(o_corners_cam[0], o_corners_cam[1], o_corners_cam[4], o_corners_cam[5], coefficient_plane5)
    # check(o_corners_cam[0], o_corners_cam[3], o_corners_cam[4], o_corners_cam[7], coefficient_plane6)

    coefficient_6planes = np.vstack((coefficient_plane1, coefficient_plane2, coefficient_plane3,
                                     coefficient_plane4, coefficient_plane5, coefficient_plane6))

    # print(f'the type of coefficient_6planes: {type(coefficient_6planes[0,0])}')
    return coefficient_6planes



# @timer
def calculate_the_depth(coefficient_6planes: np.ndarray, u: int, v: int, fx: float, fy: float, cx: float, cy: float) -> float:
    """
    Calculate the depth of a pixel in 3d bounding box planes

    - αix + βiy + γiz + di = 0, for i = 1, ..., 6       
    - fx(x/z) + cx = u     
    - fy(y/z) + cy = v

    -> z = −di/(αi(u-cx)/fx + βi(v-cy)/fy + γi) \n

    reference: MonoUNI: A Unified Vehicle and Infrastructure-side 
    Monocular 3D Object Detection Network with Sufficient Depth Clues

    Parameters
    ----------
    coefficient_6planes: np.ndarray, shape (6, 4)
        The coefficients of the 6 planes of the 3d bounding box
    u : int
        The x-coordinate of the pixel in the image
    v : int
        The y-coordinate of the pixel in the image
    fx : float
        The focal length in the x-direction
    fy : float
        The focal length in the y-direction
    cx : float
        The x-coordinate of the principal point
    cy : float
        The y-coordinate of the principal point

    """
    z = sys.float_info.max # set the initial depth to the maximum value of float
    for coefficient_plane in coefficient_6planes:
        a, b, c, d = coefficient_plane
        z_tmp = -d / (a * (u - cx) / fx + b * (v - cy) / fy + c)
        if z_tmp < z:
            z = z_tmp
    return z



# # @timer
def calculate_the_depth_plane(coefficient_plane: np.ndarray, u: int, v: int, fx: float, fy: float, cx: float, cy: float) -> float:
    """
    Calculate the depth of a pixel in a 3d bounding box plane

    - αx + βy + γz + d = 0
    - fx(x/z) + cx = u     
    - fy(y/z) + cy = v

    -> z = −d/(α(u-cx)/fx + β(v-cy)/fy + γ)

    reference: MonoUNI: A Unified Vehicle and Infrastructure-side 
    Monocular 3D Object Detection Network with Sufficient Depth Clues

    Parameters
    ----------
    coefficient_6planes: np.ndarray, shape (4,)
        The coefficients of the 6 planes of the 3d bounding box
    u : int
        The x-coordinate of the pixel in the image
    v : int
        The y-coordinate of the pixel in the image
    fx : float
        The focal length in the x-direction
    fy : float
        The focal length in the y-direction
    cx : float
        The x-coordinate of the principal point
    cy : float
        The y-coordinate of the principal point

    """
    a, b, c, d = coefficient_plane
    z = -d / (a * (u - cx) / fx + b * (v - cy) / fy + c)
    # print(f'z: {z}')
    # exit()
    return z



# # @timer
def generate_cude_depth_per_plane(p_corners_img: np.ndarray, coefficient_plane: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """
    Generate the depth of a pixel in a 3d bounding box plane

    Parameters
    ----------
    p_corners_img : np.ndarray, shape (4, 2)
        The 4 corners of a 3d bounding box plane in pixel coordinate system
    coefficient_plane : np.ndarray, shape (4,)
        The coefficients of the plane
    fx : float
        The focal length in the x-direction
    fy : float
        The focal length in the y-direction
    cx : float
        The x-coordinate of the principal point
    cy : float
        The y-coordinate of the principal point
    
    Returns
    -------
    plane_depth_img : np.ndarray, shape (1080, 1920)
        The depth of a 3d bounding box plane in a image
    """

    plane_mask_obj = mask_3dboxplaen_img(p_corners_img) # Draw the silhouette of a 3d bounding box plane in a image
    plane_depth_img = np.zeros((1080, 1920), dtype=np.uint16)
    plane_mask_obj_r = plane_mask_obj[:,:]
    h, w = plane_mask_obj_r.shape
    
    # get the minimum and maximum value p_corners_img[:, 0] and p_corners_img[:, 1]
    min_x = int(np.min(p_corners_img[:, 0])) - 1 if int(np.min(p_corners_img[:, 0])) - 1 > 0 else 0
    min_y = int(np.min(p_corners_img[:, 1])) - 1 if int(np.min(p_corners_img[:, 1])) - 1 > 0 else 0
    max_x = int(np.max(p_corners_img[:, 0])) + 1 if int(np.max(p_corners_img[:, 0])) + 1 < w else w
    max_y = int(np.max(p_corners_img[:, 1])) + 1 if int(np.max(p_corners_img[:, 1])) + 1 < h else h

    # loop through each pixel in cude_mask_obj_r
    # for each pixel, check if it is inside the 3d bounding box plane silhouette
    for v in range(min_y, max_y):
        for u in range(min_x, max_x):
            pixel = plane_mask_obj_r[v, u]
            if pixel == 255:
                # print(f'u: {u}, v: {v}')
                plane_depth = calculate_the_depth_plane(coefficient_plane, u, v, fx, fy, cx, cy) * 256
                plane_depth_img[v, u] = plane_depth
    return plane_depth_img



# # @timer
def split_3dbox_planes(o_corners_img: np.ndarray):
    """
    Split the 3d bounding box into 6 planes

    Parameters
    ----------
    o_corners_img : np.ndarray, shape (8, 2)
        The 8 corners of a 3d bounding box in pixel coordinate system

    Returns
    -------
    planes : list
        The 6 planes of the 3d bounding box

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

    The 6 planes of a 3d bounding box:
    - plane1: p0, p1, p2, p3
    - plane2: p4, p5, p6, p7
    - plane3: p2, p3, p6, p7
    - plane4: p1, p2, p5, p6
    - plane5: p0, p1, p4, p5
    - plane6: p0, p3, p4, p7

    """
    planes = list()
    # get the 6 planes of the 3d bounding box
    o_plane1_img = o_corners_img[0:4]  # (4, 2) # points of the first plane (p0, p1, p2, p3)
    o_plane2_img = o_corners_img[4:8]  # (4, 2) # points of the second plane (p4, p5, p6, p7)
    o_plane3_img = np.concatenate((o_corners_img[2:4], o_corners_img[7:8], o_corners_img[6:7]))  # (4, 2) # points of the third plane (p2, p3, p7, p6)
    o_plane4_img = np.concatenate((o_corners_img[1:3], o_corners_img[6:7], o_corners_img[5:6]))  # (4, 2) # points of the fourth plane (p1, p2, p6, p5)
    o_plane5_img = np.concatenate((o_corners_img[0:2], o_corners_img[5:6], o_corners_img[4:5]))  # (4, 2) # points of the fifth plane (p0, p1, p5, p4)
    o_plane6_img = np.concatenate((o_corners_img[0:1], o_corners_img[3:4], o_corners_img[7:8], o_corners_img[4:5]))  # (4, 2) # points of the sixth plane (p0, p3, p7, p4)
    # put the 6 planes into a list
    planes.append(o_plane1_img)
    planes.append(o_plane2_img)
    planes.append(o_plane3_img)
    planes.append(o_plane4_img)
    planes.append(o_plane5_img)
    planes.append(o_plane6_img)
    return planes


def combine_into_cude_depth_v1(plane_depth_img: np.ndarray, cude_depth_img: np.ndarray):
    """
    Assign the item values in plane_depth_img to the cude_depth_img along pixel.
    If the pixel value in cude_depth_img is not zero, then keep the minimum value of the two.
    """
    h, w, c = plane_depth_img.shape
    for v in range(h):
        for u in range(w):
            pixel = plane_depth_img[v, u, 0]
            if pixel != 0:
                if cude_depth_img[v, u, 0] == 0:
                    cude_depth_img[v, u, :] = pixel
                else:
                    cude_depth_img[v, u, :] = np.minimum(cude_depth_img[v, u, :], pixel)


# @timer
def combine_into_cude_depth_per_image(cude_depth_objs: np.ndarray, cude_depth_img: np.ndarray):
    """
    Assign the item values in plane_depth_img to the cude_depth_img along pixel.
    If the pixel value in cude_depth_img is not zero, then keep the minimum value of the two.

    parameters
    ----------
    cude_depth_objs : np.ndarray, shape (N, 1080, 1920), N is the number of objects
        The depths of 6 planes of each 3d bounding box in a image
    """
    h, w = cude_depth_objs[0].shape
    for v in range(h):
        for u in range(w):
            # the minimum value of the pixel in plane_depth_imgs and not less than 0
            pixels = cude_depth_objs[:, v, u]
            pixel = np.min(pixels[pixels > 0], axis=0) if len(pixels[pixels > 0]) > 0 else 0
            cude_depth_img[v, u] = pixel


# @timer
def combine_into_cude_depth_per_object(obj: Data, plane_depth_imgs: np.ndarray, cude_depth_img: np.ndarray):
    """
    Assign the item values in plane_depth_img to the cude_depth_img along pixel.
    If the pixel value in cude_depth_img is not zero, then keep the minimum value of the two.

    parameters
    ----------
    obj : Data (from lib.data_utils.data import Data)
        An instance of the Data class
    
    plane_depth_imgs : np.ndarray, shape (N, 1080, 1920)
        The depths of 6 planes of a 3d bounding box

    cude_depth_img : np.ndarray, shape (1080, 1920)
        The depth of a 3d bounding box
    """
    # h, w, c = plane_depth_img.shape
    # for v in range(h):
    #     for u in range(w):
    #         pixel = plane_depth_img[v, u, 0]
    #         if pixel != 0:
    #             if cude_depth_img[v, u, 0] == 0:
    #                 cude_depth_img[v, u, :] = pixel
    #             else:
    #                 cude_depth_img[v, u, :] = np.minimum(cude_depth_img[v, u, :], pixel)
    h, w = plane_depth_imgs[0].shape
    x1, y1, x2, y2 = obj.x1, obj.y1, obj.x2, obj.y2
    for v in range(y1, y2+1):
        for u in range(x1, x2+1):
            # the minimum value of the pixel in plane_depth_imgs and not less than 0
            pixels = plane_depth_imgs[:, v, u]
            pixel = np.min(pixels[pixels > 0], axis=0) if len(pixels[pixels > 0]) > 0 else 0
            cude_depth_img[v, u] = pixel


# @timer
def generate_cude_depth_per_object(obj: Data, o_corners_img: np.ndarray, o_corners_cam: np.ndarray, cam_intri: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    obj : Data (from lib.data_utils.data import Data)
        An instance of the Data class

    o_corners_img : np.ndarray, shape (8, 2)
        The 8 corners of a 3d bounding box in pixel coordinate system
    
    o_corners_cam : np.ndarray, shape (8, 3)
        The 8 corners of a 3d bounding box in camera coordinate system

    cam_intri : np.ndarray, shape (3, 3)
        The camera's intrinsic matrix

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

    The 6 planes of a 3d bounding box:
    - plane1: p0, p1, p2, p3
    - plane2: p4, p5, p6, p7
    - plane3: p2, p3, p6, p7
    - plane4: p1, p2, p5, p6
    - plane5: p0, p1, p4, p5
    - plane6: p0, p3, p4, p7

    """
    fx, _, cx, _, fy, cy, _, _, _  = cam_intri.reshape(1, 9)[0]

    coefficient_6planes = calculate_coefficient_6planes(o_corners_cam) # (6, 4)
    corners_planes = split_3dbox_planes(o_corners_img) # get the 6 plane corners of the 3d bounding box

    # generate the cude depth for each plane and combine them into a cude depth image for a object
    cude_depth_img = np.zeros((1080, 1920), dtype=np.uint16) # create a image with zeros of size 1920x1080x3
    plane_depth_imgs = np.zeros((1, 1080, 1920), dtype=np.uint16)
    for (coefficient_plane, corners_plane) in zip(coefficient_6planes, corners_planes):
        plane_depth_img = generate_cude_depth_per_plane(corners_plane, coefficient_plane, fx, fy, cx, cy)
        # combine_into_cude_depth_v1(plane_depth_img, cude_depth_img)
        plane_depth_img = plane_depth_img.reshape(1, 1080, 1920)
        plane_depth_imgs = np.vstack((plane_depth_imgs, plane_depth_img))
    plane_depth_imgs = plane_depth_imgs[1:, ...]
    combine_into_cude_depth_per_object(obj, plane_depth_imgs, cude_depth_img)

    # cude_mask_obj = mask_3dbox_img(o_corners_img) # Draw the silhouette of a 3d bounding box in a image
    # cude_mask_obj_r = cude_mask_obj[:,:,0]
    # h, w = cude_mask_obj_r.shape
    
    # # loop through each pixel in cude_mask_obj_r
    # # for each pixel, check if it is inside the 3d bounding box silhouette
    # for v in range(h):
    #     for u in range(w):
    #         pixel = cude_mask_obj_r[v, u]
    #         if pixel == 255:
    #             print(f'i: {u}, j: {v}')
    #             calculate_the_depth(coefficient_6planes, u, v, fx, fy, cx, cy)

    # cude_depth_obj = cude_mask_obj

    return cude_depth_img


# @timer
def generate_cude_depth_per_image(objs: list, filename_woe: str, dataset_dir: str):
    """

    Parameters
    ----------
    objs : list
        List of objects in the image, each object is an instance of the Data class
    filename_woe : str
        Filename without extension, e.g. '1632_fa2sd4a11North_420_1612431546_1612432197_1_obstacle'
    dataset_dir : str
        The path to the directory containing the dataset
    """

    # get the intrinsics matrix of the camera
    cam_intri, cam2world_extri, world2cam_extri = dg3d_labelloader.get_transformation_matrix(filename_woe, dataset_dir)
    
    # get the 8 corners of all the 3d bounding box in image and camera coordinate system
    obj_corners_img, obj_corners_cam = get_3dbox_corners_img_cam(objs, cam_intri, cam2world_extri, world2cam_extri)  # (N, 8, 2), (N, 8, 3)

    # create a image with zeros of size 1920x1080
    cude_depth_img = np.zeros((1080, 1920), dtype=np.uint16)
    cude_depth_objs = np.zeros((1, 1080, 1920), dtype=np.uint16)
    for i, (obj, o_corners_img, o_corners_cam) in enumerate(zip(objs, obj_corners_img, obj_corners_cam)):
        
        # generate the cude depth for each object
        cude_depth_obj = generate_cude_depth_per_object(obj, o_corners_img, o_corners_cam, cam_intri)
        cude_depth_obj = cude_depth_obj.reshape(1, 1080, 1920)
        cude_depth_objs = np.vstack((cude_depth_objs, cude_depth_obj))
    cude_depth_objs = cude_depth_objs[1:, ...]
    combine_into_cude_depth_per_image(cude_depth_objs, cude_depth_img)

    return cude_depth_img

    
# @timer
def generate_cude_depth(files_txt, dataset_dir):
    """
    """
    gt_files = list()
    with open(files_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            gt_files.append(line.strip())
    for gt_file in tqdm(gt_files, desc='Processing', colour='green'):
        # gt_file = '/mnt/data_cfl/Projects/Data/Rope3D_data/label_2/1632_fa2sd4a11North_420_1612431546_1612432197_2_obstacle.txt'
        # gt_file = '/mnt/data_cfl/Projects/Data/Rope3D_data/label_2_4cls_for_train/1632_fa2sd4a11North_420_1612431546_1612432197_2_obstacle.txt'
        # gt_file = '/mnt/data_cfl/Projects/Data/Rope3D_data/label_2_4cls_filter_with_roi_for_eval/1632_fa2sd4a11North_420_1612431546_1612432197_2_obstacle.txt'
        filename_woe = os.path.basename(gt_file).split('.')[0] # filename without extension
        gt_result_lines = open(gt_file, 'r').readlines()
        # load label from the predicted result file
        objs = [] # list of valid objects in the predicted result file
        for gt_result_line in gt_result_lines:
            gt_result_line = gt_result_line.strip()
            obj_data = Data()
            dg3d_labelloader.get_label_data(obj_data, gt_result_line)
            if obj_data.X == 0 and obj_data.Y == 0 and obj_data.Z == 0: # No 3d bounding box
                continue
            if obj_data.occlusion > 2: 
                continue
            if obj_data.truncation > 2: # the object is out of image severely
                continue
            objs.append(obj_data)

        # check if there is no object in the image
        if len(objs) == 0:
            # create a image with zeros of size 1920x1080
            cude_depth_img = np.zeros((1080, 1920), dtype=np.uint16)
        else:
            cude_depth_img = generate_cude_depth_per_image(objs, filename_woe, dataset_dir)

        output_dir = '/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/box3d_depth_dense'
        img_file = f'image_2/box3d_depth_dense/2_2/{filename_woe}.png'
        common_utils.save_img(cude_depth_img, output_dir, img_file, dataset='rope3d')

        # exit()

        



def main():
    args = args_parser()
    files_txt = args.files_txt
    dataset_dir = args.dataset_dir

    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'files_txt: {files_txt}')
    print(f'dataset_dir: {dataset_dir}')
    

    # ======================== start generating 3dbox depth dense ========================
    print('======================== start generating 3dbox depth dense ========================')
    print('====================================================================================')
    generate_cude_depth(files_txt, dataset_dir)

if __name__ == '__main__':
    main()