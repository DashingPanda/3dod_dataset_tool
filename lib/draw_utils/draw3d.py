import matplotlib.pyplot as plt
import numpy as np
import os

import open3d as o3d

import sys
sys.path.append('/mnt/data_cfl/Projects/rcooper-dataset-tools')
from lib.data_utils.post_processor import box_utils
from lib.utils import common_utils




# def draw_center_pred_gt(center_pred, center_gt):
#     """
#     Visualize predicted and ground truth center points in 3D space.

#     This function plots the predicted and ground truth center points in a 3D scatter plot,
#     with each type of point distinguished by color. The plot is saved as an image file.

#     Parameters
#     ----------
#     center_pred : list or numpy array
#         List or numpy array of predicted center points, where each point has X, Y, and Z coordinates.
        
#     center_gt : list or numpy array
#         List or numpy array of ground truth center points, where each point has X, Y, and Z coordinates.

#     Notes
#     -----
#     Predicted centers are plotted in blue, while ground truth centers are plotted in red.
#     The plot is saved to '/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_gt_visual/3d/center/3dscatter.png'.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     for point in center_pred:
#         plt.plot([point[0]], [point[1]], [point[2]], color='blue', marker='o', alpha=0.7)
#     for point in center_gt:
#         plt.plot([point[0]], [point[1]], [point[2]], color='red', marker='o', alpha=0.7)
#     # fig.savefig('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_gt_visual/3d/center/3dscatter.png')
#     # print('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_gt_visual/3d/center/3dscatter.png')


# def draw_point(points):
#     """
#     Visualize a set of points in 3D space.

#     This function creates a 3D scatter plot of the provided points, plotting each point 
#     in a specified color. The plot is saved as an image file.

#     Parameters
#     ----------
#     points : list or numpy array
#         A list or numpy array of points to be plotted, where each point has X, Y, and Z coordinates.

#     Notes
#     -----
#     Each point is plotted in blue, with a semi-transparent marker for visualization. 
#     The plot is saved to '/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_gt_visual/3d/center/3dscatter.png'.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     for point in points:
#         plt.plot([point[0]], [point[1]], [point[2]], color='blue', marker='o', alpha=0.7)
#     fig.savefig('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_gt_visual/3d/center/3dscatter.png')
#     print('/mnt/data_cfl/Projects/rcooper-dataset-tools/output/pred_gt_visual/3d/center/3dscatter.png')


# def draw_box(box_points: np.ndarray, save_img=False):
#     """
#     Draw a 3D bounding box for each set of corner points provided.

#     This function visualizes 3D bounding boxes using their 8 corner points, 
#     allowing multiple boxes to be drawn in the same 3D space. Each box is defined
#     by its 8 corner vertices, and an optional parameter enables saving the image.

#     Parameters
#     ----------
#     box_points : list or numpy array, shape (N, 8, 3+C)
#         A list or numpy array of N 3D boxes, where each box is represented by its 
#         8 corner points. Each corner has X, Y, Z coordinates, and optionally 
#         additional channels (C >= 0).

#     save_img : bool, optional
#         If True, saves the resulting 3D box visualization as an image file.

#     Notes
#     -----
#     The function expects the box corners in the following order:
    
#         4 -------- 5
#        /|         /|
#       7 -------- 6 .
#       | |        | |
#       . 0 -------- 1
#       |/         |/
#       3 -------- 2

#     Connections between vertices are drawn to form the edges of the 3D box.
#     """
#     # conection relationship of the 3D bbox vertex
#     connection = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    
#     # draw the 3D bbox
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     for box in box_points:
#         for i, j in connection:
#             ax.plot([box[i][0], box[j][0]], [box[i][1], box[j][1]], [box[i][2], box[j][2]], color='blue', alpha=0.7)
#     if save_img:
#         img_file = '/mnt/data_cfl/Projects/rcooper-dataset-tools/output/tmp/3d_bbox.png'
#         fig.savefig(img_file)
#         print(f'the 3D bbox image is saved : {img_file}')


# def draw_box_multiscene(box_points: list, save_img=False):
#     """
#     Draw 3D bounding boxes for multiple scenes, where each scene contains multiple boxes.

#     This function visualizes 3D bounding boxes across multiple scenes. Each scene may contain multiple 
#     boxes, and each box is defined by its 8 corner points. An optional parameter allows saving the 
#     visualization as an image file.

#     Parameters
#     ----------
#     box_points : list, length S, where S is the number of scenes, and each element is a list or numpy array, shape (N, 8, 3+C)
#         A list or numpy array representing S scenes. Each scene contains N 3D boxes, where each box 
#         is represented by its 8 corner points with X, Y, Z coordinates.

#     save_img : bool, optional
#         If True, saves the generated 3D box visualization as an image file.

#     Notes
#     -----
#     The box corners are expected in the following order for each box:
    
#         4 -------- 5
#        /|         /|
#       7 -------- 6 .
#       | |        | |
#       . 0 -------- 1
#       |/         |/
#       3 -------- 2

#     Connections are drawn between vertices to form the edges of each 3D box.
#     """
#     # conection relationship of the 3D bbox vertex
#     connection = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
#     # color for each scene
#     color_scene = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan', 'lime']
    
#     # draw the 3D bbox
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     for scene_id, scene_boxes in enumerate(box_points):
#         for box in scene_boxes:
#             for i, j in connection:
#                 ax.plot([box[i][0], box[j][0]], [box[i][1], box[j][1]], [box[i][2], box[j][2]], color=color_scene[scene_id], alpha=0.7)
#     if save_img:
#         img_file = '/mnt/data_cfl/Projects/rcooper-dataset-tools/output/3D/3d_bbox_multiscene.png'
#         fig.savefig(img_file)
#         print(f'the 3D bbox image is saved : {img_file}')

#     plt.show()




# ========= open3d visualization ============

def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    # if not isinstance(bbx_corner, np.ndarray):
    #     bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner, order)
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def visualize_single_sample_output_gt(gt_tensor_list, show_vis=True):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """


    def custom_draw_geometry(gt_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        # opt.point_size = 1.0

        for gt in gt_list:
            for ele in gt:
                vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    # origin_lidar = pcd
    # if not isinstance(pcd, np.ndarray):
    #     origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    # origin_lidar_intcolor = \
    #     color_encoding(origin_lidar[:, -1] if mode == 'intensity'
    #                    else origin_lidar[:, 2], mode=mode)
    # # left -> right hand
    # origin_lidar[:, :1] = -origin_lidar[:, :1]

    # o3d_pcd = o3d.geometry.PointCloud()
    # o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    # o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    # color list for 5 scenes
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0))
    oabbs_gt_list = []
    for i, gt_tensor in enumerate(gt_tensor_list):
        gt_tensor = gt_tensor[:, :, :3]
        print(f'the shape of gt_tensor is {gt_tensor.shape}')
        oabbs_gt = bbx2oabb(gt_tensor, color=color_list[i])
        oabbs_gt_list.append(oabbs_gt)

    # visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        custom_draw_geometry(oabbs_gt_list)
    # if save_path:
    #     save_o3d_visualization(visualize_elements, save_path)
        


def draw_3dbox_pred_gt(pred_corners: np.ndarray, gt_corners: np.ndarray, show_vis=True):
    """
    Parameters
    ----------
    pred_corners : np.ndarray
        (N, 8, 3) prediction.

    gt_corners : np.ndarray
        (N, 8, 3) groundtruth bbx

    show_vis : bool
        Whether to show visualization.

    """

    def custom_draw_geometry(gt_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        # opt.point_size = 1.0

        # # draw coordinate frame
        # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5*500, origin=[0, 0, 0])
        # vis.add_geometry(axis_pcd)

        for ele in gt_list:
            vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    # # left -> right hand
    # origin_lidar[:, :1] = -origin_lidar[:, :1]
    pred_corners[:, :, :1] = -pred_corners[:, :, :1]
    gt_corners[:, :, :1] = -gt_corners[:, :, :1]

    oabbs_pred = bbx2oabb(pred_corners, color=(0, 1, 0))
    oabbs_gt = bbx2oabb(gt_corners, color=(1, 0, 0))
    # concatenate the two lists
    oabbs = oabbs_gt + oabbs_pred
    
    if show_vis:
        custom_draw_geometry(oabbs)