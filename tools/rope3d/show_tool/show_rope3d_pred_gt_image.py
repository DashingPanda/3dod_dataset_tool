import os
import cv2
import numpy as np
import argparse

from lib.utils import common_utils, transformation
from lib.data_utils.data import Data
from lib.data_utils.post_processor import box_utils
from lib.draw_utils import draw2d
from lib.data_utils.dataset.rope3d import labelloader as rope3d_labelloader


def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='the path to the directory containing the dataset')
    parser.add_argument('--img_file', type=str, required=True,
                        default='/mnt/data_cfl/Projects/Data/Rcooper_original/data/106-105/105/seq-0/cam-0/1692685344.971904.jpg',
                        help='the directory of the image of the rcooper dataset')
    parser.add_argument('--pred_txt_file', type=str, required=False, 
                        help='the directory of a predict result txt file')
    parser.add_argument('--gt_file', type=str, required=False, 
                        help='the directory of a predict result file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--draw_type', type=str, required=True, choices=['3dbbox', '3dbbox_center', '3dbbox_Zcam'],
                        help='the type of the visualization of the predicted result, including 3dbbox, 3dbbox_center, 3dbbox_Zcam')
    parser.add_argument('--draw_score_threshold', type=float, required=True, default=0.1,
                        help='show the predicted result with score greater than the threshold')
    opt = parser.parse_args()
    return opt





def calculate_distance(locations: np.ndarray, denorm: np.ndarray):
    print(f'locations: {locations.shape}') # (N, 4)
    print(f'denorm: {denorm.shape}') # (4, )

    a = np.dot(locations, denorm)
    print(f'a: {a.shape}')
    print(f'a: {a}')

    # calculate denorm[0] ** 2 + denorm[1] ** 2 + denorm[2] ** 2
    denominator = np.sum(denorm[:-1] ** 2)

    t = -denorm[-1] / denominator
    project_point = denorm[:-1] * t
    distance_camera = np.sqrt(np.sum(project_point ** 2))

    t = -(np.sum(a)) / denominator
    project_points = (denorm * t)[:-1] + locations[:, :-1] # (N, 3)
    distance_objs = np.sqrt(np.sum((project_points) ** 2, axis=1))

    distance = distance_objs - distance_camera
    print(f't: {t}')
    print(f'denorm: {denorm}')
    # print(f'project_points: {project_points}') 

    print(f'distance_camera: {distance_camera}')
    print(f'distance_objs: {distance_objs}')
    print(f'distance: {distance}')
    
    return distance
    


def visualize_pred_gt(dataset_dir, img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold):
    """
    Paratemers
    ----------
    dataset_dir : str
        the path to the directory containing the dataset
    img_file : str
        the path of the image of the rcooper dataset
    """

    filename_woe = os.path.basename(img_file).split('.')[0] # filename without extension

    cam_intri, cam2world_extri, world2cam_extri = rope3d_labelloader.get_transformation_matrix(filename_woe, dataset_dir)

    # print(f'cam_intri: {cam_intri}')
    # exit()

    img = cv2.imread(img_file)
    
    if draw_type == '3dbbox_center':
        pass

    elif draw_type == '3dbbox':
        # ================ draw the predicted 3d bounding box in image coordinate system ================
        pred_result_lines = open(pred_txt_file, 'r').readlines()
        print(f'the number of pred_result_lines: {len(pred_result_lines)}')
        # load label from the predicted result file
        objs = [] # list of valid objects in the predicted result file
        for pred_result_line in pred_result_lines:
            pred_result_line = pred_result_line.strip()
            pred_data = Data()
            rope3d_labelloader.get_label_data(pred_data, pred_result_line)
            if pred_data.score < draw_score_threshold:
                continue
            objs.append(pred_data)
        # get the pred_location_dimensions_rotation_cam in camera coordinate system
        pred_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)
        print(f'the number of predicted objects after filtering: {len(objs)}')

        pred_location_cam = pred_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        pred_location_cam = common_utils.covert2homogeneous(pred_location_cam) # (N, 4), homogeneous coordinates
        pred_location_world = transformation.cam2wor(pred_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
        pred_location_world = pred_location_world[:, :3]  # (N, 3), location in world coordinate systems
        pred_rotation_cam = pred_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
        pred_rotation_world = transformation.cam2wor_rotation(pred_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
        pred_location_dimensions_rotation_world = np.concatenate([pred_location_world, pred_location_dimensions_rotation_cam[:, 3:6], pred_rotation_world], axis=1)  # (N, 7)
        # get the 8 corners of a 3d bounding box in world coordinate system
        pred_corners_world = box_utils.boxes_to_corners_3d(pred_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)
        pred_corners_world = pred_corners_world.reshape(-1, 3)  # (N*8, 3) in world coordinate system
        # convert the pred_corners_world to homogeneous coordinates
        pred_corners_world = common_utils.covert2homogeneous(pred_corners_world)  # (N*8, 4)
        # transform the pred_corners_worlde to camera coordinate system from world coordinate system
        pred_corners_cam = transformation.wor2cam(pred_corners_world, world2cam_extri)  # (N*8, 4)
        # transform the pred_corners_cam to image coordinate system from camera coordinate system
        pred_corners_img = transformation.cam2img(pred_corners_cam, cam_intri)  # (N*8, 2)
        pred_corners_img = pred_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
        # draw the predicted 3d bounding box in image coordinate system
        img_visual = draw2d.draw_3dbox(img, pred_corners_img, (0, 255, 0))

        # ================ draw the gt 3d bounding box in image coordinate system ================
        gt_result_lines = open(gt_file, 'r').readlines()
        # load label from the predicted result file
        objs = [] # list of valid objects in the predicted result file
        for gt_result_line in gt_result_lines:
            gt_result_line = gt_result_line.strip()
            gt_data = Data()
            rope3d_labelloader.get_label_data(gt_data, gt_result_line)
            if gt_data.obj_type not in ['car', 'van', 'bus', 'cyclist']: # only show 3d bounding box for car, van, bus, cyclist
                continue
            if gt_data.X == 0 and gt_data.Y == 0 and gt_data.Z == 0: # No 3d bounding box
                continue
            objs.append(gt_data)
        print(f'the number of gt objects: {len(objs)}')
        # get the gt_location_dimensions_rotation in camera coordinate system
        gt_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)

        gt_location_cam = gt_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        gt_location_cam = common_utils.covert2homogeneous(gt_location_cam) # (N, 4), homogeneous coordinates
        gt_location_world = transformation.cam2wor(gt_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
        gt_location_world = gt_location_world[:, :3]  # (N, 3), location in world coordinate systems
        gt_rotation_cam = gt_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
        gt_rotation_world = transformation.cam2wor_rotation(gt_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
        gt_location_dimensions_rotation_world = np.concatenate([gt_location_world, gt_location_dimensions_rotation_cam[:, 3:6], gt_rotation_world], axis=1)  # (N, 7)
        # get the 8 corners of a 3d bounding box in world coordinate system
        gt_corners_world = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)
        gt_corners_world = gt_corners_world.reshape(-1, 3)  # (N*8, 3) in world coordinate system
        # convert the pred_corners_world to homogeneous coordinates
        gt_corners_world = common_utils.covert2homogeneous(gt_corners_world)  # (N*8, 4)
        # transform the pred_corners_worlde to camera coordinate system from world coordinate system
        gt_corners_cam = transformation.wor2cam(gt_corners_world, world2cam_extri)  # (N*8, 4)
        # transform the pred_corners_cam to image coordinate system from camera coordinate system
        gt_corners_img = transformation.cam2img(gt_corners_cam, cam_intri)  # (N*8, 2)
        gt_corners_img = gt_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
        # draw the predicted 3d bounding box in image coordinate system
        img_visual = draw2d.draw_3dbox(img_visual, gt_corners_img)

        # ================ save the image with predicted and gt 3d bounding box ================
        common_utils.save_img(img_visual, output_dir, img_file, dataset='rope3d')

    elif draw_type == '3dbbox_Zcam':
        # TODO: trial
        basename = os.path.basename(pred_txt_file).split('.')[0]
        denorm_file = f'/mnt/data_cfl/Projects/Data/Rope3D_data/denorm/{basename}.txt'
        denorm = open(denorm_file, 'r').readlines()[0].strip().split(' ')
        denorm = np.array([float(d) for d in denorm])
        print(f'denorm: {denorm}')

        # ================ draw the predicted 3d bounding box in image coordinate system ================
        pred_result_lines = open(pred_txt_file, 'r').readlines()
        print(f'the number of pred_result_lines: {len(pred_result_lines)}')
        # load label from the predicted result file
        objs = [] # list of valid objects in the predicted result file
        for pred_result_line in pred_result_lines:
            pred_result_line = pred_result_line.strip()
            pred_data = Data()
            rope3d_labelloader.get_label_data(pred_data, pred_result_line)
            if pred_data.score < draw_score_threshold:
                continue
            objs.append(pred_data)
        # get the pred_location_dimensions_rotation_cam in camera coordinate system
        pred_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)
        print(f'the number of predicted objects after filtering: {len(objs)}')
        Zcam = pred_location_dimensions_rotation_cam[:, 2]
        # transform the pred_location_dimensions_rotation_cam to lidar coordinate system from camera coordinate system
        pred_location_cam = pred_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        pred_location_cam = common_utils.covert2homogeneous(pred_location_cam) # (N, 4), homogeneous coordinates
        distance = calculate_distance(pred_location_cam, denorm)
        pred_location_world = transformation.cam2wor(pred_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
        pred_location_world = pred_location_world[:, :3]  # (N, 3), location in world coordinate systems
        pred_rotation_cam = pred_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
        pred_rotation_world = transformation.cam2wor_rotation(pred_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
        pred_location_dimensions_rotation_world = np.concatenate([pred_location_world, pred_location_dimensions_rotation_cam[:, 3:6], pred_rotation_world], axis=1)  # (N, 7)
        # get the 8 corners of a 3d bounding box in world coordinate system
        pred_corners_world = box_utils.boxes_to_corners_3d(pred_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)
        pred_corners_world = pred_corners_world.reshape(-1, 3)  # (N*8, 3) in world coordinate system
        # convert the pred_corners_world to homogeneous coordinates
        pred_corners_world = common_utils.covert2homogeneous(pred_corners_world)  # (N*8, 4)
        # transform the pred_corners_worlde to camera coordinate system from world coordinate system
        pred_corners_cam = transformation.wor2cam(pred_corners_world, world2cam_extri)  # (N*8, 4)
        # transform the pred_corners_cam to image coordinate system from camera coordinate system
        pred_corners_img = transformation.cam2img(pred_corners_cam, cam_intri)  # (N*8, 2)
        pred_corners_img = pred_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
        # draw the predicted 3d bounding box in image coordinate system
        img_visual = draw2d.draw_3dbox_Zcam(img, pred_corners_img, Zcam, (0, 255, 0))
        # img_visual = draw2d.draw_3dbox_Zcam(img, pred_corners_img, distance, (0, 255, 0))

        # ================ draw the gt 3d bounding box in image coordinate system ================
        gt_result_lines = open(gt_file, 'r').readlines()
        # load label from the predicted result file
        objs = [] # list of valid objects in the predicted result file
        for gt_result_line in gt_result_lines:
            gt_result_line = gt_result_line.strip()
            gt_data = Data()
            rope3d_labelloader.get_label_data(gt_data, gt_result_line)
            if gt_data.obj_type not in ['car', 'van', 'bus', 'cyclist']: # only show 3d bounding box for car, van, bus, cyclist
                continue
            if gt_data.X == 0 and gt_data.Y == 0 and gt_data.Z == 0: # No 3d bounding box
                continue
            objs.append(gt_data)
        print(f'the number of gt objects: {len(objs)}')
        # get the gt_location_dimensions_rotation in lidar coordinate system
        gt_location_dimensions_rotation_cam = common_utils.get_location_dimensions_rotation(objs)
        # transform the pred_location_dimensions_rotation_cam to lidar coordinate system from camera coordinate system
        gt_location_cam = gt_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        gt_location_cam = common_utils.covert2homogeneous(gt_location_cam) # (N, 4), homogeneous coordinates
        gt_location_world = transformation.cam2wor(gt_location_cam, cam2world_extri) # (N, 4), location in world coordinate system
        gt_location_world = gt_location_world[:, :3]  # (N, 3), location in world coordinate systems
        gt_rotation_cam = gt_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
        gt_rotation_world = transformation.cam2wor_rotation(gt_rotation_cam, cam2world_extri)   # (N, 1), rotation in world coordinate system     
        gt_location_dimensions_rotation_world = np.concatenate([gt_location_world, gt_location_dimensions_rotation_cam[:, 3:6], gt_rotation_world], axis=1)  # (N, 7)
        # get the 8 corners of a 3d bounding box in world coordinate system
        gt_corners_world = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation_world, 'hwl', isCenter=False)  # (N, 8, 3)
        gt_corners_world = gt_corners_world.reshape(-1, 3)  # (N*8, 3) in world coordinate system
        # convert the pred_corners_world to homogeneous coordinates
        gt_corners_world = common_utils.covert2homogeneous(gt_corners_world)  # (N*8, 4)
        # transform the pred_corners_worlde to camera coordinate system from world coordinate system
        gt_corners_cam = transformation.wor2cam(gt_corners_world, world2cam_extri)  # (N*8, 4)
        # transform the pred_corners_cam to image coordinate system from camera coordinate system
        gt_corners_img = transformation.cam2img(gt_corners_cam, cam_intri)  # (N*8, 2)
        gt_corners_img = gt_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
        # draw the predicted 3d bounding box in image coordinate system
        # img_visual = draw2d.draw_3dbox(img_visual, gt_corners_img)
        # img_visual = draw2d.draw_3dbox_Zcam(img_visual, gt_corners_img, Zcam)

        # ================ save the image with predicted and gt 3d bounding box ================
        common_utils.save_img(img_visual, output_dir, img_file, dataset='rope3d')
        

def main():
    opt = args_parser()
    dataset_dir = opt.dataset_dir
    img_file = opt.img_file
    pred_txt_file = opt.pred_txt_file
    gt_file = opt.gt_file
    output_dir = opt.output_dir
    draw_type = opt.draw_type
    draw_score_threshold = opt.draw_score_threshold


    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'dataset_dir: {dataset_dir}')
    print(f'img_file: {img_file}')
    print(f'pred_txt_file: {pred_txt_file}')
    print(f'gt_file: {gt_file}')
    print(f'output_dir: {output_dir}')
    print(f'draw_type: {draw_type}')
    print(f'draw_score_threshold: {draw_score_threshold}')

    # ======================== start visualization ========================
    print('======================== start visualization ========================')
    print('=====================================================================')
    visualize_pred_gt(dataset_dir, img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold)
    print('======================== end visualization ========================')
    print('===================================================================')
    print('\n')



if __name__ == '__main__':
    main()