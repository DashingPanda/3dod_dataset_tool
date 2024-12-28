import cv2
import numpy as np
import json
import argparse
import os

from lib.utils import common_utils, transformation
from lib.data_utils.data import Data
from lib.data_utils.post_processor import box_utils
from lib.data_utils.dataset.rope3d import labelloader as rope3d_labelloader
from lib.data_utils.dataset.rcooper import labelloader as rcooper_labelloader
from lib.draw_utils import draw2d



def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--img_file', type=str, required=True,
                        default='/mnt/data_cfl/Projects/Data/Rcooper_original/data/106-105/105/seq-0/cam-0/1692685344.971904.jpg',
                        help='the directory of the image of the rcooper dataset')
    parser.add_argument('--pred_txt_file', type=str, required=False, 
                        help='the directory of a predict result txt file')
    parser.add_argument('--gt_file', type=str, required=False, 
                        help='the directory of a predict result file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory of the output folder of the rcooper dataset lael npy file')
    parser.add_argument('--draw_type', type=str, required=True,
                        help='the type of the visualization of the predicted result, including 3dbbox, 3dbbox_center')
    parser.add_argument('--draw_score_threshold', type=float, required=True, default=0.1,
                        help='show the predicted result with score greater than the threshold')
    opt = parser.parse_args()
    return opt


def visualize_pred_gt(img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold):
    scene_id = img_file.split('/')[8]
    camera_id = img_file.split('/')[10]
    print(f'scene_id: {scene_id}')
    print(f'camera_id: {camera_id}')
    assert scene_id in ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139']
    assert camera_id in ['cam-0', 'cam-1']

    cam_intri, lid2cam_extri, cam2lid_extri, lid2world_extri, world2lid_extri = rcooper_labelloader.get_transformation_matrix(scene_id, camera_id)

    img = cv2.imread(img_file)


    if draw_type == '3dbbox_center':
        # ======== draw the predicted location in image coordinate system ========
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
        print(f'the number of pred_result_lines after filtering: {pred_location_dimensions_rotation_cam.shape[0]}')
        pred_location_cam = pred_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        pred_location_cam = common_utils.covert2homogeneous(pred_location_cam) # (N, 4)
        # transform the pred_location_cam to image coordinate system from camera coordinate system
        pred_location_img = transformation.cam2img(pred_location_cam, cam_intri) # (N, 2)
        # draw the predicted location in image coordinate system
        # img_visual = draw2d.draw_3dboxcenter(img, pred_location_img, (0, 255, 0))
        # transform the pred_location_cam to lidar coordinate system from camera coordinate system
        pred_location_lid = transformation.cam2lid(pred_location_cam, cam2lid_extri) # (N, 4)
        img_visual = draw2d.draw_3dboxcenter_with_distence(img, pred_location_img, pred_location_lid, (0, 255, 0))
        # common_utils.save_img(img_visual, output_dir, img_file)

        # ======== draw the gt location in image coordinate system ========
        gt_label_list = json.load(open(gt_file, 'r'))
        print(f'the number of gt: {len(gt_label_list)}')
        # load label from the gt label file
        objs = [] # list of objects in the gt label file
        for gt_label in gt_label_list:
            gt_data = Data()
            rcooper_labelloader.get_label_data(gt_data, gt_label)
            objs.append(gt_data)
        # get the gt_location_dimensions_rotation in lidar coordinate system
        gt_location_dimensions_rotation = common_utils.get_location_dimensions_rotation(objs)
        gt_location_lid = gt_location_dimensions_rotation[:, :3]  # (N, 3), location in lidar coordinate system
        # convert the gt_location_lid to homogeneous coordinates
        gt_location_lid = common_utils.covert2homogeneous(gt_location_lid)  # (N, 4)
        # transform the gt_location_lid to camera coordinate system from lidar coordinate system
        gt_location_cam = transformation.lid2cam(gt_location_lid, lid2cam_extri)  # (N, 4)
        # transform the gt_location_cam to image coordinate system from camera coordinate system
        gt_location_img = transformation.cam2img(gt_location_cam, cam_intri)  # (N, 2)
        # draw the gt location in image coordinate system
        # img_visual = draw2d.draw_3dboxcenter(img, gt_location_img)
        img_visual = draw2d.draw_3dboxcenter_with_distence(img_visual, gt_location_img, gt_location_lid)

        # save the visualization image
        common_utils.save_img(img_visual, output_dir, img_file, dataset='rcooper')

    elif draw_type == '3dbbox':
        # ======== draw the predicted 3d bounding box in image coordinate system ========
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
        print(f'the number of pred_result_lines after filtering: {pred_location_dimensions_rotation_cam.shape[0]}')
        # transform the pred_location_dimensions_rotation_cam to lidar coordinate system from camera coordinate system
        pred_location_cam = pred_location_dimensions_rotation_cam[:, :3]  # (N, 3), location in camera coordinate system
        pred_location_cam = common_utils.covert2homogeneous(pred_location_cam) # (N, 4), homogeneous coordinates
        pred_location_lid = transformation.cam2lid(pred_location_cam, cam2lid_extri) # (N, 4), location in lidar coordinate system
        pred_location_lid = pred_location_lid[:, :3]  # (N, 3), location in lidar coordinate system
        pred_rotation_cam = pred_location_dimensions_rotation_cam[:, -1]  # (N, 1), rotation in camera coordinate system
        pred_rotation_lid = transformation.cam2lid_rotation(pred_rotation_cam, cam2lid_extri)
        pred_location_dimensions_rotation_lid = np.concatenate([pred_location_lid, pred_location_dimensions_rotation_cam[:, 3:6], pred_rotation_lid], axis=1)  # (N, 7)
        # get the 8 corners of a 3d bounding box in lidar coordinate system
        pred_corners_lid = box_utils.boxes_to_corners_3d(pred_location_dimensions_rotation_lid, 'whl')  # (N, 8, 3)
        pred_corners_lid = pred_corners_lid.reshape(-1, 3)  # (N*8, 3) in lidar coordinate system
        # convert the pred_corners_lid to homogeneous coordinates
        pred_corners_lid = common_utils.covert2homogeneous(pred_corners_lid)  # (N*8, 4)
        # transform the pred_corners_lid to camera coordinate system from lidar coordinate system
        pred_corners_cam = transformation.lid2cam(pred_corners_lid, lid2cam_extri)  # (N*8, 4)
        # transform the pred_corners_cam to image coordinate system from camera coordinate system
        pred_corners_img = transformation.cam2img(pred_corners_cam, cam_intri)  # (N*8, 2)
        pred_corners_img = pred_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
        # draw the predicted 3d bounding box in image coordinate system
        img_visual = draw2d.draw_3dbox(img, pred_corners_img, (0, 255, 0))


        # ======== draw the gt 3d bounding box in image coordinate system ========
        gt_label_list = json.load(open(gt_file, 'r'))
        print(f'the number of gt: {len(gt_label_list)}')
        # load label from the gt label file
        objs = [] # list of objects in the gt label file
        for gt_label in gt_label_list:
            gt_data = Data()
            rcooper_labelloader.get_label_data(gt_data, gt_label)
            objs.append(gt_data)
        # get the gt_location_dimensions_rotation in lidar coordinate system
        gt_location_dimensions_rotation = common_utils.get_location_dimensions_rotation(objs)
        # get the 8 corners of a 3d bounding box in lidar coordinate system
        gt_corners_lid = box_utils.boxes_to_corners_3d(gt_location_dimensions_rotation, 'hwl')  # (N, 8, 3)
        gt_corners_lid = gt_corners_lid.reshape(-1, 3)  # (N*8, 3) in lidar coordinate system
        # convert the gt_corners_lid to homogeneous coordinates
        gt_corners_lid = common_utils.covert2homogeneous(gt_corners_lid)  # (N*8, 4)
        # transform the gt_corners_lid to camera coordinate system from lidar coordinate system
        gt_corners_cam = transformation.lid2cam(gt_corners_lid, lid2cam_extri)  # (N*8, 4)
        # transform the gt_corners_cam to image coordinate system from camera coordinate system
        gt_corners_img = transformation.cam2img(gt_corners_cam, cam_intri)  # (N*8, 2)
        gt_corners_img = gt_corners_img.reshape(-1, 8, 2)  # (N, 8, 2) in image coordinate system
        # draw the gt 3d bounding box in image coordinate system
        img_visual = draw2d.draw_3dbox(img, gt_corners_img)
        # save the visualization image
        output_dir = os.path.join(output_dir, draw_type)
        common_utils.save_img(img_visual, output_dir, img_file, dataset='rcooper')



def main():
    opt = args_parser()
    img_file = opt.img_file
    pred_txt_file = opt.pred_txt_file
    gt_file = opt.gt_file
    output_dir = opt.output_dir
    draw_type = opt.draw_type
    draw_score_threshold = opt.draw_score_threshold


    # ======================== visualization parameters ========================
    print('======================== visualization parameters ========================')
    print('==========================================================================')
    print(f'img_file: {img_file}')
    print(f'pred_txt_file: {pred_txt_file}')
    print(f'gt_file: {gt_file}')
    print(f'output_dir: {output_dir}')
    print(f'draw_type: {draw_type}')
    print(f'draw_score_threshold: {draw_score_threshold}')

    # ======================== start visualization ========================
    print('======================== start visualization ========================')
    print('=====================================================================')
    visualize_pred_gt(img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold)
    print('======================== end visualization ========================')
    print('===================================================================')
    print('\n')



if __name__ == '__main__':
    main()