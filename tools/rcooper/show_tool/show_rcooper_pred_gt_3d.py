import cv2
import numpy as np
import json
import argparse
import os


from lib.utils import common_utils
from lib.data_utils.data import Data
from lib.draw_utils import draw3d
from lib.data_utils.dataset.rope3d import labelloader as rope3d_labelloader
from lib.data_utils.dataset.rcooper import labelloader as rcooper_labelloader



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



def draw_gt_3dbox(img, gt, lid2cam_extri, lid2cam_intri):
    """
    Parameters
    ----------
    img : np.ndarray

    gt : 3D bbox corner points in lidar coordinate system
        np.ndarray
        (N, 8, 3)
    """
    # conection relationship of the 3D bbox vertex
    connection = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    print(f'the number of gt: {len(gt)}')

    # draw the 3D bbox on the image
    for i, box_obj in enumerate(gt):
        box_obj = np.hstack((box_obj, np.ones((8,1))))
        box_obj_cam = (lid2cam_extri @ box_obj.T).T
        box_obj_cam = box_obj_cam[:, :3]
        box_obj_img = (lid2cam_intri @ box_obj_cam.T).T
        box_obj_img = box_obj_img[:, :2] / box_obj_img[:, 2:]
        # draw the 2D bbox on the image
        for connect in connection:
            cv2.line(img, (int(box_obj_img[connect[0]][0]), int(box_obj_img[connect[0]][1])), 
                    (int(box_obj_img[connect[1]][0]), int(box_obj_img[connect[1]][1])), (0, 0, 255), 2)
    return img


def draw_gt_3dboxcenter(img, gt, lid2cam_extri, lid2cam_intri):
    """
    Parameters
    ----------
    img : np.ndarray

    gt : 3D bbox center points in lidar coordinate system
        np.ndarray
        (N, 3)
    """
    print(f'lid2cam_extri:\n{lid2cam_extri}')
    print(f'lid2cam_intri:\n{lid2cam_intri}')

    # draw the 3D bbox center on the image
    for i, center in enumerate(gt):
        center_cam = (lid2cam_extri @ np.hstack((center, 1)).T).T
        center_cam = center_cam[:3]
        center_img = (lid2cam_intri @ center_cam).T
        center_img = center_img[:2] / center_img[2]
        cv2.circle(img, (int(center_img[0]), int(center_img[1])), 5, (0, 0, 255), -1)
    return img



def save_img(img, output_dir, img_file):
    output_path = os.path.join(output_dir, img_file.split('data/')[-1])
    print(f'output_path: {output_path}')
    # check if the output directory exists, if not, create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    cv2.imwrite(output_path, img)


def visualize_pred(img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold):
    scene_id = img_file.split('/')[8]
    camera_id = img_file.split('/')[10]
    print(f'scene_id: {scene_id}')
    print(f'camera_id: {camera_id}')
    assert scene_id in ['105', '106', '115', '116', '117', '118', '119', '120', '136', '137', '138', '139']
    assert camera_id in ['cam-0', 'cam-1']

    cam_intri, lid2cam_extri, cam2lid_extri, lid2world_extri, world2lid_extri = rcooper_labelloader.get_transformation_matrix(scene_id, camera_id)

    img = cv2.imread(img_file)


    if draw_type == '3dbbox_center':
        # get the predicted location in Lidar coordinate system
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
        # transform the pred_location_cam to lidar coordinate system from camera coordinate system
        pred_location_cam = pred_location_dimensions_rotation_cam[:, :3]
        ones = np.ones((len(pred_location_dimensions_rotation_cam), 1))
        pred_location_lid = (cam2lid_extri @ np.hstack((pred_location_cam, ones)).T).T[:, :3]  # (N, 3)

        # get the gt location in Lidar coordinate system
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
        gt_location_lid = gt_location_dimensions_rotation[:, :3]  # (N, 3)

        # draw the predicted and gt location in Lidar coordinate system
        # draw3d.draw_center_pred_gt(pred_location_lid, gt_location_lid)

    # save_img(img_visual, output_dir, img_file)


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
    visualize_pred(img_file, pred_txt_file, gt_file, output_dir, draw_type, draw_score_threshold)
    print('======================== end visualization ========================')
    print('===================================================================')
    print('\n')



if __name__ == '__main__':
    main()