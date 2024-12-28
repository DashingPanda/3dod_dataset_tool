import os
import argparse
import cv2
import numpy as np
import json

import sys
sys.path.append('/mnt/data_cfl/Projects/rcooper-dataset-tools')

from lib.utils import common_utils, transformation
from lib.draw_utils import draw3d
from lib.data_utils.post_processor import box_utils
from lib.dataset.rcooper import labelloader






def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--label_info_file', type=str, required=True,
                        default='/mnt/data_cfl/Projects/Data/Rcooper/label/infos/136-137-138-139_0-info.json',
                        help='the path of the label info file of the rcooper dataset')
    parser.add_argument('--frame_id', type=int, required=True, default=0,
                        help='the frame id in a sequence data to be verified')
    opt = parser.parse_args()
    return opt


def verified_in_world(label_info_frame, label_dir, scene_id_list):
    # get the label file path of the coorpative scene
    coop_label_file = label_info_frame['coop'].split('cooperative/')[-1]
    coop_label_file = os.path.join(label_dir, coop_label_file)
    ego_id = label_info_frame['ego_id']
    print(f'coop_label_file: {coop_label_file}')
    print(f'label_info_frame: {label_info_frame}')

    lid2world_extri_coop, world2lid_extri_coop = labelloader.get_transformation_matrix(ego_id)

    # retrieve the label file path of corresponding scene and save it in a dictionary
    # scene_label_file_list = {scene_id: label_file_path corresponding to the scene_id}
    scene_label_file_list = dict()
    for scene_id in scene_id_list:
        scene_label_file_list[scene_id] = os.path.join(label_dir, label_info_frame[scene_id].split('cooperative/')[-1])

    # get location, dimensions, rotation, type, and track_id of the objects in the coorpative scene
    coop_location_dimensions_rotation, coop_type_trackid = labelloader.get_label_location_dimensions_rotation(coop_label_file, return_type_trackid=True) # (N, 7), (N, 2)
    # get the 8 corners of a 3d bounding box in lidar coordinate system
    coop_corners_lid = box_utils.boxes_to_corners_3d(coop_location_dimensions_rotation, 'whl')  # (N, 8, 3)
    coop_corners_lid = coop_corners_lid.reshape(-1, 3) # (N*8, 3)
    coop_corners_lid = common_utils.covert2homogeneous(coop_corners_lid) # (N*8, 4)
    # transform the 8 corner points in lidar coordinate system to the world coordinate system
    coop_corners_world = transformation.lid2wor(coop_corners_lid, lid2world_extri_coop) # (N*8, 4)
    coop_corners_world = coop_corners_world.reshape(-1, 8, 4) # (N, 8, 4)
    # draw the 3d bounding box in the world coordinate system
    # draw3d.draw_box(coop_corners_world, save_img=True)


    # get the location, dimensions, rotation, type, and track_id of the objects in the corresponding scene
    scene_corners_world_all = list()
    for scene_id in scene_id_list:
        # get the transformatin matrix between the lidar and the world coordinate system of the corresponding scene
        lid2world_extri_scene, world2lid_extri_scene = labelloader.get_transformation_matrix(scene_id)
        # get the location, dimensions, rotation, type, and track_id of the objects 
        scene_location_dimensions_rotation, scene_type_trackid = labelloader.get_label_location_dimensions_rotation(scene_label_file_list[scene_id], return_type_trackid=True) # (N, 7), (N, 2)
        # get the 8 corners of a 3d bounding box in lidar coordinate system
        scene_corners_lid = box_utils.boxes_to_corners_3d(scene_location_dimensions_rotation, 'whl')  # (N, 8, 3)
        scene_corners_lid = scene_corners_lid.reshape(-1, 3) # (N*8, 3)
        scene_corners_lid = common_utils.covert2homogeneous(scene_corners_lid) # (N*8, 4)
        # transform the 8 corner points in lidar coordinate system to the world coordinate system
        scene_corners_world = transformation.lid2wor(scene_corners_lid, lid2world_extri_scene) # (N*8, 4)
        scene_corners_world = scene_corners_world.reshape(-1, 8, 4) # (N, 8, 4)
        print(f'the shape of scene_corners_world: {scene_corners_world.shape}')
        # save the 3d bounding box of all objects in the corresponding scene
        # scene_corners_world_all = np.hstack((scene_corners_world_all, scene_corners_world), axis=0) # (S, N, 8, 4)
        scene_corners_world_all.append(scene_corners_world)

    # draw3d.draw_box_multiscene(scene_corners_world_all, save_img=True)
    scene_corners_world = scene_corners_world[:, :, :3]
    draw3d.visualize_single_sample_output_gt(scene_corners_world_all)

    print(f'the length of scene_corners_world_all: {len(scene_corners_world_all)}')

        # # draw the 3d bounding box in the world coordinate system
        # draw3d.draw_box(scene_corners_world, save_img=True) 



        



def main():
    opt = args_parser()
    label_info_file = opt.label_info_file
    frame_id = opt.frame_id
    
    label_dir = label_info_file.split('/infos/')[0]
    print(f'label_info_file: {label_info_file}')
    print(f'label_dir: {label_dir}')
    print(f'frame_id: {frame_id}')

    label_info_json = json.load(open(label_info_file, 'r'))
    label_info_json_filename = os.path.basename(label_info_file)
    scene_id_list = label_info_json_filename.split('_')[0].split('-') # the scene IDs that the label info file contains
    print(f'scene_id_list: {scene_id_list}')
    
    if frame_id >= len(label_info_json):
        print(f'frame_id {frame_id} is out of range of the sequence data')
        exit()

    label_info_frame = label_info_json[frame_id]

    verified_in_world(label_info_frame, label_dir, scene_id_list)

    
    


if __name__ == '__main__':
    main()