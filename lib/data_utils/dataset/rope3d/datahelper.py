import os
import shutil



def show_scene_img(dataset_list: list, output_dir: str):
    """
    Show the scene image of the rope3d dataset.

    Parameters
    ----------
    dataset_list : list
        A list of paths to the rope3d dataset images
    output_dir : str
        The output directory to save the scene images
    """
    scene_id_set = set()
    for img_path in dataset_list:
        file_name = os.path.basename(img_path)
        scene_id = '_'.join(file_name.split("_")[0:2])
        if scene_id not in scene_id_set:
            print(scene_id)
            print(img_path)
            save_path = os.path.join(output_dir, file_name)
            # check if the directory exists
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            # copy the image to the output directory
            shutil.copyfile(img_path, save_path)
        scene_id_set.add(scene_id)
    print("Total scene ids: ", len(scene_id_set))


def split_dataset_scene(dataset_list: list, output_dir: str):
    """
    Split the rope3d dataset into scenes.

    Parameters
    ----------
    dataset_list : list
        A list of file names of the rope3d dataset images
    output_dir : str
        The output directory to save the dataset txt files
    """
    sceneid_data_dict = {} # to store the data of each scene id, key is scene id, value is a list of image paths
    for data in dataset_list: # data is a file name without file extension
        scene_id = '_'.join(data.split("_")[1:2])
        if scene_id not in sceneid_data_dict:
            sceneid_data_dict[scene_id] = []
        sceneid_data_dict[scene_id].append(data)
    print("Total scene ids: ", len(sceneid_data_dict))

    for scene_id, data_scene in sceneid_data_dict.items():
        print(f'scene_id: {scene_id}')
        # create a new txt file for each scene
        save_path = os.path.join(output_dir, f'{scene_id}.txt')
        # check if the directory exists
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            for data in data_scene:
                f.write(f'{data}\n')
        print(f'The txt file for scene {scene_id} is saved : {save_path}')
        