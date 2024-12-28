import os
from tqdm import tqdm



def retrieve_pmg_files(file_folder: str) -> list:
    """
    Retrieve files with .png extension from a directory.

    Parameters 
    ----------
    file_folder : str
        The path to a folder including files that you want to retrieve.
    
    Returns
    -------
    list
        A list of paths to the retrieve files with .png extension.
    """
    files = []
    for root, _, filenames in os.walk(file_folder):
        for filename in filenames:
            if filename.endswith('.png'):
                files.append(filename)
    return files


def main():
    folder1 = "/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/box3d_depth_dense/box3d_depth_dense/1_1"
    folder2 = "/mnt/data_cfl/Projects/3dod-dataset-tools/output/rope3d/box3d_depth_dense/box3d_depth_dense/2_2"
    folder3 = "/mnt/data_cfl/Projects/Data/Rope3D_data/box3d_depth_dense"
    files1 = retrieve_pmg_files(folder1)
    files2 = retrieve_pmg_files(folder2)
    files3 = retrieve_pmg_files(folder3)
    # get the intersection of the two lists
    files = list(set(files1) & set(files2))
    print(f'The length of files: {len(files)}')
    label_file_path = "/mnt/data_cfl/Projects/3dod-dataset-tools/output/common/labelv2.txt"
    with open(label_file_path, 'w') as ff:
        for f in tqdm(files3, desc='Writing file', colour='green'):
            if f not in files:
                txt_line = "/mnt/data_cfl/Projects/Data/Rope3D_data/label_2_4cls_for_train/" + f.replace('.png', '.txt')
                ff.writelines(txt_line + '\n')

if __name__ == '__main__':
    main()