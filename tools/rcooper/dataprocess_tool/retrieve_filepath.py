import os
import argparse




def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--folder', type=str, required=True,
                        help='the directory of the image of the rcooper dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory to save the output file')
    opt = parser.parse_args()
    return opt


def retrieve_path(directory: str) -> list:
    """
    This function retrieves the relative file path of all the images in the given directory and its subdirectories.

    Parameters:
    directory (str): the directory of the image of the rcooper dataset

    Returns:
    list: a list of relative file paths of all the images in the given directory
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                file_path = os.path.relpath(file_path, directory)
                file_path_split = file_path.split('/')
                # if file_path_split[1] != '138':
                #     continue
                file_path_updated = '/'.join(file_path_split[1:])
                file_paths.append(file_path_updated)
    return file_paths

def save_file_paths(file_paths: list, output_dir: str):
    """
    This function saves the list of relative file paths to a text file.

    Parameters:
    file_paths (list): a list of relative file paths of all the images in the given directory
    output_dir (str): the directory to save the output file
    """
    with open(os.path.join(output_dir, 'file_paths.txt'), 'w') as f:
        for file_path in file_paths:
            f.write(file_path + '\n')

def main():
    opt = args_parser()
    folder = opt.folder
    output_dir = opt.output_dir
    file_paths = retrieve_path(folder)
    for file_path in file_paths:
        print(file_path)
    save_file_paths(file_paths, output_dir)
        



if __name__ == '__main__':
    main()