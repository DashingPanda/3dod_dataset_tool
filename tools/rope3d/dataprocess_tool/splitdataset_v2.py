import argparse
import os




def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='The directory will be split')
    parser.add_argument('--num_parts', type=int, required=True, default=None,
                        help='The number of parts to split the directory into')

    args = parser.parse_args()
    return args

def split_dataset(data_dir: str, num_parts: int):
    """
    Split the dataset in the directory into N parts of directories.
    """
    files = os.listdir(data_dir)
    num_files = len(files)
    files_per_part = num_files // num_parts
    for i in range(num_parts):
        start = i * files_per_part
        end = (i + 1) * files_per_part
        if i == num_parts - 1:
            end = num_files
        part_dir = os.path.join(data_dir, f'part_{i}')
        os.makedirs(part_dir, exist_ok=True)
        for file in files[start:end]:
            file_path = os.path.join(data_dir, file)
            os.rename(file_path, os.path.join(part_dir, file))


def main():
    """
    The script is designed to split a dataset in a directory into N parts of directories,
    """
    data_dir = args_parser().data_dir
    num_parts = args_parser().num_parts
    split_dataset(data_dir, num_parts)
    

if __name__ == '__main__':
    main()