import os
import argparse

from tqdm import tqdm


from lib.utils import common_utils




def args_parser():
    parser = argparse.ArgumentParser(description='Retrieve files from a directory')
    parser.add_argument('--file_folder', type=str, required=True,
                        help='The path to a folder including files that you want to retrieve.')
    parser.add_argument('--dataset', type=str, required=True, default=None,
                        help='The dataset type of the files')
    return parser.parse_args()

def main():
    args = args_parser()
    file_folder = args.file_folder
    dataset = args.dataset 
    files = common_utils.retrieve_label_files(file_folder, dataset)

    # split the files into 5 parts and write into a .txt file
    n = len(files) // 1
    remainder = len(files) % 1
    output_folder = os.path.join('./output/common', dataset)
    print(f'output_folder: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)
    for i in range(1):
        output_file = os.path.join(output_folder, f'label_{i}.txt')
        with open(output_file, 'w') as f:
            start = i*n
            end = (i+1)*n if i < 0 else (i+1)*n+remainder
            for file in tqdm(files[start:end], desc=f'Writing file_{i}', colour='green'):
                f.write(file + '\n')

if __name__ == '__main__':
    main()