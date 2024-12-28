import argparse
import json



def args_parser():
    parser = argparse.ArgumentParser(description='Generate the data json file')
    parser.add_argument('--file_paths', type=str, required=True,
                        help='The path to a text file containing the relative file paths of the images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='the directory to save the output json file')
    parser.add_argument('--label', type=str, required=True,
                        help='the label of the images')
    return parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()
