import argparse

import sys
sys.path.append('/mnt/data_cfl/Projects/3dod-dataset-tools')


from lib.utils import similarity as sim





def args_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='The directory of the rope3d dataset')
    parser.add_argument('--train_image_txt', type=str, required=True,
                        help='The Txt file path of the image list of the rope3d dataset for training')
    parser.add_argument('--val_image_txt', type=str, required=True,
                        help='The Txt file path of the image list of the rope3d dataset for validation')
    # parser.add_argument('--log_file', type=str, required=True,
    #                     help='the path of a log.txt file to store the comparison results')
    opt = parser.parse_args()
    return opt


def main():
    # ================= parse arguments =================
    opt = args_parser()
    dataset_dir = opt.dataset_dir
    train_image_txt = opt.train_image_txt
    val_image_txt = opt.val_image_txt

    print('================= parse arguments =================')
    print('dataset_dir:', dataset_dir)
    print('train_image_txt:', train_image_txt)
    print('val_image_txt:', val_image_txt)

    # ================= retrieve overlap rope3d images =================
    print('================= retrieve overlap rope3d images =================')
    sim.retrieve_overlap_rope3d_images(dataset_dir, train_image_txt, val_image_txt)


if __name__ == '__main__':
    main()