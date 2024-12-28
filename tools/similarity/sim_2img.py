import cv2
import argparse
import numpy as np



def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True, help='Path to image 1')
    parser.add_argument('--img2', type=str, required=True, help='Path to image 2')
    return parser.parse_args()


def calculate_mse(gray_image1, gray_image2):
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((gray_image1 - gray_image2) ** 2)
    # # Calculate the Root Mean Squared Error (RMSE)
    # # rmse = np.sqrt(np.mean((gray_image1 - gray_image2) ** 2))
    # mse = np.sum((gray_image1 - gray_image2) ** 2)
    return mse

def main():
    args = args_parser()
    img1 = cv2.imread(args.img1, -1) / 256
    img2 = cv2.imread(args.img2, -1) / 256
    # img2 = img2[:,:,0]
    img = img1 - img2
    print(img2.shape)
    for v in range(0, img1.shape[0]):
        for u in range(0, img1.shape[1]):
            if img1[v, u] != 0:
                print('-' * 10)
                print(img1[v, u])
                print(img2[v, u])
                print('-' * 5)
    # if the value is not zero, it is set to 255
    # img[img != 0] = 255
    # save the difference image
    cv2.imwrite('/mnt/data_cfl/Projects/3dod-dataset-tools/output/tmp/diff_img.png', img)

    # gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mse = calculate_mse(img1, img2)
    print(f'MSE: {mse}')


    


if __name__ == '__main__':
    main()