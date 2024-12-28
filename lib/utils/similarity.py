import cv2
from tqdm import tqdm
import sys
import os
import numpy as np


# Add your custom library paths
sys.path.append('/gemini/data-1/Projects/dairv2xi-dataset-tools')


def calculate_mse(gray_image1, gray_image2):
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((gray_image1 - gray_image2) ** 2)
    return mse


def retrieve_overlap_rope3d_images(dir_path, dataset1_txt, dataset2_txt):
    dataset1_txt = os.path.join(dir_path, dataset1_txt)
    dataset2_txt = os.path.join(dir_path, dataset2_txt)

    dataset1_image_names = open(dataset1_txt, 'r').readlines()
    dataset1_image_names = [os.path.join(dir_path, x.strip()+'.jpg') for x in dataset1_image_names]
    dataset2_image_names = open(dataset2_txt, 'r').readlines()
    dataset2_image_names = [os.path.join(dir_path, x.strip()+'.jpg') for x in dataset2_image_names]

    grayimg_dataset1_dict = dict()
    grayimg_dataset2_dict = dict()
    index = 0
    # create a txt file to record the overlap images
    # f = open(log_file, 'w')
    for image_name in tqdm(dataset1_image_names, desc='Loading dataset1 images', colour='green'):
        # if index == 10:
        #     break
        img = cv2.imread(image_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayimg_dataset1_dict[image_name] = gray_img
        index += 1
    for image_name in tqdm(dataset2_image_names, desc='Loading dataset2 images', colour='green'):
        # if index == 10:
        #     break
        img = cv2.imread(image_name)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayimg_dataset2_dict[image_name] = gray_img
        index += 1
        
    for img_name_dataset2 in tqdm(grayimg_dataset2_dict, desc='Calculating MSE', colour='green'): 
        gray_img_dataset2 = grayimg_dataset2_dict[img_name_dataset2]
        mase_min = sys.float_info.max
        data1_image_name_flag = None
        data2_image_name_flag = None
        for img_name_dataset1 in grayimg_dataset1_dict:
            gray_img_dataset1 = grayimg_dataset1_dict[img_name_dataset1]
            mase = calculate_mse(gray_img_dataset2, gray_img_dataset1) 
            if mase < mase_min:
                mase_min = mase
                data1_image_name_flag = img_name_dataset1
                data2_image_name_flag = img_name_dataset2
            if mase == 0.0:
                break
        # message = f'mase_min: {mase_min}\n dirv2xi: {image_name_flag}\n rope3d: {img_name_rope3d_flag} \n'
        message = f'mase_min: {mase_min}\n dataset1: {data1_image_name_flag}\n datas2: {data2_image_name_flag} \n'
        print(message)
        print('-' * 50 + '\n')
        # f.writelines(message)
        # f.writelines('-' * 50 + '\n')
    # f.close()
