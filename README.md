# 3dod-dataset-tools


# Introduction
This repo contains tools for working with the 3D Object Detection Dataset. 

<img src="https://gitlab.deepglint.com/fenglianchen/3dod-dataset-tools/-/blob/main/examples/rcooper/1693909042.081970.jpg" width="500">
<img src="https://gitlab.deepglint.com/fenglianchen/3dod-dataset-tools/examples/rcooper/1693909063.315255.jpg" width="500">



The dataset involves:
-  [DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative
3D Object Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_DAIR-V2X_A_Large-Scale_Dataset_for_Vehicle-Infrastructure_Cooperative_3D_Object_Detection_CVPR_2022_paper.pdf)
-  [RCooper: A Real-world Large-scale Dataset for
Roadside Cooperative Perception](https://openaccess.thecvf.com/content/CVPR2024/papers/Hao_RCooper_A_Real-world_Large-scale_Dataset_for_Roadside_Cooperative_Perception_CVPR_2024_paper.pdf)
-  [Rope3D: The Roadside Perception Dataset for Autonomous Driving
and Monocular 3D Object Detection Task](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Rope3D_The_Roadside_Perception_Dataset_for_Autonomous_Driving_and_Monocular_CVPR_2022_paper.pdf)

The tools include:
- Visualization: This includes tools for visualizing the dataset and the results of the models. Specifically, we have tools for visualizing the 3D bounding boxes, point clouds, and 2D bounding boxes in images, plt 3d scatter plots, open3d visualizations, and more. 
- Data Processing: This includes tools for processing data to prepare for training, visualizing or other data processing 

# Usage
There are .sh scripts to launch specific Python tool files corresponding to each task (e.g., visualizing bounding boxes or other tools). The .sh script also defines and passes the task-specific parameters to the corresponding Python file.
~~~
bash scripts/rcooper/show_rcooper_gt_image.sh
~~~

# Documentation
You can find all the tool Python file in the "tools" folder. Each dataset is supported by its own spcific tools, tailored to handle various task effectively and efficiently.

## common

## dairv2x

## rope3d

## similarity