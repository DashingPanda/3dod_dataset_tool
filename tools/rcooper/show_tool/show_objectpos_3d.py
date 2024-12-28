# Read a label JSON file
import json

json_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/dataset/label/136-137-138-139/136/seq-0/1693908913.283240.json'
label_list = json.load(open(json_file, 'r'))
print(type(label_list))

# Visualize the object position in a 3D coordinate
import matplotlib.pyplot as plt

# ====== show 3D box bottom in a 3D coordinate system ======
# create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

z_max = float('-inf')
z_min = float('inf')

for label in label_list:
    type_obj = label['type']
    h = label['3d_dimensions']['h']
    # h = h
    w = label['3d_dimensions']['w']
    # h = h
    l = label['3d_dimensions']['l']
    # h = h
    pos3d = label['3d_location']
    # print(pos3d)
    # Plot the points
    x = pos3d['x']
    y = pos3d['y']
    z = pos3d['z']
    ax.scatter(x, y, z, color='red', marker='o')
    # ax.text(x, y, z, h, color='black', fontsize=6)

    if z > z_max:
        z_max = z
    if z < z_min:
        z_min = z
print('coordinate system1')
print(f'z_max: {z_max}, z_min: {z_min}')

# plt.show()


# Read a coordinate transformation matrix from a file
import numpy as np

# lidar to camera
json_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/dataset/calib/lidar2cam/136.json'
lid2cam_calib = json.load(open(json_file, 'r'))
lid2cam_calib = lid2cam_calib['cam_0']
lid2cam_extri = lid2cam_calib['extrinsic']
lid2cam_intri = lid2cam_calib['intrinsic']
lid2cam_extri = np.array(lid2cam_extri)
lid2cam_intri = np.array(lid2cam_intri)
# camera to lidar
cam2lid_extri = np.linalg.inv(lid2cam_extri)
# print(f'lid2cam_extri:\n{lid2cam_extri}\n')
# print(f'cam2lid_extri:\n{cam2lid_extri}\n')

# lidar to world
json_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/dataset/calib/lidar2world/136.json'
lid2world_calib = json.load(open(json_file, 'r'))
lid2world_calib_R = lid2world_calib['rotation']
lid2world_calib_R = np.array(lid2world_calib_R)
lid2world_calib_T = lid2world_calib['translation']
lid2world_calib_T = np.array(lid2world_calib_T)
lid2world_extri = np.hstack((lid2world_calib_R, lid2world_calib_T.reshape(3,1)))
row = np.array([0., 0., 0., 1.])
lid2world_extri = np.vstack((lid2world_extri, row))
# world to lidar
world2lid_extri = np.linalg.inv(lid2world_extri)
# print(f'lid2world_extri:\n{lid2world_extri}\n')
# print(f'world2lid_extri:\n{world2lid_extri}\n')


# trans_matrix = np.array([[ 1.0000e+00, -5.5511e-17, -5.2042e-18,  4.5475e-13],
#         [-5.5511e-17,  1.0000e+00,  0.0000e+00,  4.5475e-13],
#         [ 1.7347e-18,  0.0000e+00,  1.0000e+00,  4.2633e-14],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

# ====== show 3D box bottom in a 3D coordinate system ======
# create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

z_max = float('-inf')
z_min = float('inf')

for index, label in enumerate(label_list):
    type_obj = label['type']
    h = label['3d_dimensions']['h']
    pos3d = [label['3d_location']['x'], label['3d_location']['y'], label['3d_location']['z'], 1.]
    pos3d = np.array(pos3d)
    # pos3d_cam = lid2world_extri @ pos3d
    pos3d_cam = lid2cam_extri @ pos3d
    # pos3d_cam = cam2lid_extri @ pos3d
    # pos3d_cam = world2lid_extri @ pos3d
    # print(pos3d)

    # Plot the points
    x, y, z = pos3d_cam[:3]
    # z -= h
    ax.scatter(x, y, z, color='red', marker='o')
    
    # h = int(h)
    # ax.text(x, y, z, h, color='black', fontsize=6)

    if z > z_max:
        z_max = z
    if z < z_min:
        z_min = z
print('coordinate system1')
print(f'z_max: {z_max}, z_min: {z_min}')    


# ====== transform the 3D box in lidar coordinate system into camera coordinate system ======
# ====== show 3D box bottom in a 3D coordinate system ======

# readd the gt 3D box
gt_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/doc/gt_tensor.npy'
gt = np.load(gt_file)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# conection relationship of the 3D bbox vertex
connection = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
# connection = [[0, 1], [1, 2], [2, 3], [3, 0]]

for i, box_obj in enumerate(gt):
    for connect in connection:
        ax.plot([box_obj[connect[0]][0], box_obj[connect[1]][0]],
        [box_obj[connect[0]][1], box_obj[connect[1]][1]],
        [box_obj[connect[0]][2], box_obj[connect[1]][2]],
        color='red', linewidth=1, markersize=1, linestyle='--', marker='o', alpha=0.7)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


for i, box_obj in enumerate(gt):

    box_obj = np.hstack((box_obj, np.ones((8,1))))
    box_obj_cam = (lid2cam_extri @ box_obj.T).T
    for connect in connection:
        ax.plot([box_obj_cam[connect[0]][0], box_obj_cam[connect[1]][0]],
        [box_obj_cam[connect[0]][1], box_obj_cam[connect[1]][1]],
        [box_obj_cam[connect[0]][2], box_obj_cam[connect[1]][2]],
        color='red', linewidth=1, markersize=1, linestyle='--', marker='o', alpha=0.7)




print(gt.shape)
print(lid2cam_extri.shape)

bottom_points = gt[:, :4]
bottom_points = bottom_points.reshape(-1, 3)
bottom_points = np.hstack((bottom_points, np.ones((bottom_points.shape[0],1))))
bottom_points_cam = (lid2cam_extri @ bottom_points.T).T

def fit_plane_least_squares(points):
    # points = np.array(points)
    # 构建设计矩阵 A
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    # 目标变量 B
    B = points[:, 2]
    # 使用最小二乘法拟合平面
    coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return coeff

ground_plane_coeff = fit_plane_least_squares(bottom_points_cam)

print(f'ground_plane_coeff: {ground_plane_coeff}')

# Plane coefficients (a, b, c)
a, b, d = ground_plane_coeff
c = -1  # z的系数为-1，因此不需要显示

# Create a grid of x, y values
x = np.linspace(-130, 130, 50)
y = np.linspace(-60, 20, 50)
X, Y = np.meshgrid(x, y)

# Calculate Z values based on the plane equation ax + by + cz = 0 => z = -(a*x + b*y) / c
Z = -(a * X + b * Y + d) / c

# Plotting
# ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5, edgecolor='w')

plt.show()
