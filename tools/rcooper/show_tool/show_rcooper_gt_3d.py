import numpy as np
import matplotlib.pyplot as plt




def fit_plane_least_squares(points):
    # points = np.array(points)
    # 构建设计矩阵 A
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    # 目标变量 B
    B = points[:, 2]
    # 使用最小二乘法拟合平面
    coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return coeff



gt_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/doc/gt_tensor.npy'

gt = np.load(gt_file)
print(gt.shape)
print(len(gt.shape))
# print(gt)

# exit()


# ====== fit a plane to the ground truth ======
# pick the points in the bottom of the 3D bbox, first 4 points of each matrix
bottom_points = gt[:, :4]
print(bottom_points.shape)
print(bottom_points)
bottom_points = bottom_points.reshape(-1, 3)
print(bottom_points.shape)
print(bottom_points)


ground_plane_coeff = fit_plane_least_squares(bottom_points)
print(ground_plane_coeff)
# exit()

for i, matrix in enumerate(gt):
    print(f"Matrix {i}:\n{matrix}")

# ====== show 3D box bottom in a 3D coordinate system ======
# create a figure and axis
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

# Plane coefficients (a, b, c)
a, b, d = ground_plane_coeff
c = -1  # z的系数为-1，因此不需要显示

# Create a grid of x, y values
x = np.linspace(-150, 150, 100)
y = np.linspace(-125, 75, 100)
X, Y = np.meshgrid(x, y)

# Calculate Z values based on the plane equation ax + by + cz = 0 => z = -(a*x + b*y) / c
Z = -(a * X + b * Y + d) / c

# Plotting
# ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5, edgecolor='w')



# ====== show points in a 3D coordinate system ======


numpy_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/doc/center.npy'
centers = np.load(numpy_file)
numpy_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/doc/extent.npy'
extents = np.load(numpy_file)
numpy_file = '/Users/panda/Documents/Factory/JupyterNotebook/deepglint/rcooper/doc/Rotation.npy'
Rs = np.load(numpy_file)

print(f'shape of centers: {centers.shape}')
print(f'shape of extents: {extents.shape}')
print(f'shape of Rs: {Rs.shape}')

# exit()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

centers[:, 0] = -centers[:, 0]

for center, extentin in zip(centers, extents):
    print(f'center: {center}')
    ax.scatter(center[0], center[1], center[2], color='blue', marker='o', alpha=0.7)


# for center, box_obj in zip(centers, gt):
#     for connect in connection:
#         ax.plot([box_obj[connect[0]][0], box_obj[connect[1]][0]],
#         [box_obj[connect[0]][1], box_obj[connect[1]][1]],
#         [box_obj[connect[0]][2], box_obj[connect[1]][2]],
#         color='red', linewidth=1, markersize=1, linestyle='--', marker='o', alpha=0.7)
#     ax.scatter(center[0], center[1], center[2], color='blue', marker='o', alpha=0.7)

plt.show()