import cv2
import numpy as np




def draw_3dboxcenter(img: np.ndarray, centers: np.ndarray, color=(0, 0, 255)):
    """
    Parameters
    ----------
        img : np.ndarray

        centers : np.ndarray, (N, 2), N is the number of 3D bbox center points in image coordinate system

        color : tuple, default (0, 0, 255)
    
    Returns
    -------
        img : np.ndarray

    """

    # draw the 3D bbox center on the image
    for i, center in enumerate(centers):
        cv2.circle(img, (int(center[0]), int(center[1])), 5, color, -1)
    return img


def draw_3dboxcenter_with_distence(img: np.ndarray, centers: np.ndarray, centers_lid: np.ndarray, color=(0, 0, 255)):
    """
    Parameters
    ----------
        img : np.ndarray

        centers : np.ndarray, (N, 2), N is the number of 3D bbox center points in image coordinate system

        # centers_lid : np.ndarray, (N, 2), N is the number of 3D bbox center points in lidar coordinate system

        color : tuple, default (0, 0, 255)
    
    Returns
    -------
        img : np.ndarray

    """

    # draw the 3D bbox center on the image
    for i, center in enumerate(centers):
        x_lid = centers_lid[i][0]
        y_lid = centers_lid[i][1]
        distence = np.sqrt(x_lid**2 + y_lid**2)
        cv2.putText(img, str(round(distence, 2)), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.circle(img, (int(center[0] + 5), int(center[1] + 5)), 5, color, -1)
    return img


def draw_3dbox(img: np.ndarray, corners: np.ndarray, color=(0, 0, 255)):
    """
    Draw 3D bbox on the image



    Parameters
    ----------
        img : np.ndarray

        corners : np.ndarray, (N, 8, 2), N is the number of 3D bbox corners in image coordinate system

        color : tuple, default (0, 0, 255)
    
    Returns
    -------
        img : np.ndarray

    Notes
    -----
    The corners are returned in the following order for each box:
    
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2

    """
    # conection relationship of the 3D bbox vertex
    connection = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    # draw the 3D bbox on the image
    for i, corner in enumerate(corners):
        for j in range(len(connection)):
            cv2.line(img, (int(corner[connection[j][0]][0]), int(corner[connection[j][0]][1])), (int(corner[connection[j][1]][0]), int(corner[connection[j][1]][1])), color, 2)
    return img


def draw_3dbox_Zcam(img: np.ndarray, corners: np.ndarray, Zcam: np.ndarray, color=(0, 0, 255)):
    """
    Draw 3D bbox on the image



    Parameters
    ----------
        img : np.ndarray

        corners : np.ndarray, (N, 8, 2), N is the number of 3D bbox corners in image coordinate system

        color : tuple, default (0, 0, 255)

        Zcam : np.ndarray, (N, 1), N is the number of 3D bbox center points in camera coordinate system
            the Z value of the 3D bbox center points in camera coordinate system
    
    Returns
    -------
        img : np.ndarray

    Notes
    -----
    The corners are returned in the following order for each box:
    
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2

    """
    # conection relationship of the 3D bbox vertex
    connection = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    # draw the 3D bbox on the image
    for i, corner in enumerate(corners):
        cv2.txt = cv2.putText(img, str(round(Zcam[i], 2)), (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
        for j in range(len(connection)):
            cv2.line(img, (int(corner[connection[j][0]][0]), int(corner[connection[j][0]][1])), (int(corner[connection[j][1]][0]), int(corner[connection[j][1]][1])), color, 2)
    return img


def draw_2dbox(img: np.ndarray, corners: np.ndarray, color=(0, 0, 255)):
    """
    Draw 3D bbox on the image



    Parameters
    ----------
    img : np.ndarray

    corners : np.ndarray, (N, 4, 2), N is the number of 2D bbox corners in image coordinate system
        An array contains the 4 corner points of the 2D bounding boxes for each 3D bounding box. 
    The order of the corners is: [top-left, top-right, bottom-right, bottom-left]

    color : tuple, default (0, 0, 255)
    
    Returns
    -------
        img : np.ndarray

    Notes
    -----
    The corners are returned in the following order for each box:
    
        0 ---------- 1
        |            |
        |            |
        |            |
        |            |
        |            |
        3 ---------- 2

    """
    # conection relationship of the 2D bbox vertex
    connection = [[0, 1], [1, 2], [2, 3], [3, 0]]
    # draw the 3D bbox on the image
    for i, corner in enumerate(corners):
        for j in range(len(connection)):
            cv2.line(img, (int(corner[connection[j][0]][0]), int(corner[connection[j][0]][1])), (int(corner[connection[j][1]][0]), int(corner[connection[j][1]][1])), color, 2)
    return img