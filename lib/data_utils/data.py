


class Data:
    """ class Data """
    def __init__(self, obj_type: str=None, truncation=None, occlusion=None, \
                 x1=None, y1=None, x2=None, y2=None, w=None, h=None, l=None, \
                 X=None, Y=None, Z=None, yaw=None, score=None, detect_id=None, scene_id: str=None):
        """
        Init object data

        Parameters
        ----------
        obj_type : str
            object type
        truncation : float
            truncation level
        occlusion : int
            occlusion level
        x1 : float
            x1 coordinate of the 2D bounding box top left corner
        y1 : float
            y1 coordinate of the 2D bounding box top left corner
        x2 : float
            x2 coordinate of the 2D bounding box bottom right corner
        y2 : float
            y2 coordinate of the 2D bounding box bottom right corner
        w : float
            width of the 3D bounding box
        h : float
            height of the 3D bounding box
        l : float
            length of the 3D bounding box
        X : float
            x coordinate of the 3D bounding box center
        Y : float
            y coordinate of the 3D bounding box center
        Z : float
            z coordinate of the 3D bounding box center
        yaw : float
            yaw angle of the object
        score : float
            detection score
        detect_id : int
            detection id
        scene_id : str
            scene id

        Note
        ----
        The 3D bounding box center (X, Y, Z) is defined within the camera, LiDAR or other sensor coordinate system.

        - For rope3d dataset, the 3D bounding box center is defined in the camera coordinate system.
        - For rcooper dataset, the 3D bounding box center is defined in the LiDAR coordinate system.
        """
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.yaw = yaw
        self.score = score
        self.detect_id = detect_id


class Scene:
    def __init__(self, scene_id: str, objects: list):
        pass