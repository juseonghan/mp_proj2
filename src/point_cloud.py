import open3d as o3d
import numpy as np

class SfM_Drawer:

    def __init__(self, motion, shape):
        self.R = motion # 2F x 3
        self.S = shape # 3 x P 

    def get_camera_poses(self):

        # the poses are kf = if x jf
        camera_poses = []
        F = int(self.R.shape[0]/2)

        for f in range(F):
            kf = np.cross(self.R[f,:], self.R[f+F,:])
            camera_poses.append(kf / np.linalg.norm(kf))

        return camera_poses

    def drawSfM(self):
        camera_poses = self.get_camera_poses()

        # first draw point cloud
        S_T = self.S.T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(S_T)
        o3d.visualization.draw_geometries([pcd])