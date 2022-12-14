import open3d as o3d
import numpy as np

class SfM_Drawer:

    def __init__(self, motion, shape):
        self.R = motion # 2F x 3
        self.S = shape # 3 x P 
        

    def get_camera_poses(self):

        # the poses are kf = if x jf
        F = int(self.R.shape[0]/2)
        camera_poses = np.zeros((F, 3))

        for f in range(F):
            kf = np.cross(self.R[f,:], self.R[f+F,:])
            camera_poses[f,:] = (kf / np.linalg.norm(kf))

        return camera_poses

    def drawSfM(self):
        camera_poses = self.get_camera_poses()
        P = int(self.S.shape[1])
        F = int(self.R.shape[0]/2)

        # first draw point cloud
        S_T = self.S.T
        pcd = o3d.geometry.PointCloud()
        pcd.colors = np.zeros((P, 3))
        pcd.points = o3d.utility.Vector3dVector(S_T)

        poses = o3d.geometry.PointCloud()
        poses.colors = 0.5 * np.ones((F, 3))
        pcd.points = o3d.utility.Vector3dVector(camera_poses)

        o3d.visualization.draw_geometries([pcd, poses])

