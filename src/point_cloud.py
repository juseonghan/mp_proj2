import open3d as o3d
import numpy as np

class SfM_Drawer:

    def __init__(self, data, type):
        self.data = data         
        self.type = type

    def get_camera_poses(self):

        # the poses are kf = if x jf
        F = int(self.data.shape[0]/2)
        camera_poses = np.zeros((F, 3))

        for f in range(F):
            kf = np.cross(self.data[f,:], self.data[f+F,:])
            camera_poses[f,:] = kf / np.linalg.norm(kf)

        return camera_poses

    def drawSfM(self):
        if self.type == 'S':
            P = int(self.data.shape[1])
            S_T = self.data.T
            pcd = o3d.geometry.PointCloud()
            pcd.colors = o3d.utility.Vector3dVector(0.1 + np.zeros((P, 3)))
            pcd.points = o3d.utility.Vector3dVector(S_T)
        else:
            camera_poses = self.get_camera_poses()
            F = int(self.data.shape[0]/2)     
            pcd = o3d.geometry.PointCloud()
            pcd.colors = o3d.utility.Vector3dVector(0.5 * np.ones((F, 3)))
            pcd.points = o3d.utility.Vector3dVector(camera_poses)

        o3d.visualization.draw_geometries([pcd])