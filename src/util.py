import numpy as np
import cv2 as cv 

def get_measurement_matrix(points):
    # points is a list with F elements, each element is a Px2 matrix. 
    # the measurement matrix is a 2FxP matrix.
    W = np.zeros((2*len(points), points[0].shape[0]))

    row_num = 0; 
    for points_in_one_frame in points:
        #points_in_one_frame is a 300 x 2 (Px2) matrix. 
        W[row_num,:] = points_in_one_frame[:,1].T
        W[row_num + len(points),:] = points_in_one_frame[:,0].T
        row_num = row_num + 1 

    return W
        

def get_registered_measurement_matrix(W):
    W_tilde = np.zeros_like(W)
    for i in range(W.shape[0]/2):
        # i = [0, F)
        a_f = np.sum(W[i,:]) / W.shape[1]
        b_f = np.sum(W[i+W.shape[0]/2,:]) / W.shape[1]
        W_tilde[i,:] = W[i,:] - a_f
        W_tilde[i+W.shape[0]/2,:] - b_f

    return W_tilde

