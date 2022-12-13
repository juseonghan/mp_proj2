import numpy as np
from scipy.linalg import sqrtm 

def get_measurement_matrix(points):

    # points is a list with F elements, each element is a 2xP matrix. 
    # the measurement matrix is a 2FxP matrix.
    W = np.zeros((2*len(points), points[0].shape[1]))

    row_num = 0; 
    for points_in_one_frame in points:
        # points_in_one_frame is a 2xP matrix. 
        W[row_num,:] = points_in_one_frame[1,:]
        W[row_num + len(points),:] = points_in_one_frame[0,:]
        row_num = row_num + 1 

    return W
        

def get_registered_measurement_matrix(W):

    W_tilde = np.zeros_like(W)

    # loop over every row, i = [0,F)
    for i in range(W.shape[0]/2):

        a_f = np.sum(W[i,:]) / W.shape[1]
        b_f = np.sum(W[i+W.shape[0]/2,:]) / W.shape[1]
        W_tilde[i,:] = W[i,:] - a_f
        W_tilde[i+W.shape[0]/2,:] = W_tilde[i+W.shape[0]/2,:] - b_f

    return W_tilde

def calculate_LT_RS(O1, sigma, O2):
    
    # following tomasi and kanade paper
    O1_prime = O1[:,0:3] 
    sigma_prime = sigma[0:3,0:3]
    O2_prime = O2[0:3,:]
    R_hat = O1_prime @ sqrtm(sigma_prime)
    S_hat = sqrtm(sigma_prime) @ O2_prime

    return R_hat, S_hat 