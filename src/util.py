import numpy as np
from NonlinearOptimizer import Solver
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
    print("F is somehow:", W.shape[0]/2)
    for i in range(int(W.shape[0]/2)):

        a_f = np.sum(W[i,:]) / W.shape[1]
        b_f = np.sum(W[i+int(W.shape[0]/2),:]) / W.shape[1]
        W_tilde[i,:] = W[i,:] - a_f
        W_tilde[i+int(W.shape[0]/2),:] = W_tilde[i+int(W.shape[0]/2),:] - b_f

    return W_tilde

def calculate_LT_RS(O1, sigma, O2):

    # following tomasi and kanade paper
    O1_prime = O1[:,0:3] 
    sigma_prime = np.diag(sigma[0:3])
    O2_prime = O2[:,0:3].T

    # it might be square root of all values of sigma, not matrix square root
    R_hat = O1_prime @ sqrtm(sigma_prime) # 2Fx3 * 3x3 = 2Fx3
    S_hat = sqrtm(sigma_prime) @ O2_prime # 3x3 * 3xP = 3xP 

    return R_hat, S_hat 

def calculate_Q(R_hat, S_hat):
    optim = Solver(R_hat, S_hat)
    return optim.run()


def get_shape_and_motion(R_hat, S_hat, Q):
    R = R_hat @ Q
    S = np.linalg.inv(Q) @ S_hat
    return R, S