import numpy as np
from solver import Solver

def get_measurement_matrix(points):

    # points is a list with F elements, each element is a 2xP matrix. 
    # the measurement matrix is a 2FxP matrix.
    F = len(points)
    P = points[0].shape[1]
    U = np.zeros((F, P))   
    V = np.zeros((F, P))

    row_num = 0; 
    for points_in_one_frame in points:
        U[row_num,:] = points_in_one_frame[1,:]
        V[row_num,:] = points_in_one_frame[0,:]
        row_num = row_num + 1 

    W = np.vstack((U,V))
    print("W shape:", W.shape)
    return W
        
def get_registered_measurement_matrix(W):

    W_tilde = np.zeros_like(W)
    F = int(W.shape[0]/2)
    P = int(W.shape[1])

    U = W[:F,:] 
    V = W[F:,:]

    # loop over every row, i = [0,F)

    for i in range(F):
        
        a_f = np.sum(U[i,:]) / P
        b_f = np.sum(V[i,:]) / P
        W_tilde[i,:] = U[i,:] - a_f
        W_tilde[i+F,:] = V[i,:] - b_f

    print("W_tilde rank:", np.linalg.matrix_rank(W_tilde))
    return W_tilde

def calculate_RS(W_tilde):

    O1, sigma, O2 = np.linalg.svd(W_tilde)
    # O1: 2F x P 
    # Sigma: P x P
    # O2: P x P
    # following tomasi and kanade paper
    O1_prime = O1[:,0:3] # 2F x 3
    sigma_prime = np.diag(sigma[0:3]) # 3x3 
    O2_prime = O2[0:3,:] # 3 x P

    # it might be square root of all values of sigma, not matrix square root
    R_hat = O1_prime @ np.sqrt(sigma_prime) # 2Fx3 * 3x3 = 2Fx3
    S_hat = np.sqrt(sigma_prime) @ O2_prime # 3x3 * 3xP = 3xP 

    return R_hat, S_hat 

def calculate_Q(R_hat, S_hat):
    optim = Solver(R_hat, S_hat)
    return optim.run()


def get_shape_and_motion(R_hat, S_hat, Q):
    R = R_hat @ Q
    S = np.linalg.inv(Q) @ S_hat
    return R, S

def calculate_error(W, W_tilde):
    diff = W - W_tilde
    diff_sq = np.square(diff)
    return np.sum(diff_sq)