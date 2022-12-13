import numpy as np

class Solver:

    def __init__(self, R_hat, S_hat):
        self.R = R_hat # 2F x 3 matrix
        self.S = S_hat # 3 x P matrix

    def run(self):
        F = int(self.R.shape[0]/2)
        Is = self.R[:F,:] # Fx3
        Js = self.R[F:,:] # Fx3
        
        G = np.zeros((3*F, 6)) # 3F x 6 matrix

        for f in range(F):
            G[f,:] = self.get_mat(Is[f,:], Is[f,:])
            G[F+f,:] = self.get_mat(Js[f,:], Js[f,:])
            G[2*F + f,:] = self.get_mat(Is[f,:], Js[f,:])

        # c: 3F x 1 
        c = np.vstack(( np.ones((2*F,1)), np.zeros((F,1)) ))
        
        # u: 3F x 3F
        # s: 3F x 6 with 6 singular values
        # v: 6 x 6
        u, S, v = np.linalg.svd(G)

        lhat = u.T @ c  # 3Fx3F * 3Fx1 = 3Fx1
        lhat = lhat.flatten()

        y = np.array([lhat[0]/S[0], lhat[1]/S[1], lhat[2]/S[2], lhat[3]/S[3], lhat[4]/S[4], lhat[5]/S[5]])
        y = y.T # 6 x 1 matrix

        l = v @ y # 6 x 1 matrix
        L = np.array([ [l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]] ])

        # need to check if all the eigenvalues of L are positive
        evals, _ = np.linalg.eig(L)

        if all(i > 0 for i in evals):
            return np.linalg.cholesky(L)

        l2 = np.linalg.lstsq(G, c, rcond=None)[0]
        l2 = l2.flatten()

        L2 = np.array([ [l2[0], l2[1], l2[2]], [l2[1], l2[3], l2[4]], [l2[2], l2[4], l2[5]] ])

        return np.linalg.cholesky(L2)

    def get_mat(self, a,b):
        res = np.array( [a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[0]*b[2]+a[2]*b[0], a[1]*b[1], a[1]*b[2]+a[2]*b[1], a[2]*b[2] ])
        return res