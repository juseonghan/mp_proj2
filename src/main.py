import argparse

from util import *
from reader import *
from feature_matcher import match_feature_points
from point_cloud import SfM_Drawer

def main():

    # data read
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="castle or medusa", type=str)
    args = parser.parse_args()

    if (args.data == "castle"):
        frames = read_data_jpgs()
    elif (args.data == "medusa"):
        frames = read_data_mp4()
    else:
        print("please type medusa or castle after --data")
        return

    # start! 
    np.set_printoptions(precision=3, suppress=True)

    print('--------- matching points ---------')
    print('No. of frames:', len(frames))

    points = match_feature_points(frames, args.data)
    print('Matching:', points[0].shape[1], 'points')

    print('--------- calculating measurement matrix ---------')
    W = get_measurement_matrix(points)
    print('shape of W is:', W.shape)
    W_tilde = get_registered_measurement_matrix(W)
    print('shape of W_tilde is:', W_tilde.shape)

    print('--------- calculating affine motion and shape matrices ---------')
    O1, sigma, O2 = np.linalg.svd(W_tilde)
    R_hat, S_hat = calculate_LT_RS(O1, sigma, O2)
    print('R^hat:')
    print(R_hat)
    print("S^hat:")
    print(S_hat)

    print('--------- affine correction optimization ---------')
    Q = calculate_Q(R_hat, S_hat)
    print("Affine ambiguity correction matrix Q:")
    print(Q)

    print('--------- calculate shape and motion ---------')
    R, S = get_shape_and_motion(R_hat, S_hat, Q)
    print("Motion matrix R:")
    print(R)
    print("Shape matrix S:")
    print(S)

    print('--------- results saved to ../results/ ---------')
    np.savetxt('../results/R.txt', R, fmt='%.4f')
    np.savetxt('../results/S.txt', S, fmt='%.4f')

    drawer = SfM_Drawer(R, S)
    drawer.drawSfM()


if __name__ == "__main__": 
    main()