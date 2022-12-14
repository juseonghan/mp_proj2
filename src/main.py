import argparse

from util import *
from reader import *
from feature_matcher import match_feature_points

def main():

    # data read
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="castle or medusa", type=str)
    parser.add_argument("--points", help="number of feature points to track")
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

    points = match_feature_points(frames, args.data, args.points)
    print('Matching:', points[0].shape[1], 'points')

    print('--------- calculating measurement matrix ---------')
    W = get_measurement_matrix(points)
    print('shape of W is:', W.shape)
    W_tilde = get_registered_measurement_matrix(W)
    print('shape of W_tilde is:', W_tilde.shape)

    print('--------- calculating affine motion and shape matrices ---------')
    R_hat, S_hat = calculate_RS(W_tilde)
    print('R^hat shape:', R_hat.shape)
    # print(R_hat)
    print("S^hat shape:", S_hat.shape)
    # print(S_hat)

    print('--------- affine correction optimization ---------')
    Q = calculate_Q(R_hat, S_hat)
    print("Affine ambiguity correction matrix Q:")
    print(Q)

    print('--------- calculate shape and motion ---------')
    R, S = get_shape_and_motion(R_hat, S_hat, Q)
    print("Motion matrix R shape:", R.shape)
    # print(R)
    print("Shape matrix S shape:", S.shape)
    # print(S)

    print('--------- results saved to ../results/ ---------')

    text_nameR = "R_" + args.data + ".txt"
    text_nameS = "S_" + args.data + ".txt"
    np.savetxt('../results/' + text_nameR, R, fmt='%.4f')
    np.savetxt('../results/' + text_nameS, S, fmt='%.4f')

    print('mean-squared error between W_tilde and R*S:',calculate_error(W_tilde, R @ S) )


if __name__ == "__main__": 
    main()