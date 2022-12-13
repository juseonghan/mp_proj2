import argparse
import matplotlib.pyplot as plt

from util import *
from reader import *
from feature_matcher import match_feature_points

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
    print('--------- matching points ---------')
    print('No. of frames:', len(frames))

    points = match_feature_points(frames)

    print('--------- calculating measurement matrix ---------')
    W = get_measurement_matrix(points)
    W_tilde = get_registered_measurement_matrix(W)

    print('--------- calculating affine motion and shape matrices ---------')
    O1, sigma, O2 = np.linalg.svd(W_tilde)
    R_hat, S_hat = calculate_LT_RS(O1, sigma, O2)

    print('--------- affine correction optimization ---------')
    Q = calculate_Q(R_hat, S_hat)

    print('--------- calculate shape and motion ---------')
    R, S = get_shape_and_motion(R_hat, S_hat, Q)


if __name__ == "__main__": 
    main()