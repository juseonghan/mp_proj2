import argparse
import matplotlib.pyplot as plt

from util import *
from reader import *
from feature_matcher import match_feature_points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="castle or medusa", type=str)
    args = parser.parse_args()

    # read in data accordingly 
    if (args.data == "castle"):
        frames = read_data_jpgs()
    elif (args.data == "medusa"):
        frames = read_data_mp4()
    else:
        print("please type medusa or castle after --data")
        return

    # track points
    points = match_feature_points(frames)
    W = get_measurement_matrix(points)
    W_tilde = get_registered_measurement_matrix(W)

    O1, sigma, O2 = np.linalg.svd(W_tilde)
    R_hat, S_hat = calculate_LT_RS(O1, sigma, O2)


if __name__ == "__main__": 
    main()