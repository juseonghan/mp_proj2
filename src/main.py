import argparse

from util import *
from reader import *
from feature_matcher import track_points

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
    points = track_points(frames)
    W = get_measurement_matrix(points)
    W_tilde = get_registered_measurement_matrix(W)

    # pts0 = points[0]
    # pts_last = points[-1] 
    # img_test1 = frames[0]
    # img_test2 = frames[-1]
    # for i in range(pts0.shape[0]):
    #     img_test1 = cv.circle(img_test1, (int(pts0[i][0]), int(pts0[i][1])), radius=1, color=(0,0,255), thickness=1)
    #     img_test2 = cv.circle(img_test2, (int(pts_last[i][0]), int(pts_last[i][1])), radius=1, color=(0,0,255), thickness=1)
    # cv.imshow('img', np.hstack((img_test1, img_test2)))
    # cv.waitKey(0)


if __name__ == "__main__": 
    main()