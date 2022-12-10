import cv2 as cv
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
        

    return W_tilde


def read_data_mp4():
    frames = []
    cap = cv.VideoCapture('../data/medusa.mp4')

    if cap.isOpened() == False:
        print("Error opening medusa file")
        return
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            break
    cap.release()
    
    return frames

def read_data_jpgs():
    frames = []
    files = glob.glob("../data/castle_data/*.jpg")
    for img_path in files:
        img = cv.imread(img_path)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        frames.append(img)
    return frames


def track_points(frames):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 300,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # params for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    result = []
    # get initial points to track from first frame
    frame0_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
    curr_pts = cv.goodFeaturesToTrack(frame0_gray,  mask=None, **feature_params)
    curr_pts_resized = np.resize(curr_pts, (curr_pts.shape[0], 2))
    result.append(curr_pts_resized)

    # # checked that it's working - possiibly play around with parameters
    # plt.imshow(frames[0])
    # plt.scatter(curr_pts.squeeze()[:,0], curr_pts.squeeze()[:,1],s=1,c='orange')
    # plt.savefig('../extra/tracked_pts_medusa/frame0.png')
    # plt.clf()

    # loop through frames and track points with KLT
    for i in range(1, len(frames)):
        # perform lucas kanade optical flow
        next_pts, is_found, err = cv.calcOpticalFlowPyrLK(frames[i-1], frames[i], curr_pts, None, **lk_params)
        # keep tracked points that were found and update curr_pts
        # print(next_pts.shape)
        curr_pts = next_pts.reshape(-1, 1, 2)
        curr_pts_resized = np.resize(curr_pts, (curr_pts.shape[0], 2))
        result.append(curr_pts_resized)

    # return final tracked points 
    return result  

if __name__ == "__main__": 
    main()