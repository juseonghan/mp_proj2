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
    


def read_data_mp4():
    frames = []
    cap = cv.VideoCapture('../data/medusa.mp4')

    if cap.isOpened() == False:
        print("Error opening medusa file")
        return
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
    
    # get initial points to track from first frame
    curr_pts = cv.goodFeaturesToTrack(frames[0], mask=None, **feature_params)

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
        curr_pts = next_pts[is_found == 1].reshape(-1, 1, 2)

    # return final tracked points 
    return curr_pts  

if __name__ == "__main__": 
    main()