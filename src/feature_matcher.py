import cv2 as cv
import numpy as np

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