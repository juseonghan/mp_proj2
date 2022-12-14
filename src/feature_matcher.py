import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

def match_feature_points(frames, input, P):
    # # params for ShiTomasi corner detection
    # feature_params = dict(maxCorners = 300,
    #                     qualityLevel = 0.3,
    #                     minDistance = 7,
    #                     blockSize = 7 )

    # # params for lucas kanade optical flow
    # lk_params = dict( winSize  = (15, 15),
    #                 maxLevel = 2,
    #                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # result = []
    # # get initial points to track from first frame
    # frame0_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
    # curr_pts = cv.goodFeaturesToTrack(frame0_gray,  mask=None, **feature_params)
    # curr_pts_resized = np.resize(curr_pts, (curr_pts.shape[0], 2))
    # result.append(curr_pts_resized)

    # # # checked that it's working - possiibly play around with parameters
    # # plt.imshow(frames[0])
    # # plt.scatter(curr_pts.squeeze()[:,0], curr_pts.squeeze()[:,1],s=1,c='orange')
    # # plt.savefig('../extra/tracked_pts_medusa/frame0.png')
    # # plt.clf()

    # # loop through frames and track points with KLT
    # for i in range(1, len(frames)):
    #     # perform lucas kanade optical flow
    #     next_pts, is_found, err = cv.calcOpticalFlowPyrLK(frames[i-1], frames[i], curr_pts, None, **lk_params)
    #     # keep tracked points that were found and update curr_pts
    #     # print(next_pts.shape)
    #     curr_pts = next_pts.reshape(-1, 1, 2)
    #     curr_pts_resized = np.resize(curr_pts, (curr_pts.shape[0], 2))
    #     result.append(curr_pts_resized)

    # # return final tracked points 
    # return result  

    # SIFT based feature matching

    result = [] # will be a list with F elements (one for each frame), with each element being a 2xP numpy array
    P = int(P)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    for i in range(len(frames)-1):

        if i % 2 == 1:
            continue

        img1 = frames[i]
        img2 = frames[i+1]

        # only consider a fixed ROI in the image.
        # medusa is 720 x 576 pixels
        # castle is 768 x 576 pixels
        # if (input == "medusa"):
        #     img1 = img1[100:350,250:600,:]
        
        # if i == 0:
        #     plt.imshow(img1)
        #     plt.show()

        kp1, kp2, desc1, desc2 = get_features(img1, img2, P)

        # match features
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        # if i == 0:
        #     ratio_thresh = 0.7
        #     good_matches = []
        #     for m,n in matches:
        #         if m.distance < ratio_thresh * n.distance:
        #             good_matches.append(m)
        #     #-- Draw matches
        #     img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
        #     cv.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #     #-- Show detected matches
        #     plt.imshow('Good Matches', img_matches)
        #     plt.show()

        # # sort matches by best
        # matches = sorted(matches, key= lambda x:x.distance)

        temp_src = np.zeros((2, P))
        temp_dest = np.zeros((2, P))
            
        k = 0
        for (m,n) in matches:
            src_pt = kp1[m.queryIdx].pt  # currently a 2 element tuple
            dest_pt = kp2[m.trainIdx].pt
            temp_src[:, k] = np.array([ src_pt[0], src_pt[1] ])
            temp_dest[:, k] = np.array([ dest_pt[0], dest_pt[1] ])
            if k == P-1:
                break
            k = k + 1

        result.append(temp_src)
        result.append(temp_dest)

    return result

def get_features(img1, img2, P):

    # SIFT based feature extractor
    sift = cv.xfeatures2d.SIFT_create(nfeatures=P)

    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    return kp1, kp2, desc1, desc2 