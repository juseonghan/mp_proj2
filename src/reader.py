import glob
import cv2 as cv

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