import cv2 as cv
import glob
import numpy as np
import argparse

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
    


def read_data_mp4():
    frames = []
    cap = cv.VideoCapture('../data/medusa.mp4')

    if cap.isOpened() == False:
        print("Error opening medusa file")
        return
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
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
        frames.append(img)
    return frames



if __name__ == "__main__": 
    main()