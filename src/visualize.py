import argparse
import numpy as np
from point_cloud import SfM_Drawer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="shape or motion", type=str)
    args = parser.parse_args()

    if (args.data == "motion"):
        data = np.loadtxt('../results/R.txt')
    elif (args.data == "shape"):
        data = np.loadtxt('../results/S.txt')
    else:
        print("please type shape or motion after --data")
        return
   
    drawer = SfM_Drawer(data, args.data)
    drawer.drawSfM()


if __name__ == "__main__": 
    main()