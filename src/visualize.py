import argparse
import numpy as np
from point_cloud import SfM_Drawer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="medusa or castle", type=str)
    parser.add_argument("--type", help="R or S", type=str)
    args = parser.parse_args()
    if args.dataset == 'medusa':
        if args.type == 'R':
            data = np.loadtxt('./results/R_medusa.txt')
        elif args.type =='S':
            data = np.loadtxt('./results/S_medusa.txt')
        else:
            print("please only type R or S for type")
            return
    elif args.dataset =='castle':
        if args.type == 'R':
            data = np.loadtxt('./results/R_castle.txt')
        elif args.type =='S':
            data = np.loadtxt('./results/S_castle.txt')
        else:
            print("please only type R or S for type")
            return
    else:
        print("please only type medusa or castle for dataset")
        return

    drawer = SfM_Drawer(data, args.type)
    drawer.drawSfM()


if __name__ == "__main__": 
    main()