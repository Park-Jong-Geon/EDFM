import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--paths', type=list[str])
parser.add_argument('--save_path', type=str)

args = parser.parse_args()

for file in os.listdir(args.paths[0]):
    arr = [np.load(path+file) for path in args.paths]
    np.concatenate(arr, axis=1).save(args.save_path+file)