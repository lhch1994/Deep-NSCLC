# Author: Zhang Chi

import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to original dataset')
args = parser.parse_args()


def crawl_folders(root, folders_list):

    with open ('val_list.txt', 'w'):
        pass

    with open ('val_list.txt', 'a') as txt:
        
        
        val_list = random.sample(folders_list, 40)
        print(val_list)
        for img in val_list:
            
            txt.write(img + '\n')


    return  

if __name__ == "__main__":
    root = args.dataset_dir
    scenes = os.listdir(root)
    crawl_folders(root, scenes)