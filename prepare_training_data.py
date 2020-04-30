# Author: Zhang Chi

from __future__ import division
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from PIL import Image
import os
import re

import pydicom
import pydicom.uid
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--dump-root", type=str, required=False, help="Where to dump the data")
parser.add_argument("--height", type=int, default=512, help="image height") 
parser.add_argument("--width", type=int, default=512, help="image width")  


args = parser.parse_args()

def main():
    i = 0
    folder_list = os.listdir(args.dataset_dir)
    folder_list = sorted(folder_list, key=embedded_numbers)
    for one_patient in tqdm(folder_list):
        
        Out_dir = Path(args.dump_root+'/'+one_patient)
        Out_dir.makedirs_p()

        for find_study in os.listdir(Path(args.dataset_dir)/one_patient):  
            if "StudyID" in find_study or "CT" not in find_study:    #匹配条件，包含studyid或者没有ct
                tmp_list = os.listdir(Path(args.dataset_dir)/one_patient/find_study)
                CT_DCM_list = os.listdir(Path(args.dataset_dir)/one_patient/find_study/tmp_list[0])
                CT_DCM_list = sorted(CT_DCM_list, key=embedded_numbers)
                for i,CT_DCM in enumerate(CT_DCM_list):
                    data = pydicom.read_file(Path(args.dataset_dir)/one_patient/find_study/tmp_list[0]/CT_DCM) 
                    data = data.pixel_array # [121:512-103,:]
                    scipy.misc.imsave(Out_dir/ str(i).zfill(4)+'.jpg',data)  #

                





def embedded_numbers(s):
    re_digits = re.compile(r'(\d+)')  #正则表达式，pattern：数字重复一次及以上
    pieces = re_digits.split(s)           #切割字符串，返回切割后的部分，包括数字
    pieces[1::2] = map(int, pieces[1::2])    #将数字部分整数化
    return pieces

def resize_image(img_file):
    img = scipy.misc.imread(img_file)
    img = scipy.misc.imresize(img, (args.height, args.width))
    return img

if __name__ == '__main__':
    main()
