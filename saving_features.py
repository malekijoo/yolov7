"""
This is the file I added.
Here we are going to receive the prediction of Yolov7.
Then it calls our feature extraction and save features
besides the oreder of images and its path.


"""
import os
import math
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from tables import NaturalNameWarning
from utils.general import xyxy2xywh

warnings.filterwarnings('ignore', category=NaturalNameWarning)

class SavingPredictions:

    def __init__(self, dir_path):


        self.keys = ('x', 'y', 'w', 'h')
        self.dict_list = []

        # all Paths
        self.dir_path = dir_path
        self.csv_filename = "predictions.csv"
        self.csv_path = str(Path(dir_path, self.csv_filename))

        # it will make a folder in the directory passed by saving path
        # if not os.path.exists(self.dir_path):
        #     os.makedirs(self.dir_path)

        # initialize an DataFrame
        self.store = None

    def __call__(self, predn, shapes, si, filename):
        _dict = dict()
        gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            for sub, val in zip(self.keys, xywh):
                _dict[sub] = val
            _dict['conf'] = conf
            _dict['cls'] = cls
            _dict['path'] = str(filename)
        self.dict_list.append(_dict)

    def store2cvs(self):
        # Export the pandas DataFrame into HDF5
        self.store = pd.DataFrame.from_dict(self.dict_list)
        self.store.to_csv(self.csv_path, mode='a', header=False)
        self.dict_list = []


def chunking():

    file = './coco/train2017.txt'
    save_dir = '/content/gdrive/MyDrive/results/'
    gdrive_path = '/content/gdrive/MyDrive/results/labels/train2017_{}.txt'

    lines_per_file = 2400
    smallfile = None
    with open(file) as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = gdrive_path.format(lineno + lines_per_file)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()




if __name__ == '__main__':
    print('hello world!')
    # testing the SavingPredictions class
    # save_path = Path('./fe/').resolve()
    # h5_path = Path(save_path, 'hdf_predictions.h5')
    # obj = SavingPredictions(save_path)
    # obj(torch.tensor([1, 2, 3, 8]), 0.3, 5, 'a')
    # obj.store2hdf()
    # df1 = pd.read_hdf(h5_path)
    # obj(torch.tensor([4, 5, 6, 9]), 0.4, 2, 'b')
    # obj(torch.tensor([7, 10, 0, 1]), 0.2, 4, 'c')
    # obj.store2hdf()
    # df2 = pd.read_hdf(h5_path)
    # print('df2', df2)
    # df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
    # print("DataFrame read from the HDF5 file through pandas:")

    # testing chunking function
    # chunking()