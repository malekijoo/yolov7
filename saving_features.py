"""
This is the file I added.
Here we are going to receive the prediction of Yolov7.
Then it calls our feature extraction and save features
besides the oreder of images and its path.


"""
import os

import numpy as np
import torch
import math
import pandas as pd
from pathlib import Path
import warnings
from tables import NaturalNameWarning


warnings.filterwarnings('ignore', category=NaturalNameWarning)

class SavingPredictions:

    def __init__(self, dir_path):


        self.keys = ('x', 'y', 'w', 'h')
        self.dict_list = []

        # all Paths
        self.dir_path = dir_path
        self.hdf_filename = "hdf_predictions.h5"
        self.hdf_path = Path(dir_path, self.hdf_filename)

        # it will make a folder in the directory passed by saving path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        if os.path.exists(self.hdf_path):
            os.remove(self.hdf_path)

        # initialize an HDF5
        self.store = pd.HDFStore(self.hdf_path, mode='w')
        self.store.close()


    def __call__(self, xywh, conf, cls, path):

        _dict = {}
        for sub, val in zip(self.keys, xywh):
            _dict[sub] = val
        _dict['conf'] = conf
        _dict['cls'] = cls
        _dict['path'] = str(path)
        self.dict_list.append(_dict)

    def store2hdf(self):
        # Export the pandas DataFrame into HDF5
        df_final = pd.DataFrame.from_dict(self.dict_list)
        self.store = pd.HDFStore(self.hdf_path, 'a')
        self.store.append(self.hdf_filename,
                          df_final,
                          data_columns=True,
                          min_itemsize={'path': 50, 'cls': 15})
        self.store.close()
        self.dict_list = []


if __name__ == '__main__':

    # testing the module
    save_path = Path('./fe/').resolve()
    h5_path = Path(save_path, 'hdf_predictions.h5')
    obj = SavingPredictions(save_path)
    obj(torch.tensor([1, 2, 3, 8]), 0.3, 5, 'a')
    obj.store2hdf()
    df1 = pd.read_hdf(h5_path)
    obj(torch.tensor([4, 5, 6, 9]), 0.4, 2, 'b')
    obj(torch.tensor([7, 10, 0, 1]), 0.2, 4, 'c')
    obj.store2hdf()
    df2 = pd.read_hdf(h5_path)
    print('df2', df2)

    # df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
    # print("DataFrame read from the HDF5 file through pandas:")

