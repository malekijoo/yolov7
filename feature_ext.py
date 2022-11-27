"""
This is the file I added.
Here we are going to receive the prediction of Yolov7.
Then it calls our feature extraction and save features
besides the oreder of images and its path.


"""
import os
import torch
import math
import pandas as pd
from pathlib import Path

class SavingPredictons:
    def __init__(self, saving_path):
        self.keys = ('x1', 'y1', 'x2', 'y2', 'conf', 'cls')
        self.dict_list = []
        self.saving_path = saving_path.resolve()
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

    def __call__(self, prediction, path):
        _dict = dict.fromkeys(self.keys)
        for sub, val in zip(_dict, prediction.tolist()):
            if sub in ['x1', 'y1']:
                _dict[sub] = math.floor(val)
            elif sub in ['x2', 'y2']:
                _dict[sub] = math.ceil(val)
            else:
                _dict[sub] = val

        _dict['path'] = path
        self.dict_list.append(_dict)

    def save_to_hdf5(self):
        # Export the pandas DataFrame into HDF5
        df_final = pd.DataFrame.from_dict(self.dict_list)
        h5File = "hdf5_predictions.h5"
        df_final.to_hdf(Path(self.saving_path, h5File), key='df_final', data_columns=True)