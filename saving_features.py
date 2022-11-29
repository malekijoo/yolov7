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


class SavingPredictions:

    def __init__(self, saving_path):
        self.keys = ('x', 'y', 'w', 'h', 'cls')
        self.dict_list = []

        ''' it will make a folder in the directory passed by saving path '''
        self.saving_path = Path(saving_path, 'fe')
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

    def __call__(self, xywh, conf, cls, path):
        _dict = dict.fromkeys(self.keys)
        for sub, val in zip(_dict, xywh):
            if sub in ['x', 'y']:
                _dict[sub] = math.floor(val)
            elif sub in ['w', 'h']:
                _dict[sub] = math.ceil(val)

        _dict['conf'] = conf
        _dict['cls'] = cls
        _dict['path'] = path

        self.dict_list.append(_dict)

    def save_to_hdf5(self):
        # Export the pandas DataFrame into HDF5
        df_final = pd.DataFrame.from_dict(self.dict_list)
        h5File = "hdf5_predictions.h5"
        print(df_final.head())
        df_final.to_hdf(Path(self.saving_path, h5File), key='df_final', data_columns=True)

# if __name__ == '__main__':
#     df1 = pd.read_hdf(Path(save_path.resolve(), 'hdf5_predictions.h5'))
#     print("DataFrame read from the HDF5 file through pandas:")

