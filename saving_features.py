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
        self.hdf_filename = "predictions.txt"
        self.hdf_path = Path(dir_path, self.hdf_filename)

        # it will make a folder in the directory passed by saving path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        # initialize an HDF5
        if not os.path.isfile(self.hdf_path):
            self.store = pd.HDFStore(self.hdf_path, mode='w')
            self.store.close()
    #
    # def __call__(self, xywh, conf, cls, path):
    #
    #     _dict = {}
    #     for sub, val in zip(self.keys, xywh):
    #         _dict[sub] = val
    #     _dict['conf'] = conf
    #     _dict['cls'] = cls
    #     _dict['path'] = str(path)
    #     self.dict_list.append(_dict)
    def __call__(self, predn, shapes, si, filepath, save_conf=True):

        gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf, filepath) if save_conf else (cls, *xywh)  # label format
            with open(str(self.hdf_path), 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

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


def chunking(coco_pathdir, seen_list, seen_lists_path):
    # im = tf.io.gfile.listdir(coco_pathdir)
    print(coco_pathdir)
    img_pathdir = Path(coco_pathdir, 'images/train2017')
    del_line_from_txt(seen_list)
    # print(img_pathdir)
    # labals_pathdir = Path(coco_pathdir, 'labels/train2017')
    # print(labals_pathdir)
    img = os.listdir(img_pathdir)
    chunk_size = 2400 # 50 * batch_size 32 = 1600
    filter_img = [x for x in img if x not in seen_list]
    rm_list = np.random.choice(filter_img, chunk_size)
    _ = [os.remove(Path(img_pathdir, x)) for x in img if x not in rm_list]
    # _ = [os.remove(Path(labals_pathdir, x.replace('jpg', 'txt'))) for x in img if x not in rm_list]
    print('chunking is done, dataframe returned')
    df = pd.DataFrame(rm_list, columns=['img'])
    filename = Path(seen_lists_path, "seenlists_{}.csv")
    filename = uniqe_name(filename)
    df.to_csv(Path(seen_lists_path, filename), index=False)



def uniqe_name(filename):
    counter = 0
    filename = str(filename)
    while os.path.isfile(str(filename).format(counter)):
        counter += 1
    return filename.format(counter)

def del_line_from_txt(keys):
    path = '/Users/amir/Documents/CODE/Python/yolov7/coco/train2017.txt'
    for key in keys:
        lines = filter(lambda x: x[1:] if x.endswith(key) else '', open(path, "r"))
        open(path, "w").write("".join(lines))

if __name__ == '__main__':

    # testing the SavingPredictions class
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

    # testing chunking function
    # path = Path('/Users/amir/Desktop/Desktop/Picture')
    # seen = ['04145_coloradoablaze_2880x1800.jpg',
    #         'heath-ledger-the-joker-1920x1080-wallpaper-11056.jpg',
    #         'Screen Shot 2019-01-10 at 3.27.27 PM.png',
    #         'brain_microchip_circuits_128559_1920x1080.jpg',
    #         'img.jpg',
    #         '117_orig.jpg',
    #         'reading-book-silhouette-24.jpg',
    #         'v1.jpeg',
    #         'user-birthday-5656109189693440-lawcta.gif']
    # chunking(path, seen_list=seen)