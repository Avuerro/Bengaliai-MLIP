# PACKAGES
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import imgaug.augmenters as iga
import os
import pdb
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision.models import resnet50,resnet18
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
import time 
import copy
import cv2
import imgaug.augmenters as iaa
import gc


# LOADING THE DATA

DATA_DIR = '../'


class BigDataSet(object):
    def __init__(self,DATA_DIR):
        self.data_dir = DATA_DIR
        self.data_array = []

    def load_parquet_files(self):
        parquet_locations = [ x for x in os.listdir(self.data_dir) if 'train_image_data' in x]
        parquet_locations.sort()
        print(parquet_locations)

        train_csv_location = os.path.join(DATA_DIR, 'train.csv')
        train_data_subset_1_location = os.path.join(DATA_DIR, 'train_image_data_0.parquet')
        train_data_subset_2_location = os.path.join(DATA_DIR, 'train_image_data_1.parquet')
        train_data_subset_3_location = os.path.join(DATA_DIR, 'train_image_data_2.parquet')
        train_data_subset_4_location = os.path.join(DATA_DIR, 'train_image_data_3.parquet')
        train_csv = pd.read_csv(train_csv_location)
        labels = train_csv[['grapheme_root', 'vowel_diacritic','consonant_diacritic']].values

        train_data_subset_1 = pd.read_parquet(train_data_subset_1_location)
        train_data_subset_1 = train_data_subset_1.drop('image_id', axis='columns')
        train_data_subset_1 = train_data_subset_1.to_numpy()
        self.data_array.append(train_data_subset_1)
        del train_data_subset_1
        train_data_subset_2 = pd.read_parquet(train_data_subset_2_location)
        train_data_subset_2 = train_data_subset_2.drop('image_id', axis='columns')
        train_data_subset_2 = train_data_subset_2.to_numpy()
        self.data_array.append(train_data_subset_2)
        del train_data_subset_2
        train_data_subset_3 = pd.read_parquet(train_data_subset_3_location)
        train_data_subset_3 = train_data_subset_3.drop('image_id', axis='columns')
        train_data_subset_3 = train_data_subset_3.to_numpy()
        self.data_array.append(train_data_subset_3)
        del train_data_subset_3
        train_data_subset_4 = pd.read_parquet(train_data_subset_4_location)
        train_data_subset_4 = train_data_subset_4.drop('image_id', axis='columns')
        train_data_subset_4 = train_data_subset_4.to_numpy()
        self.data_array.append(train_data_subset_4)
        del train_data_subset_4
        self.data_array = np.concatenate(self.data_array,axis=0)
        self.data_array = self.data_array.reshape(-1,137,236)

        return self.data_array,labels

# dataset = BigDataSet(DATA_DIR)
# dataset.load_parquet_files()
