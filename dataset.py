import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import imgaug.augmenters as iaa
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
# dataloader that combines the images and labels into one convenient class
# PRE-PROCESSING
# resize
res = iaa.Resize(256)
cutout = iaa.Cutout(nb_iterations=(0, 3), size=0.1, squared=True,cval=0)

# crop
def crop_char_image(image, threshold=5./255.):
    assert image.ndim == 2
    is_white = image < threshold
    is_black_vertical = np.sum(is_white, axis=0) > 0
    is_black_horizontal = np.sum(is_white, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image

# DATALOADER
class BengaliDataLoader(Dataset):
    def __init__(self,images,labels=None, transform=None, indices=None):
        self.images = images.drop('image_id', axis='columns')
        self.images = self.images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(self.images))
        self.indices = indices
        self.train = labels is not None
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self,idx):
        # resize and crop
        img = np.zeros((256, 256, 3))
        tmp = self.images.iloc[idx].to_numpy().reshape(137,236)
        tmp = tmp/255.
        tmp = crop_char_image(tmp,threshold = 250./255.)
        tmp = res(images=tmp)
        img[..., 0] = tmp
        img[..., 1] = img[..., 0]
        img[..., 2] = img[..., 0]

        x = torch.from_numpy(img)
        if self.transform:
            # normalize
            x = x.reshape(3,256,256)
            x = self.transform(x)            
        if self.train:
            y = self.labels[idx]
            y = torch.from_numpy(y)
            return x,y
        else:
            return x


# DATALOADER
class BengaliDataLoaderCutOut(Dataset):
    def __init__(self,images,labels=None, transform=None, indices=None):
        self.images = images.drop('image_id', axis='columns')
        self.images = self.images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(self.images))
        self.indices = indices
        self.train = labels is not None
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self,idx):
        # crop, cutout and resize
        img = np.zeros((256, 256, 3))
        tmp = self.images.iloc[idx].to_numpy().reshape(137,236)
        tmp = tmp/255.
        tmp = crop_char_image(tmp,threshold = 250./255.)
        tmp = cutout(images=tmp)
        tmp = res(images=tmp)
        img[..., 0] = tmp
        img[..., 1] = img[..., 0]
        img[..., 2] = img[..., 0]

        x = torch.from_numpy(img)
        if self.transform:
            # normalize
            x = x.reshape(3,256,256)
            x = self.transform(x)            
        if self.train:
            y = self.labels[idx]
            y = torch.from_numpy(y)
            return x,y
        else:
            return x





## testing purposes
# totensor = transforms.ToTensor()

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                               std=[0.229, 0.224, 0.225])
# composed = transforms.Compose([])



# DATA_DIR = '../../data/bengaliai-cv19/'
    
# train data
# train_csv_location = os.path.join(DATA_DIR, 'train.csv')
# class_map = os.path.join(DATA_DIR, 'class_map.csv')
# train_csv = pd.read_csv(train_csv_location)[:50210]
# class_map = pd.read_csv(class_map)
# train_images_location = './results/'

# bdl = BengaliData(train_images_location,train_csv,transform=composed)

# for i in range(0,len(bdl)):
#     image,label = bdl[i]
#     image = image.numpy()
#     label = label.numpy()
#     print('----')
#     print(class_map[class_map['component_type']=='grapheme_root'].iloc[label[0]])
#     print(class_map[class_map['component_type']=='vowel_diacritic'].iloc[label[1]])
#     print(class_map[class_map['component_type']=='consonant_diacritic'].iloc[label[2]])
#     print('-----')
#     plt.imsave('ff_test_%s.png' % str(i),image.reshape(256,256,3))
#     plt.imshow(image.reshape(256,256,3))
#     plt.show()

#print(bdl.__len__())
#image,label = bdl[10]

#intermediate = image.numpy().astype(np.float32)
#check = intermediate[intermediate>1]
#print(check)
#print(np.sum(check))
#print(np.max(intermediate))
#plt.imsave('./test_afbeelding_10.png',image.numpy().T)
# print(image.numpy().astype(np.float32))




