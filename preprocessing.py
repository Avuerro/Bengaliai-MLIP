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

res = iaa.Resize(256)

# change size of image to 256x256
def make_square(img, target_size=256):
    img = img[0:-1, :]
    
    height,width = img.shape
    x = target_size
    y = target_size

    square = np.ones((x, y), np.uint8) * 255
    square[(y - height) // 2:y - (y - height) // 2, (x - width) // 2:x - (x - width) // 2] = img

    return square.T

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


def resize(image, size=(128, 128)):
    return cv2.resize(image, size)


# dataloader that combines the images and labels into one convenient class
# also uses the make_square function to reshape the images
class BengaliDataLoader(Dataset):
    def __init__(self,images,labels=None, transform=None):
        self.images = 'img_data_full.npy'
        self.labels = 'img_labels.npy'
        
        # if indices is None:
        #     indices = np.arange(len(self.images))
        # self.indices = indices
        self.train = labels is not None
        self.transform = transform
    
    def __len__(self):        
        return len(np.load(self.labels))
    
    def __getitem__(self,idx):
        self.images = np.load(self.images)
        self.labels = np.load(self.labels)
#         idx = self.indices[idx]
        img = np.zeros((256, 256, 3))
        tmp = self.images[idx]
#         tmp = (255 - tmp).astype(np.float32) / 255.
        tmp = tmp/255. #tmp.astype(np.float32)/255.

        tmp = crop_char_image(tmp,threshold = 250./255.)
        tmp = res(images=tmp)
        img[..., 0] = tmp
        img[..., 1] = img[..., 0]
        img[..., 2] = img[..., 0]


        x = torch.from_numpy(img)
        if self.transform:
            x = x.reshape(3,256,256)
            x = self.transform(x)

#             x = x.reshape(256,256,3)
            
        if self.train:
            y = self.labels[idx]
            y = torch.from_numpy(y)
            return x,y
        else:
            return x.T

## testing purposes

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#composed = transforms.Compose([normalize])
#bdl = BengaliDataLoader('img_data_full.npy','img_labels.npy',transform=composed)


#bdl.__len__()
#image,label = bdl[10]
#print(image.numpy().shape)

#print(image.numpy().reshape(256,256,3))
#plt.imshow(image.numpy().T)
#plt.imsave('test_afbeelding_%s.png' % str(10), image.numpy().T)
#print(label)
