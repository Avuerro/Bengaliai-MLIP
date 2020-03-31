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
# dataloader that combines the images and labels into one convenient class
# also uses the make_square function to reshape the images
class BengaliData(Dataset):
    def __init__(self,images,labels=None, transform=None):
        self.images_location = images
        self.labels = labels
        
        # if indices is None:
        #     indices = np.arange(len(self.images))
        # self.indices = indices
        self.train = labels is not None
        self.transform = transform
    
    def __len__(self):

        ## CHANGE THIS !!!        
        return len((self.labels['image_id'].values))
    
    def __getitem__(self,idx):

        ## image die we inladen is afhankelijk van idx 
        ## idx --> image_id uit label file
        ## de image_id is ook direct bestandsnaam voor de afbeelding..
        image_id = self.labels['image_id'].values[idx]
        label = self.labels[self.labels['image_id']==image_id][['grapheme_root','vowel_diacritic','consonant_diacritic']].values
        print(image_id)
        print(os.path.join(self.images_location,image_id + '.png'))
        image = cv2.imread(os.path.join(self.images_location,image_id +'.png'))

        x = image
        if self.transform:
            # x = x.reshape(3,256,256)
            x = self.transform(x)
#            x = x.reshape(256,256,3)
            
        if self.train:
            y = label
            y = torch.from_numpy(y)
            return x,y
        else:
            return x.T

## testing purposes
totensor = transforms.ToTensor()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
composed = transforms.Compose([totensor])



DATA_DIR = '../../data/bengaliai-cv19/'
    
# train data
train_csv_location = os.path.join(DATA_DIR, 'train.csv')

train_csv = pd.read_csv(train_csv_location)
train_images_location = './results/'

bdl = BengaliData(train_images_location,train_csv,transform=composed)


print(bdl.__len__())
image,label = bdl[10]

intermediate = image.numpy().astype(np.float32)
check = intermediate[intermediate>1]
print(check)
print(np.sum(check))
print(np.max(intermediate))
plt.imsave('./test_afbeelding_10.png',image.numpy().T)
# print(image.numpy().astype(np.float32))




