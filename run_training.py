from model import ResNet50
from data_processing import BigDataSet
from preprocessing import BengaliDataLoader
from train import TrainModel

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

## we need data
bigdataset = BigDataSet('./')
data,labels = bigdataset.load_parquet_files()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
composed = transforms.Compose([normalize])

##Pytorch Dataset Class
dataset = BengaliDataLoader(data,labels,transform=composed)


train_dataset,validation_dataset = torch.utils.data.random_split(dataset, [(int) (0.8 * len(dataset)), (int) (0.2 * len(dataset))])
## create training DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=12,shuffle=True)
## Create Validation DataLoader
validation_dataloader = torch.utils.data.DataLoader(validation_dataset,batch_size=12,shuffle=True)







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = ResNet50().to(device)
#     weights = [2,1,1]
#     class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss()
#     optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001)
#     exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode="max",factor=0.1,patience=2,verbose=True)
all_losses={}
all_accuracies={}


## model
model = ResNet50().to(device)
print(model)

## traincode


model, losses, accuracies = TrainModel(model,device,criterion,train_dataloader,validation_dataloader,optimizer_ft,exp_lr_scheduler)

