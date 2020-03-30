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




# finetuning the network
class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.resnet50 = resnet50(pretrained=True,progress=True)
#         for param in self.resnet50.parameters():
#             param.requires_grad = False

        self.learned_features =  nn.Sequential(*list(self.resnet50.children())[:-1])
        
        #self.in_features = self.resnet50.fc.in_features

#         self.fc0 = nn.Linear(32768,256)
        # grapheme_root
        self.sm = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(2048,168)
        # vowel_diacritic 
        self.fc2 = nn.Linear(2048,11)
        # consonant_diacritic
        self.fc3 = nn.Linear(2048,7)

    def forward(self, x):
#         x = self.resnet50(x)
        x = self.learned_features(x)
#         print(x.shape)
#         print(torch.flatten(x,1).shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
#         x = self.fc0(x)
        x1 = self.sm(x)
        x1 = self.fc1(x)
        x2 = self.sm(x)
        x2 = self.fc2(x)
        x3 = self.sm(x)
        x3 = self.fc3(x)
        return x1,x2,x3