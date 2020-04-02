from model import ResNet50, SeResNext50
#from data_processing import BigDataSet
#from preprocessing import BengaliDataLoader
from train import TrainModel
from dataset import BengaliDataLoader,BengaliDataLoaderCutOut

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
import gc
import imgaug.augmenters as iaa
import sklearn.metrics

## we need data 
# bigdataset = BigDataSet('../')
# data,labels = bigdataset.load_parquet_files()


totensor = transforms.ToTensor()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
composed = transforms.Compose([normalize])




# LOADING DATA
DATA_DIR = './'
# train data
train_csv_location = os.path.join(DATA_DIR, 'train.csv')
train_csv = pd.read_csv(train_csv_location)
# test data
test_csv_location = os.path.join(DATA_DIR,'test.csv')
test_csv = pd.read_csv(test_csv_location)
# class mapping
class_map_location = os.path.join(DATA_DIR, 'class_map.csv')
class_map = pd.read_csv(class_map_location)

train_csv_location = os.path.join(DATA_DIR, 'train.csv')
train_csv = pd.read_csv(train_csv_location)
## t.b.c.
##LOCAL
# train_images_location = '../train'

##CLOUD
train_images_location = './results'

dataset = BengaliDataLoader(train_images_location,train_csv,transform=composed)
# dataset = BengaliDataLoader('img_data_full.npy','img_labels.npy',transform=composed)


train_dataset,validation_dataset = torch.utils.data.random_split(dataset, [(int) (0.8 * len(dataset)), (int) (0.2 * len(dataset))])
## create training DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True, num_workers = 2)
## Create Validation DataLoader
validation_dataloader = torch.utils.data.DataLoader(validation_dataset,batch_size=4,shuffle=True, num_workers = 2)



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
num_epochs = 1
output_dir = './'

# model_trainer  = TrainModel(model,device,criterion,train_dataloader,validation_dataloader,optimizer_ft,exp_lr_scheduler,num_epochs,output_dir)




# For now use this function instead of TrainModel class
# TRAINING
def train_model(model,device ,criterion,train_data,validation_data, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    # keep track of all scores, also for each component separately    
    train_losses = []
    train_losses_grapheme = []
    train_losses_vowel = []
    train_losses_consonant = []
    
    validation_losses = []
    validation_losses_grapheme = []
    validation_losses_vowel = []
    validation_losses_consonant = []
    
    train_accuracies = []
    train_accuracies_grapheme = []
    train_accuracies_vowel = []
    train_accuracies_consonant = []
    
    validation_accuracies = []
    validation_accuracies_grapheme = []
    validation_accuracies_vowel = []
    validation_accuracies_consonant = []

    # set top scores to baseline values
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_grapheme = 0.0
    best_acc_vowel = 0.0
    best_acc_consonant = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        y_true = {"grapheme":None, "vowel": None, "consonant": None}
        y_pred = {"grapheme":None, "vowel": None, "consonant": None}
        
        dataloaders = train_data
        
        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                dataloaders = validation_data
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_grapheme = 0.0
            running_loss_vowel = 0.0
            running_loss_consonant = 0.0
            
            running_corrects = 0
            running_corrects_grapheme = 0
            running_corrects_vowel = 0
            running_corrects_consonant = 0

            # iterate over data
            for inputs, labels in dataloaders:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # get the predictions for the three different components
                    _, preds_c1 = torch.max(outputs[0], 1)
                    _, preds_c2 = torch.max(outputs[1], 1)
                    _, preds_c3 = torch.max(outputs[2], 1)

                    # calculate losses
                    loss_c1 = criterion(outputs[0], labels[:,0]) ##grapheme loss
                    loss_c2 = criterion(outputs[1], labels[:,1]) ##vowel loss
                    loss_c3 = criterion(outputs[2], labels[:,2]) ## consonant loss
                    loss = loss_c1+loss_c2+loss_c3 ## total loss
                    running_loss_grapheme += loss_c1
                    running_loss_vowel +=loss_c2
                    running_loss_consonant +=loss_c3

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                
                # calculate recall over entire dataset
                if y_pred['grapheme'] is None:
                    y_pred['grapheme'] = torch.max(outputs[0],1)[1]
                    y_pred['vowel'] = torch.max(outputs[1],1)[1]
                    y_pred['consonant']=torch.max(outputs[2],1)[1]
                    y_true['grapheme'] = labels[:,0]
                    y_true['vowel'] = labels[:,1]
                    y_true['consonant'] = labels[:,2]
                else:
                    y_pred['grapheme'] = torch.cat([ y_pred['grapheme'] ,torch.max(outputs[0],1)[1] ], dim=0)
                    y_pred['vowel'] = torch.cat([ y_pred['vowel'] ,torch.max(outputs[1],1)[1] ], dim=0) 
                    y_pred['consonant']= torch.cat([ y_pred['consonant'] ,torch.max(outputs[2],1)[1] ], dim=0) 
                    y_true['grapheme'] = torch.cat([  y_true['grapheme'], labels[:,0] ], dim =0)
                    y_true['vowel'] = torch.cat([  y_true['vowel'], labels[:,1] ], dim =0)
                    y_true['consonant'] = torch.cat([  y_true['consonant'], labels[:,2] ], dim =0)


                ## losses

                running_loss += loss.item()
                
            # recall
                
            running_corrects_grapheme = sklearn.metrics.recall_score(y_true['grapheme'].cpu().numpy(), y_pred['grapheme'].cpu().numpy(), average='macro')
            running_corrects_vowel = sklearn.metrics.recall_score(y_true['vowel'].cpu().numpy(), y_pred['vowel'].cpu().numpy(), average='macro') #recall_accuracies[1]#torch.sum(preds_c2 == labels[:,1])
            running_corrects_consonant = sklearn.metrics.recall_score(y_true['consonant'].cpu().numpy(), y_pred['consonant'].cpu().numpy(), average='macro') #recall_accuracies[2]#torch.sum(preds_c2 == labels[:,1])
            total_recal_accuracy = np.average([running_corrects_grapheme,running_corrects_vowel,running_corrects_consonant], weights=[2,1,1])
            
            running_corrects = total_recal_accuracy
        

            # calculate loss and accuracy of this epoch
            
            # Grapheme loss and accuracy
            epoch_loss_grapheme = running_loss_grapheme / len(dataloaders.dataset)
            epoch_acc_grapheme = running_corrects_grapheme

            # Vowel loss and accuracy
            epoch_loss_vowel = running_loss_vowel / len(dataloaders.dataset)
            epoch_acc_vowel = running_corrects_vowel

            # consonant loss and accuracy
            epoch_loss_consonant = running_loss_consonant / len(dataloaders.dataset)
            epoch_acc_consonant = running_corrects_consonant            
            
            #Total loss and Accuracy
            epoch_loss = running_loss / len(dataloaders.dataset) 
            epoch_acc = running_corrects 
            
            if phase == 'train':
                  scheduler.step(epoch_acc)
            
            # save loss and recall of this epoch
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_losses_grapheme.append(epoch_loss_grapheme)
                train_losses_vowel.append(epoch_loss_vowel)
                train_losses_consonant.append(epoch_loss_consonant)

                train_accuracies.append(epoch_acc)
                train_accuracies_grapheme.append(epoch_acc_grapheme)
                train_accuracies_vowel.append(epoch_acc_vowel)
                train_accuracies_consonant.append(epoch_acc_consonant)
            else:
                validation_losses.append(epoch_loss)
                validation_losses_grapheme.append(epoch_loss_grapheme)
                validation_losses_vowel.append(epoch_loss_vowel)
                validation_losses_consonant.append(epoch_loss_consonant)

                validation_accuracies.append(epoch_acc)
                validation_accuracies_grapheme.append(epoch_acc_grapheme)
                validation_accuracies_vowel.append(epoch_acc_vowel)
                validation_accuracies_consonant.append(epoch_acc_consonant)   
            
            # print results of this epoch
            print("The dataloader length %s " % str(len(dataloaders.dataset)))
            print(' {} Grapheme Loss: {:.4f} Grapheme Acc: {:.4f}'.format(phase, epoch_loss_grapheme, epoch_acc_grapheme))
            print(' {} Vowel Loss: {:.4f} Vowel Acc: {:.4f}'.format(phase, epoch_loss_vowel, epoch_acc_vowel))
            print(' {} Consonant Loss: {:.4f} Consonant Acc: {:.4f}'.format(phase, epoch_loss_consonant, epoch_acc_consonant))

            print(' {} Total Loss: {:.4f} Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))

            # if these are the best results yet, change the weights
            if (phase == 'val') and epoch_acc > best_acc:
                best_acc= epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # print results of training
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # return the model with the best weights and save all scores
    model.load_state_dict(best_model_wts)
    PATH = os.path.join(DATA_DIR,'weights', 'seresnext50_saved_weights_%s.pth' % str(time.time()).split(".")[0])
    torch.save(best_model_wts, PATH)
    all_train_losses = {"total_losses": train_losses, 
                  "grapheme_loss": train_losses_grapheme,
                  "vowel_loss":train_losses_vowel,
                  "consonant_loss":train_losses_consonant}
    all_train_accuracies = {"total_acc": train_accuracies, 
                  "grapheme_acc": train_accuracies_grapheme,
                  "vowel_acc": train_accuracies_vowel,
                  "consonant_acc":train_accuracies_consonant}
    all_val_losses = {"total_losses": validation_losses, 
              "grapheme_loss": validation_losses_grapheme,
              "vowel_loss": validation_losses_vowel,
              "consonant_loss": validation_losses_consonant}
    all_val_accuracies = {"total_acc": validation_accuracies, 
              "grapheme_acc": validation_accuracies_grapheme,
              "vowel_acc": validation_accuracies_vowel,
              "consonant_acc": validation_accuracies_consonant}
    return model, all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies


# Train model by looping over parquet files
def parquet_loader(data_dir,train=True):
    # instantiate everything
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001)
    exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode="max",factor=0.1,patience=2,verbose=True)
    
    all_train_losses = {}
    all_train_accuracies= {}
    all_val_losses = {}
    all_val_accuracies= {}
    
    if train:
        parquet_locations = [ x for x in os.listdir(DATA_DIR) if 'train_image_data' in x]
        all_losses = {"parquet_1": {}, "parquet_2":{}, "parquet_3":{},"parquet_4":{}}
        all_accuracies = {"parquet_1": {}, "parquet_2":{}, "parquet_3":{},"parquet_4":{}}

        csv = train_csv
    else:
        parquet_locations = [ x for x in os.listdir(DATA_DIR) if 'test_image_data' in x]
        csv = test_csv

    # loop over parquet files
    for index,parquet_file in enumerate(parquet_locations):
        print("Parquet file number %s " % str(index))
        
        # load parquet file
        data_location = os.path.join(DATA_DIR, parquet_file)
        data = pd.read_parquet(data_location)

        # create dataset
        ids = data['image_id']
        csv_labels = csv[csv['image_id'].isin(ids)]
        labels = csv_labels[['grapheme_root', 'vowel_diacritic','consonant_diacritic']].values
        entire_dataset = BengaliDataLoader(data, labels, transform=composed)
        
        if train:
            # split data into training and validation set
            train_dataset,validation_dataset = torch.utils.data.random_split(entire_dataset, [(int) (0.7 * len(entire_dataset)), (int) (0.3 * len(entire_dataset))])
            train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=12,shuffle=True)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset,batch_size=12,shuffle=True)

            del data
            del ids
            del csv_labels
            del labels
            del entire_dataset
            gc.collect()
            
            # train the model
            model_ft, train_losses, train_accuracies,val_losses,val_accuracies = train_model(model_ft,device, criterion,train_dataloader,validation_dataloader ,optimizer_ft, exp_lr_scheduler,num_epochs=4)
            # model, all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies = model_trainer.train_model()
            del train_dataloader
            del validation_dataloader
            gc.collect()

            np.save('all_train_losses_epoch_%s.npy' % str(index),train_losses)
            np.save('all_train_accuracies_epoch_%s.npy' % str(index),train_accuracies)
            np.save('all_val_losses_epoch_%s.npy' % str(index),val_losses)
            np.save('all_val_accuracies_epoch_%s.npy' % str(index),val_accuracies)

            del train_losses
            del train_accuracies
            del val_losses
            del val_accuracies
            gc.collect()
            
            print(torch.cuda.memory_allocated(device))
    
    # return scores
    return  all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies



# RUN
all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies = parquet_loader(DATA_DIR)

np.save('all_train_losses.npy',all_train_losses)
np.save('all_train_accuracies.npy',all_train_accuracies)
np.save('all_val_losses.npy',all_val_losses)
np.save('all_val_accuracies.npy',all_val_accuracies)