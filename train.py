import time
import copy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import imgaug.augmenters as iga
import numpy as np
import os




## accuracy for multi class problems is different
## this one is used by kaggle for evaluating test performance
import sklearn.metrics


class TrainModel(object):
    def __init__(self, model,device,criterion,train_data,validation_data,optimizer,scheduler,num_epochs,data_dir):
        self.model = model
        self.model_name = self.model.model_name
        self.device = device
        self.criterion = criterion
        self.train_data = train_data
        self.validation_data =validation_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.data_dir = data_dir


    # TRAINING
    def train_model(self):
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
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0


        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            
            y_true = {"grapheme":None, "vowel": None, "consonant": None}
            y_pred = {"grapheme":None, "vowel": None, "consonant": None}
            
            dataloaders = self.train_data
            
            # each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    dataloaders = self.validation_data
                    self.model.eval()   # Set model to evaluate mode

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
                    inputs = inputs.float().to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)

                        # get the predictions for the three different components
                        _, preds_c1 = torch.max(outputs[0], 1)
                        _, preds_c2 = torch.max(outputs[1], 1)
                        _, preds_c3 = torch.max(outputs[2], 1)

                        # calculate losses
                        loss_c1 = self.criterion(outputs[0], labels[:,0]) ##grapheme loss
                        loss_c2 = self.criterion(outputs[1], labels[:,1]) ##vowel loss
                        loss_c3 = self.criterion(outputs[2], labels[:,2]) ## consonant loss
                        loss = loss_c1+loss_c2+loss_c3 ## total loss
                        running_loss_grapheme += loss_c1
                        running_loss_vowel +=loss_c2
                        running_loss_consonant +=loss_c3

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

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
                    self.scheduler.step(epoch_acc)
                
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
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        # print results of training
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_acc))

        # return the model with the best weights and save all scores
        self.model.load_state_dict(best_model_wts)
        PATH = os.path.join(self.data_dir,'weights', '%s_saved_weights_%s.pth' % (self.model_name, str(time.time()).split(".")[0]))
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
        return self.model, all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies