import time
import copy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import imgaug.augmenters as iga
import numpy as np





## accuracy for multi class problems is different
## this one is used by kaggle for evaluating test performance
import sklearn.metrics


class TrainModel(object):
    def __init__(self, model,device,criterion,train_data,validation_data,optimizer,scheduler,num_epochs,data_dir):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.tain_data = train_data
        self.validation_data =validation_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.data_dir = data_dir

    ## 
    def recall(self, outputs, labels):
        scores = []
        for i in range(0,3):
            y_true_subset = labels[:,i].cpu().numpy()
            y_pred_subset = torch.max(outputs[i],1)[1].cpu().numpy() #.detach().numpy()
            scores.append(sklearn.metrics.recall_score(y_true_subset, y_pred_subset, average='macro'))
        final_score = np.average(scores, weights=[2,1,1])
        return scores, final_score

    # function that trains the model
    def train_model(self):
        since = time.time()
        
        ## ALERT!!!!
        ## separate into train and validation...!!!
        
        losses = []
        losses_grapheme = []
        losses_vowel = []
        losses_consonant = []
        accuracies = []
        accuracies_grapheme = []
        accuracies_vowel = []
        accuracies_consonant = []
        
        best_model_grapheme = copy.deepcopy(self.model.state_dict())
        best_model_vowel = copy.deepcopy(self.model.state_dict())
        best_model_consonant = copy.deepcopy(self.model.state_dict())
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_acc_grapheme = 0.0
        best_acc_vowel = 0.0
        best_acc_consonant = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            
            # data var will be either training or validation data depending on phase
            # however we have to initialise it ..
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
                for inputs, labels in dataloaders:#dataloaders[phase]:
                    inputs = inputs.float().to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)

                        # get the predictions for the three different components
                        _, preds_c1 = torch.max(outputs[0], 1) ## torch.max returned values en indices..
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
                    
                    ## losses

                    running_loss += loss.item() #*inputs.size(0)
                    
                    ## accuracies
    #                 running_corrects_grapheme = (preds_c1 == labels[:,0]).float().mean()
    #                 running_corrects_vowel = (preds_c2 == labels[:,1]).float().mean()
    #                 running_corrects_consonant = (preds_c3 == labels[:,2]).float().mean()

                    recall_accuracies, total_recal_accuracy = self.recall(outputs,labels)
                    
                    running_corrects_grapheme = recall_accuracies[0]#torch.sum(preds_c1 == labels[:,0])
                    running_corrects_vowel = recall_accuracies[1]#torch.sum(preds_c2 == labels[:,1])
                    running_corrects_consonant = recall_accuracies[2]#torch.sum(preds_c2 == labels[:,1])
                    total_recal_accuracy = total_recal_accuracy
    #                 print("------------")
    #                 print(running_corrects_grapheme) #.float().mean())
    #                 print(running_corrects_vowel) #.float().mean())
    #                 print(running_corrects_consonant) #.float().mean())

    #                 print("-----------------------")
    #                 running_corrects += torch.sum(preds_c1 == labels[:,0]) ##grapheme accuracy
    #                 running_corrects += torch.sum(preds_c2 == labels[:,1]) ##vowel accuracy
    #                 running_corrects += torch.sum(preds_c3 == labels[:,2]) ##consonant accuracy

                    running_corrects = total_recal_accuracy
            

                # calculate loss and accuracy of this epoch
                
                # Grapheme loss and accuracy
                epoch_loss_grapheme = running_loss_grapheme / len(self.dataloaders.dataset)
                epoch_acc_grapheme = running_corrects_grapheme #/ len(dataloaders.dataset)

                # Vowel loss and accuracy
                epoch_loss_vowel = running_loss_vowel / len(self.dataloaders.dataset)
                epoch_acc_vowel = running_corrects_vowel #/ len(dataloaders.dataset)

                
                # consonant loss and accuracy
                epoch_loss_consonant = running_loss_consonant / len(self.dataloaders.dataset)
                epoch_acc_consonant = running_corrects_consonant #/ len(dataloaders.dataset)

                
                
                #Total loss and Accuracy
                epoch_loss = running_loss / len(self.dataloaders.dataset) #dataset_sizes[phase]
                epoch_acc = running_corrects 
    #             epoch_acc = running_corrects / len(dataloaders.dataset) #dataset_sizes[phase]
                if phase == 'train':
    #                 scheduler.step(epoch_loss) ## if LRONPLATUE
                    self.scheduler.step(epoch_acc)
                
                
                
                losses.append(epoch_loss)
                losses_grapheme.append(epoch_loss_grapheme)
                losses_vowel.append(epoch_loss_vowel)
                losses_consonant.append(epoch_loss_consonant)
                
    #             accuracies.append(epoch_acc)
                accuracies_grapheme.append(epoch_acc_grapheme)
                accuracies_vowel.append(epoch_acc_vowel)
                accuracies_consonant.append(epoch_acc_consonant)
                
                print("The dataloader length %s " % str(len(self.dataloaders.dataset)))
                # print loss and accuracy score
                print(' {} Grapheme Loss: {:.4f} Grapheme Acc: {:.4f}'.format(phase, epoch_loss_grapheme, epoch_acc_grapheme))
                print(' {} Vowel Loss: {:.4f} Vowel Acc: {:.4f}'.format(phase, epoch_loss_vowel, epoch_acc_vowel))
                print(' {} Consonant Loss: {:.4f} Consonant Acc: {:.4f}'.format(phase, epoch_loss_consonant, epoch_acc_consonant))

                print(' {} Total Loss: {:.4f} Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                
                ## LET OP !!!
                ## dit klopt niet meer
                ## MOET JE BASEREN OP LOSS???
                
                
                if (phase == 'val') and epoch_acc > best_acc:
                    best_acc= epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        # print results
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        PATH = os.path.join(DATA_DIR,'weights', 'resnet50_saved_weights_%s.pth' % str(time.time()).split(".")[0])
        torch.save(best_model_wts, PATH  )#'resnet50_saved_weights.pth')
        all_losses = {"total_losses": losses, 
                    "grapheme_loss": losses_grapheme,
                    "vowel_loss":losses_vowel,
                    "consonant_loss":losses_consonant}
        all_accuracies = {"total_acc": accuracies, 
                    "grapheme_acc": accuracies_grapheme,
                    "vowel_acc":accuracies_vowel,
                    "consonant_acc":accuracies_consonant}
        return self.model, all_losses, all_accuracies