#!/usr/bin/python
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import copy

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3
layer_config= [512, 256]
num_classes = 10
num_epochs = 30
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.99
reg=0#0.001
num_training= 49000
num_validation =1000
#Flag for training feature extaction layers. When True, we will only update the reshaped parms
#When False, we will fine tune the entire model
fine_tune = True 
pretrained=True
# Flag for enabling additional print statements
verbose = True 

# early stopping patience:how long to wait after last time validation loss improved
patience = 5

data_aug_transforms = [transforms.RandomHorizontalFlip(p=0.5)]#, transforms.RandomGrayscale(p=0.05)]
#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
# Q1,
#The following transform code was already given, and it sets mean and variance/std of CIFAR10 dataset 
#to values that  IMAGENET data was normalized to when VGG11_BN was trained. Modified the code for 
#for better readibility
norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=norm_transform
                                          )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def plot_loss_accuracy(avg_train_loss, avg_val_loss, valacc_hist):
    #Plots for training vs Validation loss 
    plt.plot(avg_train_loss, label='Train')
    plt.plot(avg_val_loss,label='Val')                   
    plt.xlabel('epochs')        
    plt.ylabel('loss')
    if pretrained:
        if fine_tune:
            plt.title('Clasification Loss history with Train New FC layers')
        else:
            plt.title('Clasification Loss history with Train all Layers')
    else:
        plt.title('Clasification Loss history with Baseline Model')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    if pretrained:
        if fine_tune:
            plt.savefig('Plot_loss_pretrained.pdf')
        else:
            plt.savefig('Plot_loss_full_train.pdf')
    else:
        plt.savefig('Plot_loss_Baseline.pdf')

    #Val Accuracy curve
    plt.plot(range(1,len(valacc_hist)+1),valacc_hist,label = "Val Acc")
    if pretrained:
        if fine_tune:
            plt.title('Validation Accuracy vs Epochs with Train New FC layers')
        else:
            plt.title('Validation Accuracy vs Epochs with Train All layers')
    else:
        plt.title('Validation Accuracy vs Epochs with Baseline Model')
                    
    # find early stopping checkpoint in the Plot
    minposs = valacc_hist.index(max(valacc_hist))+1
    plt.axvline(minposs, linestyle='--', color='r',label='Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1.) # consistent scale
    plt.xlim(0, len(valacc_hist)+1) # consistent scale
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    if pretrained:
        if fine_tune:
            plt.savefig('Plot_Valacc_Pretrained.pdf')
        else:
            plt.savefig('Plot_Valcc_Full_Train.pdf')
    else:
        plt.savefig('Plot_ValAcc_Baseline.pdf')
        
    plt.close()
class VggModel(nn.Module):
    def __init__(self, n_class, fine_tune, pretrained=True):
        super(VggModel, self).__init__()
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the pretrained flag. You can enable and  #
        # disable training the feature extraction layers based on the fine_tune flag.   #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #Load the pretrained VGG11_BN model
        if pretrained:
            pt_vgg = models.vgg11_bn(pretrained=pretrained)
            if verbose:
                print('##### Before removing classifier layer #####')
                print(pt_vgg)

            #Freeze the layers
            set_parameter_requires_grad(pt_vgg, fine_tune)

            #Deleting old FC layers from pretrained VGG model
            print('### Deleting Avg pooling and FC Layers ####')
            del pt_vgg.avgpool
            del pt_vgg.classifier

            self.model_features = nn.Sequential(*list(pt_vgg.features.children()))
            
            #Adding new FC layers with BN and RELU for CIFAR10 classification
            self.model_classifier = nn.Sequential(
                nn.Linear(layer_config[0], layer_config[1]),
                nn.BatchNorm2d(layer_config[1]),
                nn.ReLU(inplace=True),
                nn.Linear(layer_config[1], num_classes),
            )
        else: # Baseline VGG11_BN model without IMAGENET weights
            self.vgg_scratch = models.vgg11_bn(pretrained=pretrained)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if pretrained:
            x = self.model_features(x)
            x = x.squeeze()
            out = self.model_classifier(x)
        else:
            out = self.vgg_scratch(x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out

# Initialize the model for this run
model= VggModel(num_classes, fine_tune, pretrained)

# Print the model we just instantiated
print(model)

#################################################################################
# TODO: Only select the required parameters to pass to the optimizer. No need to#
# update parameters which should be held fixed (conv layers).                   #
#################################################################################
print("Params to learn:")
if fine_tune:
    params_to_update = []
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
else:
    params_to_update = model.parameters()
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
epoch_valacc = 0 # Validation accuracy per epoch as model trains
best_valacc = 0 # Best Validation accuracy, used for early stopping check
valacc_hist = [] # To track validation accuracy history as model trains
train_loss_hist = [] # To track training loss as model trains
valid_loss_hist = [] # To track validation loss as the model trains
avg_train_loss = [] # Per epoch training loss
avg_val_loss = [] # Per epoch validation loss
stop_count = 0


for epoch in range(num_epochs):
    model.train()# Prepare for training
    for i, (images, labels) in enumerate(train_loader): # Train the model
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        #record training loss
        train_loss_hist.append(loss.item())

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    model.eval() #Prepare for Evaluation
    with torch.no_grad(): # Validate the model
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            outputs = model(images)
            #Caluclate the loss
            loss = criterion(outputs,labels)
            #record the loss
            valid_loss_hist.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        #Per epoch valdication accuracy calculation
        epoch_valacc = correct / total
        valacc_hist.append(epoch_valacc)
 
        # calculate average loss over an epoch
        train_loss = np.average(train_loss_hist)
        valid_loss = np.average(valid_loss_hist)
        avg_train_loss.append(train_loss)
        avg_val_loss.append(valid_loss)

        print ('Epoch [{}/{}], Avg Train Loss: {:.4f}, Avg Val Loss: {:.4f}'
                   .format(epoch+1, num_epochs, train_loss, valid_loss))

        #Clear Train and Val history after each Epoch
        train_loss_hist = []
        valid_loss_hist = []

        #################################################################################
        # TODO: Q2.b Use the early stopping mechanism from previous questions to save   #
        # the model which has acheieved the best validation accuracy so-far.            #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if best_valacc<=epoch_valacc:
            best_valacc = epoch_valacc
            #Store best model weights
            best_model_wts = copy.deepcopy(model.state_dict())
            stop_count = 0
            print('SAVING the best model')
            torch.save(best_model_wts, 'model_'+str(epoch+1)+'.ckpt')
        else:
            stop_count+=1
            if stop_count >=patience: #early stopping check
                if verbose:
                    print('End Training after [{}] Epochs'.format(epoch+1))
                    #Save the best model
                    best_model = best_model_wts
                    torch.save(best_model, 'model_earlystop.ckpt')
                    break
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        print('Validataion accuracy is: {:.2f} %'.format(100* epoch_valacc))

#Plot curves
plot_loss_accuracy(avg_train_loss,avg_val_loss,valacc_hist)
print('Validation Accuracy for the Best Model is: {:.2f} %'.format(100* best_valacc))

#################################################################################
# TODO: Use the early stopping mechanism from previous question to load the     #
# weights from the best model so far and perform testing with this model.       #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
print('LOADING the best model')
#Load the last checkpoint with the best model
model.load_state_dict(torch.load('model_earlystop.ckpt'))
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print("Test the Trained Network")
model.eval() #Prepare for evaluation
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'final_model.ckpt')

