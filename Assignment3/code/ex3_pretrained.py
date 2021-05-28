#!/usr/bin/python
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
fine_tune = True
pretrained=True
verbose = True # Debugging using print statements, will be turned off by default

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
            self.vgg_pret_model = models.vgg11_bn(pretrained=pretrained)
            if verbose:
                print("before removing classifier layers")
                print(self.vgg_pret_model)
            
            #Freezing the layers
            if fine_tune:
                set_parameter_requires_grad(self.vgg_pret_model, fine_tune)

            #Deleting old FC layers from pretrained VGG model
            print("Deleting Classifier Layers")
            del self.vgg_pret_model.avgpool
            del self.vgg_pret_model.classifer
        
            #Adding new FC layers with BN and RELU for CIFAR10 classification
            self.vgg_pret_model.classifier = nn.Sequential(
                nn.Linear(layer_config[0], layer_config[1]),
                nn.BatchNorm2d(layer_config[1]),
                nn.RELU(inplace=True),
                nn.Linear(layer_config[1], num_classes),
            )
        else: # Baseline VGG11_BN model without IMAGENET weights
            self.vgg_baseline = models.vgg11_bn(pretrained=pretrained)
            if verbose:
                print("Baseline VGG Model")
                print(self.vgg_baseline)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if pretrained:
            x = self.vgg_pret_model.features(x)
            #Flatten
            x = x.view(x.size(0),-1)
            #FC layer
            out = self.vgg_baseline.classifier(x)

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
new_valacc = 0 # Validation accuracy per epoch as model trains
old_valacc = 0 # For early stoppinh to check if val accuracy per epoch has not improved
valacc = [] # To track validation accuracies as model trains
stop_count = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        #Per epoch calculation    
        new_valacc = 100 * correct / total
        valacc.append(new_valacc)
        #################################################################################
        # TODO: Q2.b Use the early stopping mechanism from previous questions to save   #
        # the model which has acheieved the best validation accuracy so-far.            #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if old_valacc<=new_valacc:
            old_valacc = new_valacc
            if verbose:
                print('Saving the best model')
            best_model = torch.save(model.state_dict(), 'model.ckpt')
            stop_count = 0
        else:
            stop_count+=1
            if stop_count >=patience: #early stopping check
                if verbose:
                    print('End Training after Epochs: [{}]'.format(epoch))
                    fig = plt.figure(figsize=(10,8))
                    plt.plot(range(1,len(valacc)+1),valacc)
                    # find early stopping checkpoint
                    minposs = valacc.index(max(valacc))+1
                    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
                    plt.xlabel('epochs')
                    plt.ylabel('Validation Accuracy')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('Plot_Validation.pdf')
                    plt.close()
                    break
                
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        print('Validataion accuracy is: {} %'.format(old_valacc))

#################################################################################
# TODO: Use the early stopping mechanism from previous question to load the     #
# weights from the best model so far and perform testing with this model.       #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
if verbose:
    print('Loading the best model')
best_model = torch.load('model.ckpt') 
model.load_state_dict(best_model)
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
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
torch.save(model.state_dict(), 'model.ckpt')

