# This program loads in spectrogram images. Seperates train and test data.
# Calculates the mean and standard deviations of all training images 
# across all of the channels. It then seperates the train and test data,
#applying data augmentation (rolling the data forward - wrapping) on the fly to the 
#train data (requires larger number of epochs to cover all augmentations).
#Then puts it into a Convolutional Neural Network. 
# Results are postprocessed for accuracy (in the training as well as the testing sets)
# Test accuracy is outputted in the console. 



import tifffile
import torch
import torch.nn as nn
# import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
#import matplotlib.pyplot as plt
import random

#User-Controlled Parameters
n_chan = 22
patientNumber = 4
num_epochs = 50


class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index][0]),self.dataset[index][1] #The features (index 0) are mapped, but the targets (index 1) remain the same

    def __len__(self):
        return len(self.dataset)

class Wrapper(object):
    
    def __init__(self,MinWraps,MaxWraps):
        self.min = MinWraps #Minimum shift amount
        self.max = MaxWraps #Maximum shift amount
    
    def __call__(self,sample):
        sample_np = sample.numpy() #convert to numpy to be eligible for roll
        shift = random.randint(self.min,self.max) #randomly generate roll amount - here you can see why you need more epochs, to cover all augmentation possibilites (in this case roll amount)
        wrapped_np = np.roll(sample_np, shift, axis = 2) #roll along time
        wrapped_torch = torch.from_numpy(wrapped_np) #go back into torch tensor 
        return wrapped_torch

#Loader for TIFF files
def my_tiff_loader(filename):
    original = tifffile.imread(filename)
    C,H,W = original.shape
    final = np.zeros((H,W,C))
    for i in range(C):
        final[:,:,i] = original[i,:,:]
    return final

patientNumber = str(patientNumber)
storage = "D:\ComoEEG\Tyler Data\Patient " + patientNumber #directory of input folders
# #Single Data Point Testing:
# storage = "D:\ComoEEG\Tyler Data\Patient 10\ictal"
# os.chdir(storage)
# storage = storage + "\P10_1.TIFF"
# data = my_tiff_loader(storage)
trans = transforms.ToTensor()
Data = datasets.ImageFolder(root = storage, loader = my_tiff_loader,transform = trans) #Searches the directory and loads in the images from the folders

#Create Train and Validate Indices for out-of-sample testing
train_split = 0.5
dataset_size = len(Data)
indices = list(range(dataset_size))
split = int(np.floor(train_split*dataset_size))
np.random.shuffle(indices)
train_indices, valid_indices = indices[:split], indices[split:]

#Calculation of the mean and standard deviation of each channel for image normalization
mu_matrix = np.zeros((n_chan, len(train_indices))) #population storage for statistics on each image
std_matrix = np.zeros((n_chan, len(train_indices))) # ""
counter = 0
for index in train_indices: #for each image
    x,y = Data[index]
    x_np = x.numpy()
    mu_vector = np.mean(x_np, axis = (1,2)) #Calculate the mean and standard deviation for each channel
    std_vector = np.std(x_np, axis = (1,2))
    #Put it into the population matrix
    mu_matrix[:,counter] = mu_vector
    std_matrix[:,counter] = std_vector
    counter = counter + 1
pop_mean = np.mean(mu_matrix, axis = 1) #Calculate the mean of the stats over all images - numerical estimation of the true
pop_std = np.mean(std_matrix, axis = 1)

#New transformation with normalization
transNorm = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = pop_mean, std = pop_std)])
NormalizedData = datasets.ImageFolder(root = storage, loader = my_tiff_loader, transform = transNorm)
#This goes with the custom class defined above - 
# for inputs,labels,paths in NormalizedData:
#     print(paths)

#It's necessary to seperate out the train from the test before batching because augmentation can only be applied to train
tng_predataset = Subset(NormalizedData,train_indices) 
valid_dataset = Subset(NormalizedData,valid_indices)
#Apply augmentation
DataAugmentation = Wrapper(0,1)
tng_dataset = MapDataset(tng_predataset,DataAugmentation)

size = 16
TrainData = DataLoader(tng_dataset, batch_size = size)
TestData = DataLoader(valid_dataset, batch_size = size)
#Used for extracting data from the post-batch:
# for idx, (x,y) in enumerate(TrainData):
#     print(x.shape)

# Defining Model architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 1,stride = 1)) #Dummy Layer because layer 1 cannot accept over 3-dimensional input
        self.layer2 = nn.Sequential(
            nn.Conv2d(n_chan,8,kernel_size = 3, stride = 1, padding = (1,1)), #Same Padding
            nn.ReLU(),
            nn.BatchNorm2d(8, eps = 1e-05), #Batch Norm to prevent covariate shift
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size = 5, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps = 1e-05),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), #Dropout regularization
            nn.Linear(61*2*16, 70),#Formula for calculating post-layer shape: floor((input+2*padding-kernelsize)/stride length +1)
            nn.Dropout(p=0.5),
            nn.Linear(70,2))
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0),-1) #flatten
        out = self.classifier(out)
        return out

#Hyperparams
num_classes = 2
learning_rate = .001

#Model Object
model = ConvNet()
model = model.float()

criterion = nn.CrossEntropyLoss() #Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-3) #Optimizer

model.train() #Train Mode
total_step = len(TrainData)
acc_list = [] #Storing accuracy for each batch
for epoch in range(num_epochs): #For each epoch
    for i, (images,labels) in enumerate(TrainData):#For each batch
        outputs = model(images.float()) #Predict
        loss = criterion(outputs,labels) #Loss
        optimizer.zero_grad() #Reset
        loss.backward() #Calculate gradients
        optimizer.step() #Update
        total = labels.size(0) #total
        _,predicted = torch.max(outputs.data, axis = 1) #Predictions from highest value from output neurons
        correct = (predicted == labels).sum().item() #Number correct (item needed to access a single value)
        acc_list.append(correct/total) #Calc and append accuracy

model.eval() #Evaluation mode - done to deactivate the batch norm and dropout layers
with torch.no_grad():
    correctTest = 0
    totalTest = 0
    for images,labels in TestData:
        outputs = model(images.float())
        _,predicted = torch.max(outputs.data,1)
        totalTest += labels.size(0)
        correctTest += (predicted == labels).sum().item()

print(correctTest/totalTest)

##Uncommentable couple lines to output the summary of the network (how data flows through and params)
#from torchsummary import summary
#summary(model,input_size = (22,257,18))

