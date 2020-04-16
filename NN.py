# This program loads in spectrogram images. Seperates train and test data.
# Calculates the mean and standard deviations of all training images 
# across all of the channels. Then puts it into a Convolutional Neural Network. 
# Results are postprocessed for accuracy (in the training as well as the testing sets)
# Test accuracy is outputted in the console. 



import tifffile
import torch
import torch.nn as nn
# import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 
#import matplotlib.pyplot as plt

##Custom Image I/O that returns paths but is not compatible with the rest
##of the program -
# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths. Extends
#     torchvision.datasets.ImageFolder
#     """
#     # override the __getitem__ method. this is the method that dataloader calls
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns 
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         # the image file path
#         path = self.imgs[index][0]
#         # make a new tuple that includes original and the path
#         tuple_with_path = (original_tuple + (path,))
#         return tuple_with_path


#Loader for TIFF files
def my_tiff_loader(filename):
    original = tifffile.imread(filename)
    C,H,W = original.shape
    final = np.zeros((H,W,C))
    for i in range(C):
        final[:,:,i] = original[i,:,:]
    return final

#User-Controlled Parameters
n_chan = 22
patientNumber = 5


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

train_sampler = SubsetRandomSampler(train_indices) #sampler based on train indices
valid_sampler = SubsetRandomSampler(valid_indices)
size = 16
TrainData = DataLoader(NormalizedData, batch_size = size, sampler = train_sampler)
TestData = DataLoader(NormalizedData, batch_size = size, sampler = valid_sampler)
#Used for extracting data from the post-batch:
# for idx, (x,y) in enumerate(TrainData):
#     print(x.shape)

#Defining Model architecture
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
            # nn.Linear(13440,2) - This is for 87.5% overlap
            nn.Linear(1792, 2))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0),-1) #flatten
        out = self.classifier(out)
        return out

#Hyperparams
num_epochs = 50
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

