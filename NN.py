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

n_chan = 22

def my_tiff_loader(filename):
    original = tifffile.imread(filename)
    C,H,W = original.shape
    final = np.zeros((H,W,C))
    for i in range(C):
        final[:,:,i] = original[i,:,:]
    return final

patientNumber = 10
patientNumber = str(patientNumber)
storage = "D:\ComoEEG\Tyler Data\Patient " + patientNumber 
# #Single Data Point Testing:
# storage = "D:\ComoEEG\Tyler Data\Patient 10\ictal"
# os.chdir(storage)
# storage = storage + "\P10_1.TIFF"
# data = my_tiff_loader(storage)
trans = transforms.ToTensor()
Data = datasets.ImageFolder(root = storage, loader = my_tiff_loader,transform = trans)

train_split = 0.5
dataset_size = len(Data)
indices = list(range(dataset_size))
split = int(np.floor(train_split*dataset_size))
np.random.shuffle(indices)
train_indices, valid_indices = indices[:split], indices[split:]

mu_matrix = np.zeros((n_chan, len(train_indices)))
std_matrix = np.zeros((n_chan, len(train_indices)))
counter = 0
for index in train_indices:
    x,y = Data[index]
    x_np = x.numpy()
    mu_vector = np.mean(x_np, axis = (1,2))
    std_vector = np.std(x_np, axis = (1,2))
    mu_matrix[:,counter] = mu_vector
    std_matrix[:,counter] = std_vector
    counter = counter + 1
pop_mean = np.mean(mu_matrix, axis = 1)
pop_std = np.mean(std_matrix, axis = 1)

transNorm = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = pop_mean, std = pop_std)])
NormalizedData = datasets.ImageFolder(root = storage, loader = my_tiff_loader, transform = transNorm)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
size = 16
TrainData = DataLoader(NormalizedData, batch_size = size, sampler = train_sampler)
TestData = DataLoader(NormalizedData, batch_size = size, sampler = valid_sampler)
# for idx, (x,y) in enumerate(TrainData):
#     print(x.shape)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 1,stride = 1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(n_chan,8,kernel_size = 3, stride = 1, padding = (1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(8, eps = 1e-05),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size = 5, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps = 1e-05),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(61*2*16, 70),#Formula for calculating post-layer shape: floor((input+2*padding-kernelsize)/stride length +1)
            nn.Dropout(p=0.5),
            nn.Linear(70,2))
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0),-1)
        out = self.classifier(out)
        return out

num_epochs = 50
num_classes = 2
learning_rate = .001

model = ConvNet()
model = model.float()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-3)

model.train()
total_step = len(TrainData)
acc_list = []
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(TrainData):
        outputs = model(images.float())
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = labels.size(0)
        _,predicted = torch.max(outputs.data, axis = 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct/total)

model.eval()
with torch.no_grad():
    correctTest = 0
    totalTest = 0
    for images,labels in TestData:
        outputs = model(images.float())
        _,predicted = torch.max(outputs.data,1)
        totalTest += labels.size(0)
        correctTest += (predicted == labels).sum().item()

print(correctTest/totalTest)

#from torchsummary import summary
#summary(model,input_size = (22,257,18))

