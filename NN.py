import tifffile
import torch.nn as nn
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

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
# storage = "D:\ComoEEG\Tyler Data\Patient 10\ictal"
# os.chdir(storage)
# storage = storage + "\P10_1.TIFF"
# data = my_tiff_loader(storage)
trans = transforms.ToTensor()
Data = datasets.ImageFolder(root = storage, loader = my_tiff_loader,transform = trans)

train_split = 0.8
dataset_size = len(Data)
indices = list(range(dataset_size))
split = int(np.floor(train_split*dataset_size))
size = dataset_size - split 
np.random.shuffle(indices)
train_indices, valid_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
TrainData = DataLoader(Data, batch_size = size, sampler = train_sampler)
TestData = DataLoader(Data, batch_size = size, sampler = valid_sampler)
for idx, (x,y) in enumerate(TrainData):
    print(x.shape)
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size = 5, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer1 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 5, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.drop_out - nn.Dropout()
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000,10)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

num_epochs = 100
num_classes = 2
learning_rate = .01

model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_step = len(TrainData)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(TrainData):
        outputs = model(images)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    for images,labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

