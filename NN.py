import tifffile
import torch
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

validation_split = 0.8
dataset_size = len(Data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split*dataset_size))
batch_size = dataset_size - split 
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
TrainData = DataLoader(Data, batch_size = batch_size, shuffle = True, sampler = train_sampler)
TestData = DataLoader(Data, batch_size, shuffle = True, sampler = valid_sampler)
