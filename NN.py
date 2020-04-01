import tifffile
import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 

def my_tiff_loader(filename):
    return tifffile.imread(filename)

patientNumber = 10
patientNumber = str(patientNumber)
storage = "D:\ComoEEG\Tyler Data\Patient " + patientNumber
os.chdir(storage)
data = datasets.ImageFolder(root = storage,loader = my_tiff_loader ,transform = transforms.ToTensor())
Data = DataLoader(data, batch_size = 2, shuffle = True)
for idx, (x,y) in enumerate(Data):
    print(idx)
