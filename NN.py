import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 


patientNumber = 10
patientNumber = str(patientNumber)
root = "D:\ComoEEG\Tyler Data\Patient " + patientNumber
os.chdir(root)
data = datasets.ImageFolder(root,transforms.ToTensor())
Data = DataLoader(data, batch_size = 2, shuffle = True)
for idx, (x,y) in enumerate(Data):
    print(idx)
