import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

patientNumber = 10
patientNumber = str(patientNumber)
root = "D:\ComoEEG\Tyler Data\Patient " + patientNumber
os.chdir(root)
dataset = datasets.ImageFolder(root,transforms.ToTensor())