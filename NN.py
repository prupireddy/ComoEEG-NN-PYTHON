import torch
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

root = os.getcwd()
dataset = datasets.ImageFolder(root,transforms.ToTensor())