import tifffile
import imagecodecs
import torch
from PIL import Image
import os
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 
from PIL import Image
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def tiff(path):
    return io.imread(path)


def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    img.seek(0)
    tiffarray = np.zeros((22,img.height,img.width))
    for i in range(img.n_frames):
        img.seek(i)
        tiffarray[i,:,:] = np.array(img)
    return np.array(tiffarray)

def my_tiff_loader(filename):
    return tifffile.imread(filename)

patientNumber = 10
patientNumber = str(patientNumber)
storage = "D:\ComoEEG\Tyler Data\Patient " + patientNumber
storage = storage + "\ictal\P10_1.TIFF"
data = tiff(storage)
# os.chdir(storage)
# data = datasets.ImageFolder(root = storage,loader = read_tiff ,transform = transforms.ToTensor())
# Data = DataLoader(data, batch_size = 2, shuffle = True)
# for idx, (x,y) in enumerate(Data):
#     print(idx)
