"""

Code from here: https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset

"""

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# create special obeject to make sure the tensors can be transformed later
class CustomTensorDataset(Dataset):

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
        
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def get_transform(split):
    if split == "train":
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.5101, 0.5193, 0.5548), (0.2032, 0.2001, 0.2025)),
            transforms.RandomErasing()
        ])
        return transform_train
    elif split == "valid":
        transform_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5101, 0.5193, 0.5548), (0.2032, 0.2001, 0.2025)),
        ])
        return transform_valid
    elif split == "test":
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5101, 0.5193, 0.5548), (0.2032, 0.2001, 0.2025)),
        ])
        return transform_test
    elif split == "debug":
        transform_debug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        return transform_debug
    else:
        print("error, wrong split")

# function to get the pickle file
def test_unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        temp_dict = pickle.load(fo, encoding='bytes')
    return temp_dict