import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train" :  transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale = (0.5, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val" : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase = "train"):
        return self.data_transform[phase](img)

def make_datapath_list(phase = "train"):
    
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    print(target_path)

    path_list = []  

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

class HymenopteraDataset(data.Dataset):
    
    def __init__(self, file_list, traSsform = None, phase = "train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.file_list)
    
    # __getitem__을 작동시킬려고 하려면 dataset(index)
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1
        
        return img_transformed, label
    
