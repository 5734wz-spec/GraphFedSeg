import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scipy import ndimage
import albumentations as A
from PIL import Image
import cv2
from scipy.ndimage import zoom
import os
from glob import glob
import random
import numpy as np
import json
import pdb
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import skimage


class Polyp(Dataset):
    def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'val', 'test']

        self.num_classes = 2
        self.fl_method = fl_method

        self.client_name = ['client1', 'client2', 'client3', 'client4']
        self.client_idx = client_idx    # obtain the dataset of client_name[client_idx]

        self.mode = mode
        self.transform = transform

        self.data_list = []

        with open("/opt/data/private/wz/DATA/data_split/Polyp/{}_{}.txt".format(self.client_name[self.client_idx], mode), "r") as f: 
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = self.data_list[idx]
        data = np.load(data_path)

        image = data[..., 0:3]
        label = data[..., 3:]   

        sample = {'image':image, 'label':label,'file_path': data_path}  # 添加文件路径字段
        if self.transform is not None:
            sample = self.transform(sample) 

        return idx, sample

# Fundus
class Fundus(Dataset):
    def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'val', 'test']

        self.num_classes = 2
        self.fl_method = fl_method

        self.client_name = ['client1', 'client2', 'client3', 'client4','client5','client6']
        self.client_idx = client_idx    # obtain the dataset of client_name[client_idx]

        self.mode = mode
        self.transform = transform

        self.data_list = []

        with open("/opt/data/private/wz/1/Dataset/data_split/Fundus/{}_{}.txt".format(self.client_name[self.client_idx], mode), "r") as f: 
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = self.data_list[idx]
        data = np.load(data_path)

        image = data[..., 0:3]
        label = data[..., 3:]   

        sample = {'image':image, 'label':label}
        if self.transform is not None:
            sample = self.transform(sample) 

        return idx, sample

#  Prostate
   
# class Prostate(Dataset):
#     def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
#         assert mode in ['train', 'val', 'test']

#         self.num_classes = 2
#         self.fl_method = fl_method

#         self.client_name = ['client1', 'client2', 'client3', 'client4','client5','client6']
#         self.client_idx = client_idx    # obtain the dataset of client_name[client_idx]

#         self.mode = mode
#         self.transform = transform

#         self.data_list = []

#         with open("/opt/data/private/wz/FedEvi/Dataset/data_split/Prostate1/{}_{}.txt".format(self.client_name[self.client_idx], mode), "r") as f: 
#             self.data_list = json.load(f)

#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx: int):
#         data_path = self.data_list[idx]
#         data = np.load(data_path)

#         image = data[..., 0:3]
#         label = data[..., 3:]   

#         sample = {'image':image, 'label':label}
#         if self.transform is not None:
#             sample = self.transform(sample) 

#         return idx, sample

class Prostate(Dataset):
    def __init__(self, fl_method, client_idx=None, mode='train', transform=None):
        assert mode in ['train', 'val', 'test']

        self.num_classes = 2
        self.fl_method = fl_method
        self.client_name = ['client1', 'client2', 'client3', 'client4','client5','client6']
        self.client_idx = client_idx    # 6个客户端对应不同医疗机构
        self.mode = mode
        self.transform = transform
        self.data_list = []

        # 加载数据路径列表
        with open("/opt/data/private/wz/1/Dataset/data_split/Prostate/{}_{}.txt".format(
            self.client_name[self.client_idx], mode), "r") as f:
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = self.data_list[idx]
        data = np.load(data_path)  # 加载预处理好的npy文件

        # 分离图像和标签(前3通道是图像，后1通道是标签)
        image = data[..., 0:3]  # 保持与Polyp相同的3通道结构
        label = data[..., 3:]    # 单通道分割标签

        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)  # 应用数据增强

        return idx, sample  # 返回索引和样本


class RandomFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        if np.random.uniform() > self.p:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            
        return {'image': image, 'label': label, 'file_path': sample['file_path'] }


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image'].transpose(2, 0, 1).astype(np.float32)
        label = sample['label'].transpose(2, 0, 1)

        return {'image': torch.from_numpy(image.copy()/image.max()), 'label': torch.from_numpy(label.copy()).long(),'file_path': sample['file_path']}


def generate_dataset(dataset, fl_method, client_idx):

    if dataset == 'Polyp':
        from data.dataset import Polyp as Med_Dataset
        train_transform = T.Compose([RandomFlip(p=0.5),ToTensor()]) 
        test_transform = T.Compose([ToTensor()])

    if dataset == 'Fundus':
        from data.dataset import Fundus as Med_Dataset
        train_transform = T.Compose([RandomFlip(p=0.5),ToTensor()]) 
        test_transform = T.Compose([ToTensor()])
    

    if dataset == 'Prostate':
        from data.dataset import Prostate as Med_Dataset
        train_transform = T.Compose([RandomFlip(p=0.5),ToTensor()]) 
        test_transform = T.Compose([ToTensor()])



    data_train = Med_Dataset(fl_method=fl_method, 
                                client_idx=client_idx,
                                mode='train',
                                transform=train_transform)
    
    data_val = Med_Dataset(fl_method=fl_method,
                            client_idx=client_idx,
                            mode='val',
                            transform=test_transform)

    data_test = Med_Dataset(fl_method=fl_method,
                                client_idx=client_idx,
                                mode='test',
                                transform=test_transform)
                                
    return data_train, data_val, data_test

