import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
from modules.datasets import *
import matplotlib.pyplot as plt


def get_mnist_dataloaders(img_size, train_batch_size ,torch_seed=10, task_type= 'phase2intensity', shrinkFactor = 1,cfg=None, **kwargs):
    '''
        Function to return train, validation MNIST dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            task_type        : type of input to output conversion
            shrinkFactor     : shrink factor for the image

        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    torch.manual_seed(torch_seed)
    
    data_dir= '/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/mnist'

    train_data = datasets.MNIST(root=data_dir, train=True, download=True)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True)

    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor)), antialias=None), transforms.CenterCrop((img_size, img_size))])

    train_loader = DataLoader(mnist_dataset(data= train_data.data[:54000], labels= train_data.targets[:54000], transform= my_transform, task_type= task_type, cfg= cfg), batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(mnist_dataset(data= train_data.data[54000:], labels= train_data.targets[54000:], transform= my_transform, task_type= task_type, cfg= cfg), batch_size=32, shuffle=False, drop_last= True)

    test_loader = DataLoader(mnist_dataset(data= test_data.data, labels= test_data.targets, transform= my_transform, task_type= task_type, cfg= cfg), batch_size=32, shuffle=False, drop_last= True)

    return train_loader, val_loader, test_loader

def get_qpm_np_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= None, task_type= 'phase2intensity', shrinkFactor = 1, cfg= None, **kwargs):
    '''
        Function to return train, validation HeLa dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierarchy as `./train/amp/00001.png`
            task_type        : type of input to output conversion
            shrinkFactor     : shrink factor for the image
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''

    data_dir= '/content/datasets/hela/'
    #data_dir= '/n/holyscratch01/wadduwage_lab/fypteam_22/datasets/qpm_np_v3_RemovedInverted'
    torch.manual_seed(torch_seed)

    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor)),antialias=None), transforms.CenterCrop((img_size, img_size))])

    train_data = qpm_np_dataset(data_dir=data_dir, type_= 'train', transform = my_transform, task_type= task_type, cfg= cfg)
    val_data   = qpm_np_dataset(data_dir=data_dir, type_= 'val',   transform = my_transform, task_type= task_type, cfg= cfg)
    test_data  = qpm_np_dataset(data_dir=data_dir, type_= 'test',  transform = my_transform, task_type= task_type, cfg= cfg)
    
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader  = DataLoader(val_data, batch_size=15, shuffle=False, drop_last= True)
    test_loader = DataLoader(test_data, batch_size=15, shuffle=False, drop_last= True)

    return train_loader, val_loader, test_loader


def get_bacteria_dataloaders(img_size, train_batch_size ,torch_seed=10, data_dir= None, task_type= 'phase2amp',shrinkFactor = 1, biasOnoise=0, photon_count=1, cfg= None, **kwargs):
    '''
        Function to return train, validation QPM dataloaders
        Args:
            img_size         : Image size to resize
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierachy as `./train/amp/00001.png`
        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
    '''
    
    data_dir= '/content/datasets/bacteria/'
    #data_dir= '/n/holyscratch01/wadduwage_lab/D2NN_QPM_classification/datasets/bacteria_np'
    
    torch.manual_seed(torch_seed)
    # transforms.ToPILImage(), 
    my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((int(img_size//shrinkFactor), int(img_size//shrinkFactor)),antialias=None), transforms.CenterCrop((img_size, img_size))])

    train_data = bacteria_dataset(data_dir=data_dir, type_= 'train', transform = my_transform, task_type= task_type, biasOnoise = biasOnoise, photon_count = photon_count, cfg= cfg)
    val_data   = bacteria_dataset(data_dir=data_dir, type_= 'val',   transform = my_transform, task_type= task_type, biasOnoise = biasOnoise, photon_count = photon_count, cfg= cfg)
    test_data   = bacteria_dataset(data_dir=data_dir, type_= 'test',   transform = my_transform, task_type= task_type, biasOnoise = biasOnoise, photon_count = photon_count, cfg= cfg)
    
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, drop_last= True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, drop_last= True)

    return train_loader, val_loader, test_loader
