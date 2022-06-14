import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import glob
import cv2
import torchvision
import matplotlib.pyplot as plt
  
class qpm_np_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get HeLa dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/amp/00001.png`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
            task_type : type of input to output conversion
            cfg       : config dictionary
    '''
    
    def __init__(self, data_dir='datasets/qpm_np', type_= 'train', transform= None, task_type= 'phase2intensity', cfg= None, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.cfg= cfg

        self.amp_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/amp/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        self.phase_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/phase/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        
        assert len(self.amp_img_dirs) == len(self.phase_img_dirs), 'Number of phase and amp images are different !!!'
    def __len__(self):
        return len(self.amp_img_dirs)
        
    def __getitem__(self, idx): 
        amp_img = np.load(self.amp_img_dirs[idx])
        phase_img = np.load(self.phase_img_dirs[idx])

        amp_img= self.transform(amp_img)
        phase_img= self.transform(phase_img) 
        
        if 'dataset_debug_opts' in self.cfg.keys():
            if 'clip_phase' in self.cfg['dataset_debug_opts']: # 'clip_phase@phase_set_pi'
                delta = 0.000001
                angle_max= eval(self.cfg['angle_max'])
                if self.cfg['angle_max'] == 'np.pi': # this is for HeLa pi dataset
                    phase_img= torch.clip(phase_img, min=0, max= (2*np.pi) - delta)
                else:
                    phase_img= torch.clip(phase_img, min=0, max= angle_max - delta)

                if 'phase_set_pi' in self.cfg['dataset_debug_opts']:
                    if self.cfg['angle_max'] == 'np.pi': # this is for HeLa pi dataset
                        phase_img = (phase_img/(2*np.pi))*angle_max
                    else:
                        phase_img = phase_img/angle_max * np.pi
            else:
                raise ValueError(f"no 'clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
        else:
            raise ValueError(f"no 'dataset_debug_opts: clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
                
        if self.task_type=='phase2intensity':
            qpm_img= amp_img * torch.exp(1j*phase_img)   
        else:
            raise NotImplementedError(f"Code should be verified for task_type : {task_type}")

        return qpm_img, phase_img
    

class mnist_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get mnist dataset
        
        Args:
            data      : Numpy arrays (n_samples, 1, image_size, image_size) ## to be completed
            labels    : Numpy array of targets (n_samples,) ## to be completed
            transform : torchvision.transforms
            task_type : type of input to output conversion
    '''
    
    def __init__(self, data= None, labels= None,transform= None,task_type= 'phase2intensity', **kwargs):
        self.transform= transform
        self.task_type= task_type

        self.data= np.array(data)
        self.labels= np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transformed_img = self.transform(Image.fromarray(self.data[idx]))
        
        if self.task_type=='phase2intensity':
            mnist_img = torch.exp(1j*transformed_img*np.pi)  # Convert input ground truth images to phase images
        else:
            raise NotImplementedError(f"Code should be verified for task_type : {task_type}")
        
        return mnist_img, torch.tensor(self.labels[idx])

    
    
class wide_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get QPM dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/P1024_0.npy`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='datasets/wide_dataset', type_= 'train', transform= None, task_type= 'phase2amp', **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.complex_img_dirs   = sorted(glob.glob(f'{data_dir}/{type_}/*'))
       
    def __len__(self):
        return len(self.complex_img_dirs)

    def __getitem__(self, idx): 
        complex_img = np.load(self.complex_img_dirs[idx])

        amp_img_   = Image.fromarray(np.abs(complex_img).astype('float64'), 'RGB')
        phase_img_ = Image.fromarray(np.angle(complex_img).astype('float64'), 'RGB')

        amp_img   = self.transform(amp_img_)
        phase_img = self.transform(phase_img_)
        
        if self.task_type=='phase2intensity':
            qpm_img= amp_img * torch.exp(1j*phase_img*np.pi)
        else:
            raise NotImplementedError(f"Code should be verified for task_type : {task_type}")
        
        return qpm_img, 0

class bacteria_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get the bacteria dataset
        
        Args:
            data_dir  : data directory which contains data hierarchy as `./train/amp/00001.png`
            type_     : whether dataloader if train/ val 
            transform : torchvision.transforms
    '''
    
    def __init__(self, data_dir='datasets/bacteria_np', type_= 'train', transform= None, task_type= 'phase2amp',biasOnoise=0, photon_count=1, cfg= None, **kwargs):
        self.transform= transform
        self.task_type= task_type
        self.cfg = cfg
        
        self.biasOnoise = biasOnoise
        self.photon_count = photon_count

        self.phase_img_dirs = sorted(glob.glob(f'{data_dir}/{type_}/*'), key= lambda x: int(x.split('/')[-1][:-4]))
        self.phase_img_dirs = self.phase_img_dirs[:int(len(self.phase_img_dirs)*0.15)]
        
    def __len__(self):
        return len(self.phase_img_dirs)
        
    def __getitem__(self, idx): 
        phase_img = np.load(self.phase_img_dirs[idx], allow_pickle=True)[0].astype('float32')

        phase_img= self.transform(phase_img) + self.biasOnoise
        
        if 'dataset_debug_opts' in self.cfg.keys():
            if 'clip_phase' in self.cfg['dataset_debug_opts']: # 'clip_phase@phase_set_pi'
                delta = 0.000001
                angle_max= eval(self.cfg['angle_max'])
                phase_img= torch.clip(phase_img, min=0, max= angle_max - delta)
                # phase_img= torch.clip(phase_img, min=0, max= (2*np.pi) - delta)

                if 'phase_set_pi' in self.cfg['dataset_debug_opts']:
                    phase_img = phase_img/angle_max * np.pi
                    # phase_img = (phase_img/(2*np.pi))*angle_max
            else:
                raise ValueError(f"no 'clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
        else:
            raise ValueError(f"no 'dataset_debug_opts: clip_phase' in dataloader --> SSIM calculation will be wrong !!!")
        
        if self.task_type=='phase2amp' or self.task_type=='phase2intensity' or self.task_type=='phasenoamp2intensity':
            qpm_img= torch.exp(1j*phase_img)   
        else:
            raise NotImplementedError(f"Code should be verified for task_type : {task_type}")
        
        return qpm_img, phase_img
