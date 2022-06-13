import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import os
import shutil

from modules.train_utils import loop
from modules.dataloaders import *
from modules.fourier_model import *
from modules.d2nn_models import *
from modules.other_models import *
from modules.loss import *
from modules.vis_utils import *
from modules.eval_metrics import *


def train_and_log(cfg):
    '''
        Function to train and log results
        
        Args:
            cfg: The dictionary containing all the required configurations
    '''
    
    
    torch_seed = cfg['torch_seed']
    task_type  = cfg['task_type']
    model_type = cfg['model']
    testing    = cfg['testing']
    shrinkFactor = cfg['shrink_factor'] if 'shrink_factor' in cfg.keys() else 1
    exp_name   = cfg['exp_name']
    save_results_local = cfg['save_results_local'] # Indicates after how many number of epochs results should be saved locally

    train_loader, val_loader = eval(cfg['get_dataloaders'])(cfg['img_size'], cfg['train_batch_size'], torch_seed,  task_type= task_type, shrinkFactor = shrinkFactor, cfg=cfg)
    device = cfg['device']

    torch.manual_seed(torch_seed)
    model = eval(model_type)(cfg).to(device)
            
    criterion= eval(cfg['loss_func']) # Loss function
    opt= torch.optim.Adam(model.parameters(), lr= cfg['learning_rate']) # Initializing the optimizer

    losses_train, losses_val = [], [] # To store train and validation losses
    
    if not os.path.isdir("../results"):
        os.mkdir(f'../results')
    
    print(f'exp results dir: ../results/{exp_name}')
    
    if os.path.isdir(f'../results/{exp_name}'):
        print(f'Deleting existing directory : ../results/{exp_name}')
        shutil.rmtree(f'../results/{exp_name}')
        
    os.mkdir(f'../results/{exp_name}')

    for epoch in range(cfg['epochs']):
        # Train loop:
        loss_train, ssim11rd_train,l1_train,_ ,_               = loop(model, train_loader, criterion, opt, device, type_='train', model_type=model_type, testing= testing, cfg = cfg)
        # Test loop:
        loss_val, ssim11rd_val, l1_val, gt_img_val, pred_img_val = loop(model, val_loader, criterion, opt, device, type_= 'val', model_type=model_type, testing= testing, cfg = cfg)
        
        losses_train.append(loss_train)
        losses_val.append(loss_val)
        
        plot_losses(losses_val, losses_train, return_fig= True)
        
        if (epoch+1)%save_results_local==0:
            plt.savefig(f'../results/{exp_name}/losses_latest.png')
            plt.show()

        caption = f"epoch{epoch+1}(val)@@loss_BerHu({np.round(loss_val, decimals= 5)})@ssim11({np.round(ssim11rd_val, decimals= 5)})@l1({np.round(l1_val, decimals= 5)})"     
        
        if (cfg['model'] == 'fourier_model'):
            fig = plot_phase_amp_weights_fourier(model, pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, cfg = cfg, return_fig = True)  # To plot the phase, amplitudes of ground truth image and predicted image and the weights of the model
        else:
            fig = plot_phase_amp_set(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, cfg = cfg, return_fig = True)  # To plot the phase and amplitudes of a ground truth image and predicted image
            
        if (epoch+1)%save_results_local==0: # Plot 
            fig.savefig(f'../results/{exp_name}/{caption}.png')
            plt.show()
        
        
        fig_clipped = plot_phase_amp_set_clipped(pred_img_val.detach().cpu(), gt_img_val.detach().cpu(), caption= caption, cfg = cfg, return_fig = True)
        
            
        if (epoch+1)%save_results_local==0:
            fig_clipped.savefig(f'../results/{exp_name}/{caption}.png')
            plt.show()
        
        # Save model
        save_model_name=  f'../results/{exp_name}/latest_model.pth' 
        torch.save({
            'state_dict': model.state_dict(),
            'cfg': cfg,
            'epoch': epoch}, save_model_name)
