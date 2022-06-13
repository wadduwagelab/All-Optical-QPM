import torch
import numpy as np
from modules.eval_metrics import *


def loop(model, loader, criterion, opt, device, type_= 'train', model_type='fourier_model', testing= False, cfg = None):
    '''
        Function to execute an epoch
        
        Args:
            model          : The model
            loader         : The train/ validation loader to load phase images (dtype=cfloat)
            criterion      : The loss function
            opt            : The optimizer
            device         : Device 
            type_          : The type of the loop - 'train' or 'val'. Defaults to 'train'
            model_type     : The model type
            testing        : Before training with higher number of epochs this configuration can be used for code testing
            cfg            : Configurations dictionary
            
        Returns:
             np.mean(losses_t)  : Mean train loss (BerHu) of the epoch
             np.mean(ssim11rd_t): Mean SSIM (k=11) of the epoch
             np.mean(l1_t)      : Mean L1 distance of the epoch
             ground_truth       : The last batch of ground truth images in an epoch ((n_samples, img_size, img_size), dtype=cfloat) | amp range: [0,1], phase range: [0, 2pi] or [0, pi] based on the dataset
             pred_img           : The last batch of predicted images in an epoch ((n_samples, img_size, img_size), dtype=cfloat)
    '''
    
    losses_t= []
    ssim11rd_t = []
    l1_t= []

    img_size   = cfg['img_size']
    shrinkFactor = cfg['shrink_factor'] if 'shrink_factor' in cfg.keys() else 1
    inp_circular = cfg['input_circular'] if 'input_circular' in cfg.keys() else False # If the input field is propagated through a circular aperture
    
    if inp_circular: # Creating a circular mask to apply on the input
        rc = (img_size//2)//shrinkFactor
        xc = torch.arange(-img_size//2,img_size//2,1) 
        xc = torch.tile(xc, (1,img_size)).view(img_size,img_size).to(torch.cfloat)

        yc = torch.arange(img_size//2,-img_size//2,-1).view(img_size,1)
        yc = torch.tile(yc, (1,img_size)).view(img_size,img_size).to(torch.cfloat)

        circ = (abs(xc)**2 + abs(yc)**2 <= (rc)**2).to(torch.float32).view(1,img_size,img_size).to(device)
    else:
        circ = torch.ones(1,img_size,img_size).to(device)

    if(shrinkFactor!=1):
        # To obtain the starting position and ending position of the original image within the padded image 
        csize = int(img_size/shrinkFactor)
        spos  = int((img_size - csize)/2)
        epos  = spos + csize
    else:
        spos = 0
        epos = img_size
        
    angle_max = eval(cfg['angle_max']) if 'angle_max' in cfg.keys() else 2*np.pi # The phase value in the input dataset

    for idx, (x, y) in enumerate(loader):
        if testing== True and idx>2:break
        
        # CLIP ANGLE TO -> [0, angle_max]
        if cfg['get_dataloaders'] != 'get_mnist_dataloaders': # For all datasets except the MNIST dataset the y will have the original phase image
            y = torch.clip(y, min= 0, max= angle_max).to(device) * circ
        
        ground_truth = x[:,0].to(device) * circ # Remove channel dimension

        if type_=='train':
            model.train()
            opt.zero_grad()
            pred_img, out_scale = model(ground_truth) 
        
            pred_img = pred_img[:,spos:epos,spos:epos] # Crop the pred image to extract the region of interest
            if cfg['get_dataloaders'] == 'get_qpm_np_dataloaders' or (cfg['get_dataloaders'] == 'get_bacteria_dataloaders'):
                ground_truth = y[:,0].to(device)[:,spos:epos,spos:epos] /angle_max # Crop and normalize the groundtruth image
                gt_angle = ground_truth
                gt_abs = ground_truth

            else: # For the MNIST dataloader
                ground_truth = ground_truth[:,spos:epos,spos:epos] # Crop the groundtruth image
                gt_angle = ground_truth.angle()/np.pi
                gt_abs = ground_truth.abs()
            
            pred_out= out_scale * pred_img.abs()**2

            loss = criterion(pred_out, gt_angle)
                
            loss.backward()
            opt.step()
        else:

            model.eval()
            
            with torch.no_grad():
                pred_img, out_scale = model(ground_truth)

                pred_img = pred_img[:,spos:epos,spos:epos] # Crop the pred image to extract the region of interest
                if cfg['get_dataloaders'] == 'get_qpm_np_dataloaders' or (cfg['get_dataloaders'] == 'get_bacteria_dataloaders'):
                    gt = y[:,0].to(device)[:,spos:epos,spos:epos] /angle_max # Crop and normalize the groundtruth image
                    gt_angle = gt
                    gt_abs = gt
                    ground_truth = ground_truth[:,spos:epos,spos:epos].abs() + 1j*gt # Preparing the groundtruth in a suitable format for the plot functions
                else:
                    ground_truth = ground_truth[:,spos:epos,spos:epos] # Crop the groundtruth image
                    gt_angle = ground_truth.angle()/np.pi
                    gt_abs = ground_truth.abs()
                    ground_truth = gt_abs + 1j*gt_angle # Preparing the groundtruth in a suitable format for the plot functions

                pred_out= out_scale * pred_img.abs()**2

                loss = criterion(pred_out, gt_angle)  
                
        losses_t.append(loss.item())    
        ssim11rd_t.append(ssim_pytorch(pred_out, (gt_angle), k= 11, range_independent = False))
        l1_t.append(L1_distance(pred_out, gt_angle))


    return np.mean(losses_t), np.mean(ssim11rd_t), np.mean(l1_t), ground_truth, pred_out