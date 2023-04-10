import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import cv2
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_phase_amp_set(pred_img_set, gt_img_set, caption= 'no caption', cfg = None, return_fig = False):    
    '''
        Function to plot phases and amplitudes of ground truth and predicted complex images
        (Mainly used for D2NN training)

            Args:
                pred_img_set : Predicted image set | torch.Tensor
                gt_img_set   : Ground truth complex image set | torch.Tensor
                caption      : Caption for logging and titles | string
                cfg          : Configuration dictionary | dict
                return_fig   : Whether to return the figure | bool

            Returns:
                fig          : figures for results | Figure  
    '''
    
    pred_img_set= pred_img_set.unsqueeze(dim= 1)[0:4] # a set of 4 images from the predicted images batch
    gt_img_set= gt_img_set.unsqueeze(dim= 1)[0:4] # a set of 4 images from the ground truth images batch
    
    gt_angle = gt_img_set.detach().cpu().imag
    gt_abs = gt_img_set.detach().cpu().real
        
    fig = plt.figure(figsize= (9.5,11))
    plt.subplot(3,2,1)
    plt.imshow(cv2.cvtColor(make_grid(gt_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    
    plt.subplot(3,2,2)
    plt.imshow(cv2.cvtColor(make_grid(gt_abs, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')
    
    plt.subplot(3,2,3)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.angle().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    
    plt.subplot(3,2,4)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu()**0.5, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")
    
    plt.subplot(3,2,5)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Intensity")
    
    plt.suptitle(caption)    
    
    log_wandb= False
    
    ## Wandb
    if log_wandb:
        images = wandb.Image(fig  , caption=caption)
    else:images= None
    
    if return_fig:
        return images, fig
    else:
        plt.show()
        return images
    
def plot_phase_amp_set_clipped(pred_img_set, gt_img_set, caption= 'no caption', cfg = None, return_fig = False):  
    '''
        Function to plot phases and amplitudes of ground truth and predicted complex images
        - Outputs are clipped to [0,1]

            Args:
                pred_img_set : Predicted image set | torch.Tensor
                gt_img_set   : Ground truth complex image set | torch.Tensor
                caption      : Caption for logging and titles
                cfg          : Configuration dictionary | dict
                return_fig   : Whether to return the figure | bool

            Returns:
                fig          : figures for results | Figure     
    '''  
    pred_img_set= pred_img_set.unsqueeze(dim= 1)[0:4] # a set of 4 images from the predicted images batch
    gt_img_set= gt_img_set.unsqueeze(dim= 1)[0:4] # a set of 4 images from the groundtruth images batch
    
    gt_angle = gt_img_set.detach().cpu().imag
    gt_abs = gt_img_set.detach().cpu().real
        
    fig = plt.figure(figsize= (9.5,11))
    plt.subplot(3,2,1)
    plt.imshow(cv2.cvtColor(make_grid(gt_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY),vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    
    plt.subplot(3,2,2)
    plt.imshow(cv2.cvtColor(make_grid(gt_abs, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY),vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')
    
    plt.subplot(3,2,3)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.angle().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    
    plt.subplot(3,2,4)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu()**0.5, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY),vmin = 0, vmax =1)
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")

    plt.subplot(3,2,5)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY),vmin=0,vmax=1)
    plt.colorbar()
    plt.title("Reconstructed : Intensity")
    
    plt.suptitle(caption)

    if return_fig:
        return fig
    else:
        plt.show()

def plot_phase_amp_weights_fourier(model, pred_img_set, gt_img_set, caption= 'no caption', cfg = None, return_fig = False):   
    '''
        Function to plot phases and amplitudes of ground truth and predicted complex images
        (Mainly used for Fourier filter training)

            Args:
                pred_img_set : Predicted image set | torch.Tensor
                gt_img_set   : Ground truth complex image set | torch.Tensor
                caption      : Caption for logging and titles
                cfg          : Configuration dictionary | dict
                return_fig   : Whether to return the figure | bool

            Returns:
                fig          : figures for results | Figure     
    '''
    
    pred_img_set= pred_img_set.unsqueeze(dim= 1)[0:4] # a set of 4 images from the predicted images batch
    gt_img_set= gt_img_set.unsqueeze(dim= 1)[0:4] # a set of 4 images from the ground truth images batch
    n_layers = len(model.layer_blocks) 
    add_rows = math.ceil((n_layers*3)/6)
    
    gt_angle = gt_img_set.detach().cpu().imag
    gt_abs = gt_img_set.detach().cpu().real
        
    fig = plt.figure(figsize= (25,8))
    plt.subplot(1+add_rows,6,1)
    plt.imshow(cv2.cvtColor(make_grid(gt_angle, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Phase')
    
    plt.subplot(1+add_rows,6,2)
    plt.imshow(cv2.cvtColor(make_grid(gt_abs, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title('Ground Truth : Amplitude')
    
    plt.subplot(1+add_rows,6,3)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.angle().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Phase")
    
    plt.subplot(1+add_rows,6,4)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu()**0.5, nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Amplitude")
    
    plt.subplot(1+add_rows,6,5)
    plt.imshow(cv2.cvtColor(make_grid(pred_img_set.abs().detach().cpu(), nrow=2, padding= 1).permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY))
    plt.colorbar()
    plt.title("Reconstructed : Intensity")
    
    # Plotting the fourier model layers
    for idx in range(n_layers):
        ts_amp = torch.sigmoid(model.layer_blocks[idx].amp_weights.detach().cpu()) # amplitude coefficients of the layer
        ts_phase = model.layer_blocks[idx].phase_weights.detach().cpu() # phase coefficients of the layer

        plt.subplot(1+add_rows,6,7+(3*idx))
        plt.imshow(ts_phase.numpy())
        plt.colorbar()
        plt.title(f"t (Unwrapped Phase) : Layer{idx}")
        
        plt.subplot(1+add_rows,6,7+(3*idx)+1)
        plt.imshow(ts_phase.numpy()%(2*np.pi))
        plt.colorbar()
        plt.title(f"t (Wrapped Phase) : Layer{idx}")

        plt.subplot(1+add_rows,6,7+(3*idx)+2)
        plt.imshow(ts_amp.numpy())
        plt.colorbar()
        plt.title(f"t (Amplitude) : Layer{idx}")
            
    
    plt.suptitle(caption)   
    
    log_wandb= False
    
    ## Wandb
    if log_wandb:
        images = wandb.Image(fig  , caption=caption)
    else:images= None
    
    if return_fig:
        return images, fig
    else:
        plt.show()
        return images 

def plot_losses(losses_val, losses_train, return_fig= False):
    '''
        Function to plot losses
            Args:
                losses_val   : Validation losses of each epoch | list
                losses_train : Train losses of each epoch | list
    '''


    plt.figure()
    plt.plot(losses_val, label= 'val loss')
    plt.plot(losses_train, label= 'train loss')
    plt.legend()
    plt.title('losses')
    
    if not return_fig:plt.show()
    