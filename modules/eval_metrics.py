import torch.nn.functional as F
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ski_ssim
from pytorch_msssim import ssim as pytorch_ssim
import warnings
warnings.simplefilter('always', UserWarning)

def mse_distance(X_hat, X):
    '''
        Function to calculate MSE distance between predicted and ground truth.
        
        Args:
            X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float)  
            X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float)  
            
        Returns:
             MSE distance
    '''
    return F.mse_loss(X_hat.unsqueeze(dim=1), X.unsqueeze(dim=1)).item()

def L1_distance(X_hat, X):
    '''
        Function to calculate L1 distance between predicted and ground truth.
        
        Args:
            X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float)  
            X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float)  
            
        Returns:
             L1 distance
    '''
    return F.l1_loss(X_hat.unsqueeze(dim=1), X.unsqueeze(dim=1)).item()

def ssim_alt(img1, img2, win_size, data_range, K = [0.01, 0.03]):
        '''
            Author : TTSR: Learning Texture Transformer Network for Image Super-Resolution Resources
            https://github.com/researchmm/TTSR/blob/2836600b20fd8f38e0f1550ab0b87c8d2a2bd276/utils.py
        '''

        C1 = (K[0] * data_range)**2
        C2 = (K[1] * data_range)**2 

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def ssim_verify(X_hat, X, k, dr, value_to_check):
    '''
        Function to verify SSIM with Skimage and alt implementation.
    '''

    X_hat = X_hat.cpu().numpy()
    X = X.cpu().numpy()
    
    ski_ssim_vals  = []
    alt_ssim_vals = []

    for i in range(0,X_hat.shape[0]):
        ski_ssim_vals.append(  ski_ssim(X_hat[i], X[i], win_size=k, gaussian_weights=True, data_range = dr))
        alt_ssim_vals.append(ssim_alt(X_hat[i], X[i], win_size=k, data_range = dr))
    
    v1 = np.mean(ski_ssim_vals)
    v2 = np.mean(alt_ssim_vals)

    check1 = np.allclose(v1, v2, atol = 5e-4)
    check2 = np.allclose(v1, value_to_check.cpu().numpy(), atol = 5e-4)
    
    if((check1 or check2) == False):
        print(check1)
        print(check2)
        raise AssertionError("Skimage ssim = ", v1, ",alt ", v2, ",Pytorch SSIM = ", value_to_check.numpy())

def ssim_pytorch(X_hat, X, k = 11, data_range = 1.0, range_independent = True, double_check = False):
    '''
        Function to calculate SSIM score between predicted and ground truth.
        
        Args:
            X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float) | torch.Tensor
            X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float) | torch.Tensor
            k     : Kernel size for SSIM (defaults to 11) | int
            data_range : Data range for SSIM (defaults to 1.0) | float
            range_independent : Whether to use range independent SSIM (defaults to True) | bool
            double_check : Double check the SSIM score with other implementations (defaults to False) | bool
            
        Returns:
             SSIM score
    '''

    X_hat_ = X_hat.unsqueeze(dim=1)
    X_     = X.unsqueeze(dim=1)
    
    if(range_independent):
        ans =  torch.mean(pytorch_ssim(X_hat_, X_, win_size = k, data_range = data_range, size_average = False, K = (1e-6,1e-6)))
        
        if(double_check):
            ssim_verify(X_hat, X, k, data_range, ans)
            print("Checks Passed!")
        return ans.item()

    else:
        ### Regular SSIM calculation with default parameters ;
        ans = torch.mean(pytorch_ssim(X_hat_, X_, win_size = k, data_range = data_range , size_average = False))
        
        if(double_check):
            ssim_verify(X_hat, X, k, data_range, ans)
            print("Checks Passed!")
        return ans.item()