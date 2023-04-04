import torch
import torch.nn.functional as F
import torch.nn as nn

class BerHu(nn.Module):
    def __init__(self, reduction: str = 'mean', threshold: float = 0.2) -> None :
        '''
            Args:
                reduction (string, optional): Specifies the reduction to apply to the output:
                                              default ('mean')
                threshold (float, optional) : Specifies the threshold at which to change between threshold-scaled L1 and L2 loss.
                                              The value must be positive.  Default: 0.2
                                              (Value based on Mengu et al. https://arxiv.org/abs/2108.07977)
        '''
        
        super(BerHu, self).__init__()
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, X_hat, X):
        '''
            Function to calculate reversed huber distance between predicted and ground truth.
            
            Args:
                X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float)  
                X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float)  
                
            Returns:
                Reversed huber loss (BerHu loss)
        '''
        diff = torch.abs(X-X_hat)

        phi =  torch.std(X, unbiased=False) * self.threshold 

        L1 = -F.threshold(-diff, -phi, 0.)                         # L1 loss for values less than thresh (phi)
        L2 =  F.threshold(diff**2 - phi**2, 0., -phi**2.) + phi**2 # L2 loss for values greater than thresh (phi)


        L2_ = F.threshold(L2, phi**2, -phi**2) + phi**2 # L2 loss + phi^2 for values greater than thresh (phi)
        L2_ = L2_ / (2.*phi) # Equation : (L2 + phi^2)/(2*Phi)

        loss = L1 + L2_

        if(self.reduction ==  'mean'):
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss