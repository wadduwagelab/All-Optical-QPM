import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexBatchNorm1d, ComplexConv2d, ComplexConvTranspose2d, ComplexLinear, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

class complex_cnn(nn.Module):
    '''
        Complex-valued CNN
    '''
    def __init__(self,cfg):
        '''
            Initialize the CNN model

            Args : 
                cfg : configuration dictionary
        '''
        super(complex_cnn,self).__init__()
        
        self.n_i = cfg['img_size']
        self.kernel_size = cfg['kernel_size']
        self.last_bias = cfg['last_bias'] or cfg['all_bias'] # Adding a bias term to the last conv later | bool
        self.all_bias = cfg['all_bias'] # Adding a bias term to all conv layers | bool
        
        self.conv1 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.all_bias)
        self.conv2 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.all_bias)
        self.conv3 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.all_bias)
        self.conv4 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.all_bias)
        self.conv5 = ComplexConv2d(in_channels= 1, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.last_bias)
        
        output_scale=cfg['output_scale']
        
        if cfg['output_scale_learnable']:
            self.output_scale= nn.Parameter(torch.tensor(output_scale))
        else:
            self.output_scale= output_scale
        
    def forward(self, input_e_field):
        '''
            Function for forward pass
            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i)
            Returns:
                output_e_field : Output E-field of complex-valued CNN
                output_scale   : Scaling factor for the reconstructed image
        '''
        
        x = input_e_field.view(-1, 1, self.n_i, self.n_i)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x[:,0], self.output_scale # Removing the channel dimension
        