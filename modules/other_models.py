import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexBatchNorm1d, ComplexConv2d, ComplexConvTranspose2d, ComplexLinear, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

def conv_block(in_c, out_c, k_size, stride, padding, bias):
    '''
        Function for a single convolutional layer with batchnorm

        Args:
            in_c: input number of channels
            out_c: output number of channels
            k_size: kernel size
            stride: convolutional stride in a layer
            padding: padding size
            bias: True/False to add a bias term
    '''
    return nn.Sequential(
        ComplexConv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=stride, padding=padding, bias=bias),
        ComplexBatchNorm2d(out_c)
    )

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
        self.last_bias = cfg['last_bias'] or cfg['all_bias']
        self.all_bias = cfg['all_bias']
        self.n_layers = cfg['n_layers']
        self.n_channels = cfg['n_channels']
                
        self.layer_blocks: nn.ModuleList[conv_block] = nn.ModuleList()
        
        self.layer_blocks.append(conv_block(1, self.n_channels, self.kernel_size, 1, self.kernel_size//2, self.all_bias))
        
        for idx in range(self.n_layers-2):
            self.layer_blocks.append(conv_block(self.n_channels, self.n_channels, self.kernel_size, 1, self.kernel_size//2, self.all_bias))
        
        self.last = ComplexConv2d(in_channels= self.n_channels, out_channels= 1, kernel_size=self.kernel_size, stride= 1, padding=self.kernel_size//2, bias=self.last_bias)

        output_scale=cfg['output_scale']
        output_bias= cfg['output_bias']
    
        if cfg['output_scale_learnable']:
            self.output_scale= nn.Parameter(torch.tensor(output_scale))
        else:
            self.output_scale= output_scale

        if cfg['output_bias_learnable']:
            self.output_bias= nn.Parameter(torch.tensor(output_bias))
        else:
            self.output_bias= output_bias
        
    def forward(self, input_e_field):
        
        x = input_e_field.view(-1, 1, self.n_i, self.n_i)
        
        for idx in range(len(self.layer_blocks)):
            x= self.layer_blocks[idx](x)
        x = self.last(x)
        
        return x[:,0], self.output_bias, self.output_scale # Removing the channel dimension
