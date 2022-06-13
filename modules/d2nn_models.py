import torch
import numpy as np
from torch import nn
from modules.diffraction import *
from modules.d2nn_layers import *

class d2nnASwWindow(nn.Module):
    '''
        Diffractive Deep Neural Network
    
        Uses angular spectrum method to simulate wave propagation
    '''
    def __init__(self, cfg):
        '''
            Initialization of D2NN

            Args:
                cfg : Configuration dictionary
        '''
        super(d2nnASwWindow, self).__init__()
        
        self.n_i = cfg['img_size']
        self.n_o= cfg['img_size']

        self.delta_z = cfg['delta_z']
        self.lambda_ = cfg['lambda_']
        self.neuron_size= cfg['neuron_size']
        self.learn_type= cfg['learn_type']
        self.device= cfg['device']
        self.n_layers= cfg['n_layers']
        self.energy_type =  'passive'
        self.in_dist  = cfg['in_dist']
        self.out_dist = cfg['out_dist']
        self.window_size= cfg['window_size']
        self.in_dist= cfg['in_dist']
        
        n_hidden= (self.n_i+ self.n_o)//2
        d2nn_layer= d2nnASwWindow_layer
  
        self.layer_blocks: nn.ModuleList[d2nn_layer] = nn.ModuleList()
        
        self.layer_blocks.append(d2nn_layer(self.n_i, n_hidden, self.in_dist, self.lambda_, neuron_size= self.neuron_size, learn_type='no', device= self.device, window_size= self.window_size)) # Initialize a non-learnable layer which mimics the propogation from the specimen to the first layer of D2NN.

        for idx in range(self.n_layers-1): # Initialize middle learnable layers of the D2NN.
            self.layer_blocks.append(d2nn_layer(n_hidden, n_hidden, self.delta_z, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size))
            
        self.layer_blocks.append(d2nn_layer(n_hidden, self.n_o, self.out_dist, self.lambda_, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, energy_type = self.energy_type, window_size= self.window_size)) # Initialize the learnable last layer of D2NN which propogates the field on to the imaging plane.
        
        output_scale=torch.tensor(cfg['output_scale'])
            
        if cfg['output_scale_learnable']: 
            self.output_scale= nn.Parameter(output_scale) # Make the scaling at the output learnable
        else:
            self.output_scale= output_scale
            
        
    def forward(self, input_e_field):
        '''
            Function for forward pass

            Args:
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i) comes from the specimen
                
            Returns:
                x : Output electric field of D2NN on the imaging plane
                self.output_bias: The learned/ fixed bias applied to the output
                self.output_scale: The learned/ fixed scaling applied to the output 
        '''
        x= input_e_field.view(-1, self.n_i, self.n_i)
        device = input_e_field.device
            
        for idx in range(len(self.layer_blocks)):
            x= self.layer_blocks[idx](x)
        
        return x, self.output_scale.to(device)