import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

class fourier_layer(nn.Module):
    '''
        Learnable Fourier Filter in the 4-F system
    '''
    def __init__(self, n_neurons_input, n_neurons_output, neuron_size, learn_type='both', device= 'cpu', weights= None, circular = False, **kwargs):
        '''
            Initialize the Fourier Filter

            Args : 
                n_neurons_input  : number of neurons in the input layer
                n_neurons_output : number of neurons in the output layer
                neuron_size : size of the neurons in the input layer
                learn_type  : type of learning to be used for the filter ('amp','phase','both')
                device  : device to be used for the filter
                weights : weights to be used for the filter (if pretrained filter is used)
                circular: whether the filter is circular or not
        '''

        super(fourier_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.circular          = circular # if the filter is circular or not
        
        if weights!= None:
            amp_weights= weights['amp_weights']
            phase_weights= weights['phase_weights']    
            
        
        if (self.learn_type=='amp'):
            print('Learnable transmission coefficient: Amplitude only')
            if weights== None:
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
            else:
                print('loading weights ... ')
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
                self.phase_weights= phase_weights.to(device)   
        elif (self.learn_type=='phase'):
            print('Learnable transmission coefficient: Phase only')
            if weights== None:
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device) *100000
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.amp_weights = amp_weights.to(device)
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                
        elif (self.learn_type=='both'):
            print('Learnable transmission coefficient: Amplitude and Phase')
            if weights== None:
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
        else:
            print('No learnable transmission coefficients')
            if weights== None:
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device)*100000
            else:
                print('loading weights ... ')
                self.phase_weights= phase_weights.to(device)
                self.amp_weights = amp_weights.to(device)*100000  
                
    def forward(self, input_e_field):
        '''
            Forward pass of the Fourier Filter

            Args:
                input_e_field : input electric field (batch_size, self.n_i, self.n_i)

            Returns:    
                output_e_field : output electric field
        '''
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        
        ts = (torch.sigmoid(self.amp_weights) * torch.exp(1j*self.phase_weights)).view(1, self.n_i, self.n_i)
        if self.circular:
            rc = self.n_i//2
            xc = torch.arange(-self.n_i//2,self.n_i//2,1) 
            xc = torch.tile(xc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            yc = torch.arange(self.n_i//2,-self.n_i//2,-1).view(self.n_i,1)
            yc = torch.tile(yc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            circ = (abs(xc)**2 + abs(yc)**2 <= (rc)**2).to(torch.float32).view(1,self.n_i,self.n_i).to(device)
            
            ts = ts * circ
        
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i)
        
        output_e_field = input_e_field * ts

        return output_e_field

class fourier_ring_layer(nn.Module):
     '''
        Learnable Ring Filter (LRF) in the 4-F system
    '''
    def __init__(self, n_neurons_input, n_neurons_output, neuron_size, learn_type='both', device= 'cpu', weights= None, circular = False, ring_step=1, **kwargs):
        '''
            Initialize the LRF layer

            Args : 
                n_neurons_input  : number of neurons in the input layer
                n_neurons_output : number of neurons in the output layer
                neuron_size : size of the neurons in the input layer
                learn_type  : type of learning to be used for the filter ('amp','phase','both')
                device  : device to be used for the filter
                weights : weights to be used for the filter (if pretrained filter is used)
                circular: whether the filter is circular or not
                ring_step: number of pixels in a ring
        '''
        
        super(fourier_ring_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.circular          = circular
        self.ring_step         = ring_step

        ## rings formation
        xx = torch.tensor(list(range(-self.n_i//2, 0)) + list(range(1, self.n_i//2+1)))
        yy = torch.flip(xx.view(self.n_i,1),(0,1))
        xx = torch.tile(xx, (1,self.n_i)).view(1, self.n_i,self.n_i)
        yy = torch.tile(yy, (1,self.n_i)).view(1, self.n_i,self.n_i)

        r_ins= torch.arange(1, self.n_i//2, self.ring_step).view(-1, 1, 1)
        r_outs = r_ins + ring_step

        c_o_outs = (abs(xx)**2 + abs(yy)**2 <= (r_outs**2)).to(torch.cfloat)
        c_o_ins = (abs(xx)**2 + abs(yy)**2 <= (r_ins**2)).to(torch.cfloat)

        self.rings= (c_o_outs - c_o_ins).to(device)
        
        n_rings = self.rings.shape[0]
        
        if weights!= None:
            amp_weights= (weights['amp_weights'] * torch.ones(n_rings,1,1)).to(device)
            phase_weights = torch.zeros(n_rings,1,1)
            phase_weights[0]= weights['phase_weights'] 

        if (self.learn_type=='both'): 
            if weights == None:
                self.phase_weights= nn.Parameter(torch.randn((n_rings, 1, 1), dtype= torch.float).to(device), requires_grad= True)
                self.amp_weights = nn.Parameter(torch.randn((n_rings, 1, 1), dtype= torch.float).to(device), requires_grad= True)
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True).to(device)
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True).to(device)
        elif (self.learn_type=='phase'): ##
            if weights == None:
                self.phase_weights= nn.Parameter(torch.randn((n_rings, 1, 1), dtype= torch.float).to(device), requires_grad= True)
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device) *100000
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True).to(device)
                self.amp_weights = amp_weights.to(device)
        
    def forward(self, input_e_field):
        '''
            Forward pass of the LRF

            Args:
                input_e_field : input electric field (batch_size, self.n_i, self.n_i)

            Returns:    
                output_e_field : output electric field
        '''
        
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        
        ts=  (self.rings * torch.sigmoid(self.amp_weights)* torch.exp(1j*self.phase_weights)).sum(dim=0).view(1, self.n_i, self.n_i)

        if self.circular:
            rc = self.n_i//2
            xc = torch.arange(-self.n_i//2,self.n_i//2,1) 
            xc = torch.tile(xc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            yc = torch.arange(self.n_i//2,-self.n_i//2,-1).view(self.n_i,1)
            yc = torch.tile(yc, (1,self.n_i)).view(self.n_i,self.n_i).to(torch.cfloat)

            circ = (abs(xc)**2 + abs(yc)**2 <= (rc)**2).to(torch.float32).view(1,self.n_i,self.n_i).to(device)
            
            ts = ts * circ
        
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i)
        
        output_e_field = input_e_field * ts

        return output_e_field

class fourier_model(nn.Module):
    '''
        Learnable Fourier Filter Model Class
    '''
    def __init__(self, cfg, layer= fourier_layer):
        '''
            Initialize the Fourier Filter Model

            Args : 
                cfg : configuration dictionary
                layer : layer class to be used for the filter
        '''
        super(fourier_model, self).__init__()
        
        self.n_i = cfg['img_size']
        self.n_o= cfg['img_size']
        self.neuron_size= cfg['neuron_size']
        self.learn_type= cfg['learn_type']
        self.device= cfg['device']
        self.n_layers= 1
        self.circular = cfg['filter_circular'] if 'filter_circular' in cfg.keys() else False
        self.weights = cfg['weights'] if 'weights' in cfg.keys() else None
        self.ring_step = cfg['ring_step'] if 'ring_step' in cfg.keys() else 1
        
        if self.weights== None:
            self.weights= {}
            for idx in range(self.n_layers):
                self.weights[f'layer{idx}']= None
  
        self.layer_blocks: nn.ModuleList[layer] = nn.ModuleList()
            
        self.layer_blocks.append(layer(self.n_i, self.n_o, neuron_size= self.neuron_size, learn_type=self.learn_type, device= self.device, weights= self.weights[f'layer{self.n_layers-1}'], circular = self.circular, ring_step = self.ring_step))
        
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
                output_e_field : Output E-field of the LFF
                output_scale   : Scaling factor for the reconstructed image
        '''
        x= input_e_field.view(-1, self.n_i, self.n_i)
        device = input_e_field.device

        Fs = torch.fft.fft2(x)
        X = torch.fft.fftshift(Fs)
            
        for idx in range(len(self.layer_blocks)):
            X= self.layer_blocks[idx](X)
        
        x_o = torch.fft.ifft2(torch.fft.ifftshift(X))

        return x_o, self.output_scale
