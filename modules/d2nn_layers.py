import torch
import numpy as np
from torch import nn
from modules.diffraction import *

class d2nnASwWindow_layer(nn.Module):
    '''
        A diffractive layer of the D2NN
        - uses Angular Spectrum Method to compute wave propagation
    '''
    def __init__(self, n_neurons_input, n_neurons_output, delta_z, lambda_, neuron_size, learn_type='both', device= 'cpu', window_size= 4, weights= None, **kwargs):
        '''
            Initlialize the diffractive layer

            Args : 
                n_neurons_input : Number of input neurons
                n_neurons_output : Number of output neurons
                delta_z : Distance between two adjacent layers of the D2NN
                lambda_ : Wavelength
                neuron_size : Size of the neuron
                learn_type : Type of learnable transmission coefficients
                device  : Device to run the model on
                window_size : Angular Spectrum method computational window size factor(default=4)
                weights : Weights to be loaded if using a pretrained model
        '''
        super(d2nnASwWindow_layer, self).__init__()
        self.n_i               = n_neurons_input
        self.n_o               = n_neurons_output
        self.delta_z           = delta_z
        self.lambda_           = lambda_
        self.neuron_size       = neuron_size
        self.learn_type        = learn_type
        self.w                 = window_size
        
        self.G = get_G_with_window(self.n_i, self.neuron_size, self.delta_z, self.lambda_, w= self.w).to(device) # Obtain frequency domain diffraction propogation function/ transformation

        if weights!= None:
            amp_weights= weights['amp_weights']
            phase_weights= weights['phase_weights']    
            
        
        if (self.learn_type=='amp'): # Learn only the amplitides of transmission coefficients in diffraction layers
            print('Learnable transmission coefficient: Amplitude only')
            if weights== None:
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.phase_weights= torch.zeros((self.n_i, self.n_i), dtype= torch.float).to(device)
            else:
                print('loading weights ... ')
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
                self.phase_weights= phase_weights.to(device)   
        elif (self.learn_type=='phase'): # Learn only the phases of transmission coefficients in diffraction layers
            print('Learnable transmission coefficient: Phase only')
            if weights== None:
                self.amp_weights = torch.ones((self.n_i, self.n_i), dtype= torch.float).to(device) *100000
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.amp_weights = amp_weights.to(device)
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                
        elif (self.learn_type=='both'):  # Learn both the amplitides, phases of transmission coefficients in diffraction layers
            print('Learnable transmission coefficient: Amplitude and Phase')
            if weights== None:
                self.phase_weights= nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
                self.amp_weights = nn.Parameter(torch.randn((self.n_i, self.n_i), dtype= torch.float), requires_grad= True)
            else:
                print('loading weights ... ')
                self.phase_weights= nn.Parameter(phase_weights, requires_grad= True)
                self.amp_weights = nn.Parameter(amp_weights, requires_grad= True)
        else: # Diffraction layers do not have learnable transmission coefficients
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
            Function for forward pass

            Args: 
                input_e_field  : Input electric field (batch_size, self.n_i, self.n_i) immediately before the current layer

            Returns:
                output_e_field : Output electric field (batch_size, self.n_o, self.n_o) immediately before the next layer
        '''
        device = input_e_field.device
        batch_size = input_e_field.shape[0]
        
        ts = (torch.sigmoid(self.amp_weights) * torch.exp(1j*self.phase_weights)).view(1, self.n_i, self.n_i) # Obtain transmission coefficients
        
        input_e_field = input_e_field.view(batch_size, self.n_i, self.n_i)
        
        im_pad= torch.zeros((batch_size, self.w*self.n_i, self.w*self.n_i), dtype= torch.cfloat).to(device)
        im_pad[:, (self.w-1)*self.n_i//2:self.n_i+(self.w-1)*self.n_i//2, (self.w-1)*self.n_i//2:self.n_i+(self.w-1)*self.n_i//2]= input_e_field * ts # field is propogated through the layer which can be larger than the field (i.e. determine by self.w)

        A= torch.fft.fftshift(torch.fft.fft2(im_pad)) * self.G.view(1, self.w*self.n_i, self.w*self.n_i) # propogation (from just after the current layer to the next layer) in frequency domain
        B= torch.fft.ifft2(torch.fft.ifftshift(A)) # convert back to spatial domain
        
        U = B[:, (self.w-1)*self.n_i//2:(self.w-1)*self.n_i//2+self.n_i, (self.w-1)*self.n_i//2:(self.w-1)*self.n_i//2+self.n_i]
        
        output_e_field = U

        return output_e_field