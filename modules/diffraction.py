import torch
import numpy as np

def get_G_with_window(n_neurons_input = 300, neuron_size = 0.0003, delta_z = 0.004, lambda_ = 750e-6, w = 4):
    '''
        Function to calculate the propagation transfer function (frequency domain)

        Args:
            n_neurons_input  : number of neurons on one side of the input layer
            neuron_size      : size of a neuron
            delta_z          : distance between two adjacent layers
            lambda_          : wavelength
            w                : window size
    
        Returns:
            G                : Propogation transfer function (frequency domain) 
    '''
    
    dx= neuron_size
    N= n_neurons_input
    
    if (delta_z == 0): #usecase : delta_z =0 , there is no diffraction.
        return torch.ones((w*N,w*N), dtype=torch.cfloat)
    
    fx = torch.arange(-1/(2*dx),1/(2*dx),1/(w*N*dx)) 
    fx = torch.tile(fx, (1,w*N)).view(w*N,w*N).to(torch.cfloat)

    fy = torch.arange(1/(2*dx),-1/(2*dx),-1/(w*N*dx)).view(w*N,1)
    fy = torch.tile(fy, (1,w*N)).view(w*N,w*N).to(torch.cfloat)

    non_evancent_area = (abs(fx)**2 + abs(fy)**2 <= (1/lambda_)**2)

    power_for_G= 1j*2*np.pi*torch.sqrt((1/lambda_**2)-(fx**2)-(fy**2))*delta_z
    G= torch.exp(power_for_G*non_evancent_area)*non_evancent_area
    
    return G