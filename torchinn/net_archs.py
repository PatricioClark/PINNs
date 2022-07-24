# python 3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
class Siren(nn.Module):
    def __init__(self, din, dout, depth, width,
                 mask=None,
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        
        net = []

        # Mask inputs
        if mask is not None:
            net.append(mask)

        # Input layer
        net.append(SineLayer(din, width,
                             is_first=True, omega_0=first_omega_0))

        # Hidden layers
        for _ in range(depth):
            net.append(SineLayer(width, width,
                                 is_first=False, omega_0=hidden_omega_0))

        # Output layer
        final_linear = nn.Linear(width, dout)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6/width) / hidden_omega_0, 
                                          np.sqrt(6/width) / hidden_omega_0)
        net.append(final_linear)
        
        # Create model
        self.model = nn.Sequential(*net)
    
    def forward(self, x):
        return self.model(x)

# Basic MLP class
class MLP(nn.Module):
    def __init__(self, din, dout, depth, width, activation=nn.ELU, mask=None):
        """ Basic MLP

        mask : None or Tensor
            Tensor with mask
        """
        super().__init__()

        net = []

        # input layer
        net.append(nn.Linear(din, width))

        # hidden layers
        for _ in range(depth):
            net.append(nn.Linear(width, width))
            net.append(activation())

        # Output layer
        net.append(nn.Linear(width, dout))

        # Create model
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

