""" Different NN architectures """

import torch
from torch import nn

import numpy as np


# Basic MLP class
class MLP(nn.Module):
    """Basic MLP"""

    def __init__(self, dims, activation='elu', mask=None, **kwargs):
        """ Basic MLP """
        super().__init__()

        # Input and output sizes
        self.din = dims[0]
        self.dout = dims[1]

        # Depth and width of network
        depth = dims[2]
        width = dims[3]

        net = []

        # Mask
        if mask is None:
            self.mask = torch.ones(self.dout)
        else:
            self.mask = torch.tensor(mask)

        # Input layer
        net.append(ActivatedLayer(self.din,
                                  width,
                                  activation=activation,
                                  is_first=True,
                                  **kwargs))

        # Hidden layers
        for _ in range(depth):
            net.append(ActivatedLayer(width,
                                      width,
                                      activation=activation,
                                      **kwargs))

        # Output layer
        net.append(ActivatedLayer(width,
                                  self.dout,
                                  activation=activation,
                                  is_last=True,
                                  **kwargs))

        # Create model
        self.model = nn.Sequential(*net)

    def forward(self, x):
        """Forward pass"""
        return self.model(self.mask * x)


class ActivatedLayer(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 activation='elu',
                 is_first=False,
                 is_last=False,
                 **kwargs):
        super().__init__()
        self.is_first = is_first
        self.is_last  = is_last

        # Define activation function
        if activation == 'elu':
            self.act_fn = nn.functional.elu

        elif activation == 'siren':
            self.omega_0 = kwargs['hidden_omega_0']
            if self.is_first:
                self.omega_0 = kwargs['first_omega_0']
            self.act_fn = lambda x: torch.sin(self.omega_0 * x)

        if self.is_last:
            self.act_fn = nn.Identity()

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

        # Activate weights
        if activation == 'siren':
            self.init_weights()

    def init_weights(self):
        '''Initialize weights'''
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return self.act_fn(self.linear(x))
