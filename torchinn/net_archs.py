# python 3

import torch
import torch.nn as nn


# Basic MLP class
class MLP(nn.Module):
    def __init__(self, din, dout, depth, width, activation=nn.ELU, mask=None):
        """ Basic MLP """
        super().__init__()

        self.din  = din
        self.dout = dout

        net = []

        # Mask
        if mask is None:
            self.mask = torch.ones(self.dout)
        else:
            self.mask = torch.tensor(mask)

        # Input layer
        net.append(nn.Linear(din, width))

        # Hidden layers
        for _ in range(depth):
            net.append(nn.Linear(width, width))
            net.append(activation())

        # Output layer
        net.append(nn.Linear(width, dout))

        # Create model
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(self.mask*x)

# class ActivatedLayer(nn.Module):
#     def __init__(self, in_features, out_features,
#                  activation, omega_0,
#                  is_first=False, is_last=False):
#         super().__init__()
#         self.omega_0    = omega_0
#         self.is_first   = is_first
#         self.is_last    = is_last
#         self.activation = activation
#
#         # Define activation function
#         if self.activation == 'siren':
#             self.act_fn = torch.sin
#         elif self.activation == 'elu':
#             self.act_fn = nn.functional.elu
#
#         if self.is_last:
#             self.act_n = nn.Identity
#
#         self.in_features = in_features
#         self.linear = nn.Linear(in_features, out_features)
#
#         if self.activation == 'siren':
#             self.init_weights()
#
#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features, 
#                                              1 / self.in_features)      
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6/self.in_features)/self.omega_0, 
#                                              np.sqrt(6/self.in_features)/self.omega_0)
#
#     def forward(self, x):
#         return self.act_fn(self.omega_0*self.linear(x))
#
# class MLP(nn.Module):
#     def __init__(self, din, dout, depth, width,
#                  activation='elu',
#                  first_omega_0=1.0, hidden_omega_0=1.0):
#         super().__init__()
#    
#         net = []
# 
#         # Input layer
#         net.append(ActivatedLayer(din, width, activation, first_omega_0, is_first=True))
# 
#         # Hidden layers
#         for _ in range(depth):
#             net.append(ActivatedLayer(width, width, activation, hidden_omega_0))
# 
#         # Output layer
#         net.append(ActivatedLayer(width, dout, activation, hidden_omega_0, is_last=True))
#         
#         # Create model
#         self.model = nn.Sequential(*net)
#     
#     def forward(self, x):
#         return self.model(x)
