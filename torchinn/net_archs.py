# python 3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

# Basic MLP class
# class MLP(nn.Module):
#     def __init__(self, din, dout, depth, width, activation=nn.ELU, mask=None):
#         """ Basic MLP
# 
#         mask : None or Tensor
#             Tensor with mask
#         """
#         super().__init__()
# 
#         net = []
# 
#         # input layer
#         net.append(nn.Linear(din, width))
# 
#         # hidden layers
#         for _ in range(depth):
#             net.append(nn.Linear(width, width))
#             net.append(activation())
# 
#         # Output layer
#         net.append(nn.Linear(width, dout))
# 
#         # Create model
#         self.model = nn.Sequential(*net)
# 
#     def forward(self, x):
#         return self.model(x)
# 
