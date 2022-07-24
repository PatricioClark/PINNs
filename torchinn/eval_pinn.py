# python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import sys
import numpy as np
import matplotlib.pyplot as plt

from pinn import PhysicsInformedNN, nn_grad

# Load PINN (pde is not loaded by default, but can be passed)
PINN = PhysicsInformedNN.load_from_checkpoint('ckpt/last.ckpt')

# inspect results
print(PINN.inv_ctes[0])
print(PINN.inv_ctes[1])

# Data for plotting
numps = 50
x1 = np.linspace(-1,1,num=numps, dtype=np.float32)
x2 = np.linspace(-1,1,num=numps, dtype=np.float32)
x1 = x1.reshape(-1,1)
x2 = x2.reshape(-1,1)
zs = np.zeros(numps, dtype=np.float32).reshape(-1,1)

# Evalute along x_2=0
X  = np.concatenate((x1,zs), 1)
xt = torch.from_numpy(X).float()
xt = xt.clone().detach().requires_grad_(True)

zt = PINN.main_net(xt)
z1 = zt[:,0]
z2 = zt[:,1]
dt1 = torch.autograd.grad(z1, [xt], grad_outputs=torch.ones_like(z1), create_graph=True)[0]
dt2 = torch.autograd.grad(z2, [xt], grad_outputs=torch.ones_like(z1), create_graph=True)[0]
zt = zt.detach().numpy()
dt1 = dt1.detach().numpy()
dt2 = dt2.detach().numpy()

plt.figure(1)
plt.plot(x1,zt[:,0])
plt.plot(x1,np.sin(np.pi*x1)+1,'ro')
plt.title(r'$f_1(x_1,0)$')

plt.figure(2)
plt.plot(x1,zt[:,1])
plt.plot(x1,x1**3,'ro')
plt.title(r'$f_2(x_1, 0)$')

plt.figure(3)
plt.clf()
plt.plot(x1, dt1[:,0])
plt.plot(x1, np.pi*np.cos(np.pi*x1),'ro')
plt.title(r'$\frac{\partial f_1}{\partial x_1}(x_1,0)$')

plt.figure(4)
plt.clf()
plt.plot(x1, dt2[:,0])
plt.plot(x1, 3*x1**2,'ro')
plt.plot(x1, 3*x1**2,'ro')

# Evalute along x_1=0
X  = np.concatenate((zs,x2), 1)
xt = torch.from_numpy(X).float()
xt = xt.clone().detach().requires_grad_(True)

zt = PINN.main_net(xt)
z1 = zt[:,0]
z2 = zt[:,1]
inv_field = PINN.inv_fields[0](xt[:,1:2]).detach().numpy()
dt1 = torch.autograd.grad(z1, [xt], grad_outputs=torch.ones_like(z1), create_graph=True)[0]
dt2 = torch.autograd.grad(z2, [xt], grad_outputs=torch.ones_like(z1), create_graph=True)[0]
zt = zt.detach().numpy()
dt1 = dt1.detach().numpy()
dt2 = dt2.detach().numpy()

plt.figure(5)
plt.clf()
plt.plot(x2,zt[:,0])
plt.plot(x2,np.cos(np.pi*x2),'ro')
plt.title(r'$f_1(0, x_2)$')

plt.figure(6)
plt.clf()
plt.plot(x2,zt[:,1])
plt.plot(x2,x2**2,'ro')
plt.title(r'$f_2(0, x_2)$')

plt.figure(7)
plt.clf()
plt.plot(x2, dt1[:,1])
plt.plot(x2, inv_field,'g')
plt.plot(x2, -np.pi*np.sin(np.pi*x2),'ro')
plt.title(r'$\frac{\partial f_1}{\partial x_2}(0, x_2)$')

plt.figure(8)
plt.clf()
plt.plot(x2, dt2[:,1])
plt.plot(x2, 2*x2,'ro')
plt.title(r'$\frac{\partial f_2}{\partial x_2}(0, x_2)$')

plt.show()
