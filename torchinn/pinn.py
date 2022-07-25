# python3

import torch
import torch.nn as nn
import pytorch_lightning as pl

import numpy as np

# PINN module
class PhysicsInformedNN(pl.LightningModule):
    def __init__(self, din, dout, depth, width,
                 activation='elu', first_omega_0=1, hidden_omega_0=1,
                 pde=None, eq_params=[], inv_ctes=[], inv_fields=[], lr=1e-3):
        super().__init__()

        self.din   = din
        self.dout  = dout
        self.depth = depth
        self.width = width
        self.activation     = activation
        self.first_omega_0  = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        # Other params
        self.lr = lr

        # Main network
        self.main_net   = MLP(din, dout, depth, width,
                              activation     = activation,
                              first_omega_0  = first_omega_0,
                              hidden_omega_0 = hidden_omega_0,
                              )

        # PDE
        if pde is None:
            self.pde = lambda x,y,z: [torch.tensor([0.0])]
        else:
            self.pde = pde

        # Eq params and inverse parts
        self.eq_params  = eq_params
        cte_list        = [nn.Parameter(torch.tensor(k)) for k in inv_ctes]
        self.inv_ctes   = nn.ParameterList(cte_list)
        field_list      = [MLP(inv_in, inv_out, inv_w, inv_d,
                               activation     = activation,
                               first_omega_0  = first_omega_0,
                               hidden_omega_0 = hidden_omega_0,
                               )
                           for inv_in, inv_out, inv_w, inv_d in inv_fields]
        self.inv_fields = nn.ModuleList(field_list)

        # Save hyperparams
        self.save_hyperparameters(ignore=['pde'])

    def forward(self, x):
        out = self.main_net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.clone().detach().requires_grad_(True)
        z = self.main_net(x)
        data_loss = nn.functional.mse_loss(y, z)
        self.log("data_loss", data_loss)

        pde = self.pde(self, z, x)
        phys_loss = torch.stack([eq**2 for eq in pde]).mean()
        self.log("phys_loss", phys_loss)

        return data_loss + phys_loss

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 activation, omega_0, is_first=False):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        self.activation = activation
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()
    
    def init_weights(self):
        if self.activation=='siren':
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.in_features, 
                                                 1 / self.in_features)      
                else:
                    self.linear.weight.uniform_(-np.sqrt(6/self.in_features)/self.omega_0, 
                                                 np.sqrt(6/self.in_features)/self.omega_0)
            
    def forward(self, x):
        return self.linear(x)
    
class SirenAct(nn.Module):
    """ Siren activation function """
    def __init__(self, activation=torch.sin, omega0=30.0, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.omega0     = omega0

    def forward(self, x):
        out = self.activation(self.omega_0*x)
        return out

class MLP(nn.Module):
    def __init__(self, din, dout, depth, width,
                 activation='siren',
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        
        # Define activation function
        if   activation=='siren': 
            self.act_fn = SirenAct
        elif activation=='elu':
            self.act_fn = nn.ELU

        net = []

        # Input layer
        net.append(LinearLayer(din, width, activation, first_omega_0, is_first=True))
        net.append(self.act_fn())

        # Hidden layers
        for _ in range(depth):
            net.append(LinearLayer(width, width, activation, hidden_omega_0))
            net.append(self.act_fn())

        # Output layer
        net.append(LinearLayer(width, dout, activation, hidden_omega_0))
        
        # Create model
        self.model = nn.Sequential(*net)
    
    def forward(self, x):
        return self.model(x)

def nn_grad(out, x):
    ''' Returns gradient of out. One component at a time '''
    return torch.autograd.grad(out, [x],
                               grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]
