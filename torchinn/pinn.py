# python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

from net_archs import Siren, MLP


# PINN module
class PhysicsInformedNN(pl.LightningModule):
    def __init__(self, din, dout, depth, width,
                 pde=None, eq_params=[], inv_ctes=[], inv_fields=[], lr=1e-3):
        super().__init__()

        self.din   = din
        self.dout  = dout
        self.depth = depth
        self.width = width
        BaseNN = MLP

        # Other params
        self.lr = lr

        # Main network
        self.main_net   = BaseNN(din, dout, depth, width)

        # PDE
        if pde is None:
            self.pde = lambda x,y,z: [torch.tensor([0.0])]
        else:
            self.pde = pde

        # Eq params and inverse parts
        self.eq_params  = eq_params
        cte_list        = [nn.Parameter(torch.tensor(k)) for k in inv_ctes]
        self.inv_ctes   = nn.ParameterList(cte_list)
        field_list      = [BaseNN(inv_in, inv_out, inv_w, inv_d)
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

def nn_grad(out, x):
    ''' Returns gradient of out. One component at a time '''
    return torch.autograd.grad(out, [x],
                               grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]
