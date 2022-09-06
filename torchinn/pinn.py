"""Physics Informed Neural Net Class

Implemented with Pytorch Lightning
"""

import torch
from torch import nn
import pytorch_lightning as pl

from net_archs import MLP


# PINN module
class PhysicsInformedNN(pl.LightningModule):
    '''Physics Informed NN class'''

    def __init__(self,
                 nn_dims,
                 base_nn='mlp',
                 nn_kwargs=None,
                 lphys=1.0,
                 data_mask=None,
                 eq_params=None,
                 inv_ctes=None,
                 inv_fields=None,
                 lr=1e-3):
        super().__init__()

        # Params
        self.nn_dims = nn_dims
        self.base_nn = base_nn
        self.lr = lr

        # Physics hyperparams
        if isinstance(lphys, dict):
            self.lphys = lphys
        else:
            self.lphys = {'value': lphys, 'rule': 'constant'}

        # Choose network type
        if base_nn == 'mlp':
            BaseNN = MLP

        # Main network
        if nn_kwargs is None:
            nn_kwargs = {}
        self.main_net = BaseNN(nn_dims, **nn_kwargs)

        # Datamask
        if data_mask is None:
            self.data_mask = torch.ones(self.main_net.dout)
        else:
            self.data_mask = torch.tensor(data_mask)

        # Eq params and inverse parts
        self.eq_params  = eq_params
        if inv_ctes is not None:
            cte_list        = [nn.Parameter(torch.tensor(k)) for k in inv_ctes]
            self.inv_ctes   = nn.ParameterList(cte_list)
        if inv_fields is not None:
            field_list      = [BaseNN(inv_args, **inv_kwargs)
                               for inv_args, inv_kwargs in inv_fields]
            self.inv_fields = nn.ModuleList(field_list)

        # Save hyperparams
        self.save_hyperparameters()

    def pde(self, out, coords):
        '''pde to enforce'''
        return [torch.tensor([0.0])]

    def forward(self, coords):
        '''forward pass'''
        out = self.main_net(coords)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def training_step(self, batch, batch_idx):
        '''forward pass'''

        # Evaluate model
        x, y = batch
        x = x.clone().detach().requires_grad_(True)
        z = self.main_net(x)

        # Data part
        dy = self.data_mask*y
        dz = self.data_mask*z
        data_loss = nn.functional.mse_loss(dy, dz)
        self.log("data_loss", data_loss)

        # Physics part
        pde = self.pde(z, x)
        phys_loss = torch.stack([eq**2 for eq in pde]).mean()
        self.log("phys_loss", phys_loss)

        # Update lphys
        self.phys_balance(data_loss,
                          phys_loss,
                          batch_idx)

        return data_loss + self.lphys['value']*phys_loss

    def phys_balance(self, data_loss, phys_loss, batch_idx):
        '''Updates self.lphys, the hyperparamter balancing the data and
        physics parts.

        If one is interested in just saving the gradients, it is also possible
        to use the on_before_optimizer_step hook
        '''
        if self.lphys['rule'] == 'adam-like' and batch_idx % 10 == 0:
            grad_data = torch.autograd.grad(data_loss,
                                            self.parameters(),
                                            create_graph=True,
                                            retain_graph=True,
                                            allow_unused=True)
            grad_phys = torch.autograd.grad(phys_loss,
                                            self.parameters(),
                                            create_graph=True,
                                            retain_graph=True,
                                            allow_unused=True)
            sum_data = torch.stack([gd.abs().sum()
                                    for gd in grad_data if gd is not None])
            sum_data = sum_data.sum().detach()
            sum_phys = torch.stack([gp.abs().sum()
                                    for gp in grad_phys if gp is not None])
            sum_phys = sum_phys.sum().detach()

            self.lphys['value'] = (0.9*self.lphys['value'] +
                                   0.1*(sum_data/sum_phys))

        elif self.lphys['rule'] == 'constant':
            pass


def nn_grad(out, x):
    ''' Returns gradient of out. Out must be a scalar '''
    return torch.autograd.grad(out, [x],
                               grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]
