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
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-ancestors

    def __init__(self,
                 nn_dims,
                 nn_kwargs=None,
                 data_mask=None,
                 base_nn='mlp',
                 eq_params=None,
                 inv_ctes=None,
                 inv_fields=None,
                 lr=1e-3):
        # pylint: disable=too-many-arguments
        super().__init__()

        # Params
        self.nn_dims = nn_dims
        self.base_nn = base_nn
        self.lr = lr

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
        # pylint: disable=unused-argument
        return [torch.tensor([0.0])]

    def forward(self, coords):
        '''forward pass'''
        # pylint: disable=arguments-differ
        out = self.main_net(coords)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        '''forward pass'''
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        x, y = batch
        x = x.clone().detach().requires_grad_(True)
        z = self.main_net(x)
        dy = self.data_mask*y
        dz = self.data_mask*z
        data_loss = nn.functional.mse_loss(dy, dz)
        self.log("data_loss", data_loss)

        pde = self.pde(z, x)
        phys_loss = torch.stack([eq**2 for eq in pde]).mean()
        self.log("phys_loss", phys_loss)

        total_loss = data_loss + phys_loss
        return total_loss


def nn_grad(out, x):
    ''' Returns gradient of out. Out must be a scalar '''
    return torch.autograd.grad(out, [x],
                               grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]
