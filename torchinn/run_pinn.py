# python3

import torch
import pytorch_lightning as pl
import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import sys
import numpy as np

from pinn import PhysicsInformedNN, nn_grad

# Create data for training
numps = 50
p     = 1.0
idxs  = np.random.choice([True, False], size=numps, p=[p,p-1])
x1    = np.linspace(-1,1,num=numps, dtype=np.float32)[idxs]
x2    = np.linspace(-1,1,num=numps, dtype=np.float32)[idxs]
xt1, xt2 = np.meshgrid(x1, x2, indexing='ij')
xt1 = xt1.flatten().reshape(-1,1)
xt2 = xt2.flatten().reshape(-1,1)

yt1 = np.sin(np.pi*xt1) + np.cos(np.pi*xt2)
yt2 = xt1**3 + xt2**2

X  = np.concatenate((xt1,xt2), 1)
Y  = np.concatenate((yt1,yt2), 1)
xt = torch.from_numpy(X).float()
yt = torch.from_numpy(Y).float()

train = data_utils.TensorDataset(xt, yt)
train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)

# Define pde and PINN
def pde(self, out, x):
    dfdt = nn_grad(out[:,0], x)
    dgdt = nn_grad(out[:,1], x)

    eq1 = dfdt[:,0] - np.pi*torch.cos(np.pi*x[:,0])
    eq2 = self.inv_fields[0](x[:,1:2])[:,0] + np.pi*torch.sin(np.pi*x[:,1])

    eq3 = dgdt[:,0] - self.inv_ctes[1]*x[:,0]**2
    eq4 = dgdt[:,1] - self.inv_ctes[0]*x[:,1]

    return [eq1, eq2, eq3, eq4]
PINN = PhysicsInformedNN(2,2,2,64, pde=pde, inv_ctes=[1.0, 1.0], inv_fields=[(1,1,2,32)], lr=1e-5)

# Create Trainer with checkpointing and logging
checkpoint_callback = ModelCheckpoint(dirpath='ckpt',
                                      save_last=True,
                                      save_top_k=5,
                                      monitor='epoch',
                                      mode='max',
                                      )
logger = CSVLogger(save_dir='.',name='')#, version='')
# Puedo jugar con los loggers y checkpoints, esta bueno para ver en cliente web desde un servidor
trainer = pl.Trainer(max_epochs=150,
                     enable_progress_bar=False,
                     callbacks=[checkpoint_callback],
                     logger=[logger],
                     )

# Train (resuming if possible)
try:
    trainer.fit(model=PINN, train_dataloaders=train_loader, ckpt_path='ckpt/last.ckpt')
except FileNotFoundError:
    trainer.fit(model=PINN, train_dataloaders=train_loader)
