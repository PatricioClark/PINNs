"""Run a PINN example"""

import torch
import torch.utils.data as data_utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import numpy as np

from pinn import PhysicsInformedNN, nn_grad

# Create data for training
NUMPS = 50
x1    = np.linspace(-1, 1, num=NUMPS, dtype=np.float32)
x2    = np.linspace(-1, 1, num=NUMPS, dtype=np.float32)
zs    = np.zeros(NUMPS, dtype=np.float32).reshape(-1, 1)
xt1, xt2 = np.meshgrid(x1, x2, indexing='ij')
xt1 = xt1.flatten().reshape(-1, 1)
xt2 = xt2.flatten().reshape(-1, 1)

yt1 = np.sin(np.pi*xt1) + np.cos(np.pi*xt2)
yt2 = xt1**3 + xt2**2

X  = np.concatenate((xt1, xt2), 1)
Y  = np.concatenate((yt1, yt2), 1)
xt = torch.from_numpy(X).float()
yt = torch.from_numpy(Y).float()

training_data = data_utils.TensorDataset(xt, yt)
train_loader  = data_utils.DataLoader(training_data,
                                      batch_size=32,
                                      shuffle=True)


# Define pde and PINN
class TestPINN(PhysicsInformedNN):
    '''Test PINN'''
    # pylint: disable=too-many-ancestors
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pde(self, out, coords):
        dfdt = nn_grad(out[:, 0], coords)
        dgdt = nn_grad(out[:, 1], coords)
        invf = self.inv_fields[0](coords)

        eq1 = dfdt[:, 0] - np.pi*torch.cos(np.pi*coords[:, 0])
        eq2 = invf[:, 0] + np.pi*torch.sin(np.pi*coords[:, 1])
        eq3 = dgdt[:, 0] - self.inv_ctes[1]*coords[:, 0]**2
        eq4 = dgdt[:, 1] - self.inv_ctes[0]*coords[:, 1]

        return [eq1, eq2, eq3, eq4]


def main():
    """main train function"""
    # Instantiate model
    momegas = {'mask': [0, 1], 'first_omega_0': 10.0, 'hidden_omega_0': 10.0}
    omegas = {'first_omega_0': 10.0, 'hidden_omega_0': 10.0}
    PINN = TestPINN([2, 2, 1, 64],
                    inv_ctes=[1.0, 1.0],
                    inv_fields=[([2, 1, 1, 32], momegas)],
                    base_nn='siren',
                    nn_kwargs=omegas,
                    lr=1e-5)

    # Create Trainer with checkpointing and logging
    checkpoint_callback = ModelCheckpoint(dirpath='ckpt',
                                          save_last=True,
                                          save_top_k=5,
                                          monitor='epoch',
                                          mode='max',
                                          )
    logger = CSVLogger(save_dir='logs', name='')
    trainer = pl.Trainer(max_epochs=200,
                         enable_progress_bar=False,
                         callbacks=[checkpoint_callback],
                         logger=[logger],
                         )

    # Train
    trainer.fit(model=PINN, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
