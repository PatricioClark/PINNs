# 2D Rayleigh-Benard PINN

import os
import numpy as np

from equations import RB2D as Eqs

from mod import generate_data, build_model, cte_validation

import params

# Output directory (loss history, validation and checkpoints)
os.makedirs('odir', exist_ok=True)

# Load data: X = (t, x, z), Y = (u, w, T, p)
X_data, Y_data = generate_data(params, params.path)

# Normalization layers. There is no pressure data (the p column is zero), so
# its std would be 0 and the network could only output p = 0: use mean 0 and
# std 1 for p instead.
inorm = [X_data.min(0), X_data.max(0)]
means = Y_data.mean(0)
stds  = Y_data.std(0)
means[3] = 0.0
stds[3]  = 1.0
onorm = [means, stds]

# Initialize model. If odir/ckpt exists the model resumes from it
PINN = build_model(params, inorm, onorm)
PINN.optimizer.learning_rate.assign(params.lr)

# Validation function
PINN.validation = cte_validation(PINN, params, params.path, params.valid_snapshot)

# Train
eq_params = [np.float32(params.Ra), np.float32(params.Pr)]
def block(lambda_phys, epochs):
    PINN.train(X_data,
               Y_data,
               Eqs,
               epochs=epochs,
               batch_size=params.mbsize,
               eq_params=eq_params,
               lambda_data=1.0,
               lambda_phys=lambda_phys,
               print_freq=1,
               valid_freq=1,
               save_freq=1,
               data_mask=[True,True,True,False],   # no data for p
               )

if not PINN.manager.latest_checkpoint:
    block(0.0, params.pre_epochs)                  # data only, fresh run only
block(params.lp, params.epochs)                    # data + physics
