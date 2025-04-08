# State preparation

import keras_tuner
import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import NS3D as Eqs

from   dom import *
from   mod import *
import numpy as np
import time
import random

# Get parameters
params = Run()

# Load data
X_data, Y_data, flags = generate_data(params, '../hit_base/odir')

# Normalization layer
inorm = [X_data.min(0), X_data.max(0)]
means     = Y_data.mean(0)
means[-1] = params.P
stds      = Y_data.std(0)
stds[-1]  = params.sig_p
onorm = [means, stds]

X_plot  = plot_points(params, tidx=5)
path = '../hit_base/odir'
tidx = 5
ref = np.array([abrirbin(f'{path}/v{comp}.{tidx+1:04}.out', params.N)
                for comp in ['x', 'y', 'z']])
validation_data = (X_plot, ref)
def validation(model, validation_data):
    N = 32

    # Get predicted
    X_plot = validation_data[0]
    Y  = model(X_plot)[0].numpy()

    u_p = Y[:,0].reshape((N,N,N))
    v_p = Y[:,1].reshape((N,N,N))
    w_p = Y[:,2].reshape((N,N,N))

    pinn = np.array([u_p, v_p, w_p])
    ref = validation_data[1]
    err  = [np.sqrt(np.mean((ref[ff]-pinn[ff])**2))/np.std(ref[ff]) for ff in range(1)]

    return err[0]

class HyperPINN(keras_tuner.HyperModel):
    def build(self, hp):
        """Builds a PINN model."""
        hu = hp.Int('hu', 100, 200, step=10, default=200)
        layers = hp.Int('layers', 6, 8, step=1, default=8)
        dsteps = hp.Int('dsteps', 18*100, 18*400, step=18*100, default=18*200)
        drate  = hp.Float('drate', 0.9, 0.99, step=0.01, default=0.99)
        lr0    = hp.Float('lr0', 1e-4, 1e-2, sampling='log', default=1e-3)

        # NN params
        layers = [4]+[hu]*layers+[4]

        # Optimizer scheduler
        lr = keras.optimizers.schedules.ExponentialDecay(lr0, dsteps, drate)

        # Initialize model
        nu = 12.5e-3
        eq_params = ([np.float32(nu)])
        eq_params = [np.float32(p) for p in eq_params]
        PINN = PhysicsInformedNN(layers,
                                 norm_in=inorm,
                                 norm_out=onorm,
                                 activation='siren',
                                 optimizer=keras.optimizers.Adam(learning_rate=lr),
                                 restore=False,
                                 eq_params=eq_params)
        self.PINN = PINN
        return PINN.model

    def fit(self, hp, model, X_data, Y_data, validation_data, callbacks=None, **kwargs):
        # Train
        mbsize = hp.Int('mbsize', 5000, 15000, step=1000, default=10000)
        lphys  = hp.Float('lphys', 1e-6, 1e-3, sampling='log', default=1e-3)
        print(f'Iters per epoch: {len(X_data)/mbsize}')
        self.PINN.train(X_data, Y_data,
                        Eqs,
                        epochs=5000,
                        batch_size=mbsize,
                        flags=flags,
                        lambda_phys=lphys,
                        print_freq=999999999,
                        valid_freq=0,
                        save_freq=99999999999,
                        data_mask=(True,True,True,False))

        objective = validation(model, validation_data)
        if np.isnan(objective):
            print('Objective was nan')
            objective=np.inf
        print(objective)
        return objective

tuner = keras_tuner.RandomSearch(
           max_trials=10,
           hypermodel=HyperPINN(),
           directory='results',
           )

tuner.search(X_data, Y_data, validation_data=validation_data)
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)
