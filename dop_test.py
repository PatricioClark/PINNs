# DeepONet test

import time
import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from onet import DeepONet

def get_data(ell, m, num):
    try:
        X1_train = np.load('X1_train.npy')
        X2_train = np.load('X2_train.npy')
        Y_train = np.load('Y_train.npy')
        X1_test  = np.load('X1_test.npy')
        X2_test  = np.load('X2_test.npy')
        Y_test  = np.load('Y_test.npy')
    except:
        t0 = time.time()
        X1_train, X2_train, Y_train = generate_data(ell, m, num_training)
        print('Time for generation:', time.time()-t0)
        X1_test,  X2_test,  Y_test  = generate_data(ell, m, num_testing)
        np.save('X1_train.npy', X1_train)
        np.save('X2_train.npy', X2_train)
        np.save('Y_train.npy', Y_train)
        np.save('X1_test.npy',  X1_test)
        np.save('X2_test.npy',  X2_test)
        np.save('Y_test.npy',  Y_test)
    W_train = np.ones(np.shape(Y_train))
    W_test = np.ones(np.shape(Y_train))
    return (X1_train, X2_train, Y_train, W_train,
            X1_test,  X2_test,  Y_test,  W_test)

def generate_data(ell, m, num):
    # Specify Gaussian Process
    kernel = RBF(length_scale=ell)
    gp = GaussianProcessRegressor(kernel=kernel)

    # Create sensors
    sensors = np.linspace(0, 1, num=m)[:, None]

    # Create u's
    u_samples = gp.sample_y(sensors, num, None).T

    # Create y's
    y_samples = np.random.rand(num)[:, None]

    # Get G(u)(y)
    u_funcs = [interpolate.interp1d(sensors[:,0],
                                    sample,
                                    kind='cubic',
                                    copy=False,
                                    assume_sorted=True)
                                    for sample in u_samples]

    solutions = [solve_ivp(lambda y,s: f(y), [0, yf[0]], [0], method="RK45").y[0,-1:]
                 for f, yf in zip(u_funcs, y_samples)]
    solutions = np.array(solutions)

    return u_samples, y_samples, solutions

# Parameters
ell = 0.2
m   = 100
num_training = 10000
num_testing  = 10000
epochs       = 50000
bsize        = 10000

(Xf_train,
 Xp_train,
 Y_train,
 W_train,
 Xf_test,
 Xp_test,
 Y_test,
 W_test)= get_data(ell, m, num_testing)

# Initialize
donet = DeepONet(m=m, dim_y=1, depth_branch=2, depth_trunk=2, p=40)

# Train
donet.train(Xf_train, Xp_train, Y_train,
        epochs=epochs, batch_size=bsize,
        verbose=True,
        timer=True,
        print_freq=1000,
        save_freq=1000,
        valid_freq=1000,
        Xf_test=Xf_test,
        Xp_test=Xp_test,
        Y_test=Y_test,
        W_test=W_test)

# Test example
ii = 20
kernel = RBF(length_scale=ell)
gp = GaussianProcessRegressor(kernel=kernel)

sensors = np.linspace(0, 1, num=m)[:, None]
ys      = sensors

u = gp.sample_y(sensors, 1, None).T
u = Xf_train[ii].reshape(1,-1)
u_func = interpolate.interp1d(sensors[:,0],
                                u,
                                copy=False,
                                assume_sorted=True)
f = lambda y,s: u_func(y)

solutions = solve_ivp(f, [0, 1], [0], method="RK45", t_eval=ys[:,0]).y[0,:]
solutions = np.array(solutions)

dy = np.diff(ys[:,0])[10]
deriv = np.gradient(solutions, dy)

us = np.concatenate([u for _ in range(m)])
pred = donet.model((us, ys))

Y_pred = donet.model((Xf_train, Xp_train))

plt.figure(1)
plt.clf()
plt.plot(sensors, u[0], label='u')
plt.plot(sensors, solutions, label='s')
plt.plot(sensors, pred, label='Onet')
plt.plot(sensors, deriv, label='deriv')
plt.plot(Xp_train[ii], Y_train[ii], 'ro')
plt.plot(Xp_train[ii], Y_pred[ii], 'go')
plt.legend()
plt.draw()
plt.show()
