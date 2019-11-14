# Functions used by pinn.py

# Disables warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def xavier_init(size):
    '''Initialization function'''
    in_dim  = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                                  stddev=xavier_stddev,
                                                  dtype=tf.float64,seed=None),
                       dtype=tf.float64)
    
def DNN(X, layers,weights, biases, act='relu'):
    """A fully-connected NN"""
    if  act=='relu':
        act_fn = tf.nn.relu
    elif act=='tanh':
        act_fn = tf.nn.tanh
    L = len(layers)
    H = X 
    for l in range(0,L-2):
        W = weights[l]
        b = biases[l]
        H = act_fn(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    loss_reg = 0.0
    # for item in weights:
    #     loss_reg = loss_reg + tf.nn.l2_loss(item)
    return Y, loss_reg

def generate_data(Nt, Nx, Ny, Nz, sample_prob):
    """Populate data arrays"""
    points = np.array([(ix,iy,iz) for ix in range(Nx)
                                  for iy in range(Ny)
                                  for iz in range(Nz)])
    probs  = np.array([np.random.random()<sample_prob for i in range(Nx*Ny*Nz)])
    points = points[probs]

    dt       = 0.0065
    dx       = 8*np.pi/2048
    dz       = 3*np.pi/1536
    y0       = 239
    y_points = np.loadtxt('y_points.txt')[y0:y0+Ny]
    t_d, x_d, y_d, z_d = [], [], [], []
    u_d, v_d, w_d      = [], [], []
    for tidx in range(Nt):
        velocity = np.load('data/velos.{:02}.npy'.format(tidx))
        i0 = round(0.45*dt*tidx/dx)
        for pp in points:
            ix, iy, iz = pp
            t_d.append(tidx*dt)
            x_d.append((ix-i0)*dx + 0.45*tidx*dt)
            y_d.append(y_points[iy])
            z_d.append(iz*dz)
            u_d.append(velocity[0,ix,iy,iz])
            v_d.append(velocity[1,ix,iy,iz])
            w_d.append(velocity[2,ix,iy,iz])

    # Convert into arrays with correct shape
    t_d = np.array(t_d).reshape(-1,1)
    x_d = np.array(x_d).reshape(-1,1)
    y_d = np.array(y_d).reshape(-1,1)
    z_d = np.array(z_d).reshape(-1,1)
    u_d = np.array(u_d).reshape(-1,1)
    v_d = np.array(v_d).reshape(-1,1)
    w_d = np.array(w_d).reshape(-1,1)

    return t_d, x_d, y_d, z_d, u_d, v_d, w_d

def plot_points(Nx, Ny, Nz, tidx=0):
    """Populate plotting points arrays"""
    points = np.array([(ix,iy,iz) for ix in range(Nx)
                                  for iy in range(Ny)
                                  for iz in range(Nz)])

    dt       = 0.0065
    dx       = 8*np.pi/2048
    dz       = 3*np.pi/1536
    y0       = 239
    y_points = np.loadtxt('y_points.txt')[y0:y0+Ny]
    t_d, x_d, y_d, z_d = [], [], [], []
    i0 = round(0.45*dt*tidx/dx)
    for pp in points:
        ix, iy, iz = pp
        t_d.append(tidx*dt)
        x_d.append((ix-i0)*dx + 0.45*tidx*dt)
        y_d.append(y_points[iy])
        z_d.append(iz*dz)

    # Convert into arrays with correct shape
    t_d = np.array(t_d).reshape(-1,1)
    x_d = np.array(x_d).reshape(-1,1)
    y_d = np.array(y_d).reshape(-1,1)
    z_d = np.array(z_d).reshape(-1,1)

    return t_d, x_d, y_d, z_d

def param_number(layers):
    L = len(layers)
    weights = ([layers[l]*layers[l+1] + layers[l+1]
                for l in range(0, L-1)])
    return sum(weights)
