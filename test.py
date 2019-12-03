# PINN 2.0 test

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys

from pinn import PhysicsInformedNN
    
# -----------------------------------------------------------------------------
# Equations to enforce
# -----------------------------------------------------------------------------

@tf.function
def some_eqs(model, coords, params):
    dout = 2
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(coords)
        Yp = model(coords)[0]
        fields = [Yp[:,jj] for jj in range(dout)]
    df = [tape.gradient(fields[jj], coords) for jj in range(dout)]
    df1dx1 = df[0][:,0]
    df1dx2 = df[0][:,1]
    df2dx1 = df[1][:,0]
    df2dx2 = df[1][:,1]
    del tape
    
    eq1 = df1dx1 - np.pi*tf.cos(np.pi*coords[:,0])
    eq2 = df1dx2 + np.pi*tf.sin(np.pi*coords[:,1])

    # params[0] should be equal to 2
    # params[1] should be equal to 3
    eq3 = df2dx1 - params[1]*coords[:,0]**2
    eq4 = df2dx2 - params[0]*coords[:,1]
    return [eq1, eq2, eq3, eq4]

# -----------------------------------------------------------------------------
# Generate data
# -----------------------------------------------------------------------------

# For training
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

X = np.concatenate((xt1,xt2), 1)
Y = np.concatenate((yt1,yt2), 1)

# For plotting
x1 = np.linspace(-1,1,num=numps, dtype=np.float32)
x2 = np.linspace(-1,1,num=numps, dtype=np.float32)
x1 = x1.reshape(-1,1)
x2 = x2.reshape(-1,1)
zs = np.zeros(numps, dtype=np.float32).reshape(-1,1)

# -----------------------------------------------------------------------------
# Initialize PINN
# -----------------------------------------------------------------------------
layers  = [2] + 2*[64] + [2]
PINN = PhysicsInformedNN(layers,
                         eq_params=[2,3],
                         inverse=['const', False],
                         restore=False)

# -----------------------------------------------------------------------------
# Train PINN
# -----------------------------------------------------------------------------
PINN.train(X, Y, some_eqs, epochs=1, batch_size=32, verbose=False)

t0 = time.time()
PINN.train(X, Y, some_eqs, epochs=100, batch_size=32, verbose=False)
print('Time per epoch:', (time.time()-t0)/100)

# -----------------------------------------------------------------------------
# Plot and validate
# -----------------------------------------------------------------------------

prefix = 'fig'

# Plot loss functions
ep, lu, lf = np.loadtxt('output.dat', unpack=True)

ep, c1 = np.loadtxt('inverse.dat', unpack=True)

plt.figure(0)
plt.plot(ep, lu, label='Data loss')
plt.plot(ep, lf, '--', label='Eqs loss')
plt.legend()
plt.savefig(prefix+'_0')

plt.figure(10)
plt.plot(ep, c1, label='Parameter')
plt.axhline(2,color='k',ls='--',label='Real value')
plt.legend()
plt.savefig(prefix+'_10')

# Evalute along x_2=0
X      = np.concatenate((x1,zs), 1)
fields = PINN.model(X)[0]

plt.figure(1)
plt.clf()
plt.plot(x1,fields[:,0])
plt.plot(x1,np.sin(np.pi*x1)+1,'ro')
plt.title(r'$f_1(x_1,0)$')
plt.savefig(prefix+'_1')

plt.figure(3)
plt.clf()
plt.plot(x1,fields[:,1])
plt.plot(x1,x1**3,'ro')
plt.title(r'$f_2(x_1, 0)$')
plt.savefig(prefix+'_3')

# Evalute along x_1=0
X      = np.concatenate((zs, x2), 1)
fields = PINN.model(X)[0]

plt.figure(2)
plt.clf()
plt.plot(x2,fields[:,0])
plt.plot(x2,np.cos(np.pi*x2),'ro')
plt.title(r'$f_1(0, x_2)$')
plt.savefig(prefix+'_2')

plt.figure(4)
plt.clf()
plt.plot(x2,fields[:,1])
plt.plot(x2,x2**2,'ro')
plt.title(r'$f_2(0, x_2)$')
plt.savefig(prefix+'_4')

# Get gradients

# First convert to tensors
x1  = tf.convert_to_tensor(x1)
x2  = tf.convert_to_tensor(x2)
zs  = tf.convert_to_tensor(zs)

# Make tape
with tf.GradientTape(persistent=True) as tape:
    tape.watch([x1,zs])
    X       = tf.concat((x1,zs), 1)
    tot     = PINN.model(X)[0]
    pred1   = tot[:,0]
    pred2   = tot[:,1]
g11 = tape.gradient(pred1,  x1)
g21 = tape.gradient(pred2,  x1)
del tape

plt.figure(5)
plt.clf()
plt.plot(x1, g11)
plt.plot(x1, np.pi*np.cos(np.pi*x1),'ro')
plt.title(r'$\frac{\partial f_1}{\partial x_1}(x_1,0)$')
plt.savefig(prefix+'_5')

plt.figure(7)
plt.clf()
plt.plot(x1, g21)
plt.plot(x1, 3*x1**2,'ro')
plt.title(r'$\frac{\partial f_2}{\partial x_1}(x_1,0)$')
plt.savefig(prefix+'_7')

with tf.GradientTape(persistent=True) as tape:
    tape.watch([x2,zs])
    X       = tf.concat((zs,x2), 1)
    tot     = PINN.model(X)[0]
    pred1   = tot[:,0]
    pred2   = tot[:,1]
g12 = tape.gradient(pred1,  x2)
g22 = tape.gradient(pred2,  x2)
del tape

plt.figure(6)
plt.clf()
plt.plot(x2, g12)
plt.plot(x2, -np.pi*np.sin(np.pi*x2),'ro')
plt.title(r'$\frac{\partial f_1}{\partial x_2}(0, x_2)$')
plt.savefig(prefix+'_6')

plt.figure(8)
plt.clf()
plt.plot(x2, g22)
plt.plot(x2, 2*x2,'ro')
plt.title(r'$\frac{\partial f_2}{\partial x_2}(0, x_2)$')
plt.savefig(prefix+'_8')

plt.draw()
plt.show()
