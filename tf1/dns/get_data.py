# Get velocity field from the JHUTDB

# The velocity fields are filtered and then the exact SGS stresses, the
# filtered viscous stresses, and the Smagorinsky SGS stresses are calculated.

# The filtered velocity fields saved are wider than the stresses because the
# boundaries are needed in the nonlocal integration. The stresses have the same
# horizontal dimensions as those that will be calculated nonlocally.

# import modules
import numpy as np
import pyJHTDB
import scipy.signal
import models
import sys

# Initialize and add token
auth_token  = "edu.jhu.pato-ca56ca00" 
lJHTDB = pyJHTDB.libJHTDB()
lJHTDB.initialize()
lJHTDB.add_token(auth_token)

# Dimensions
dims        = 3
visc_length = 1.0006e-3
dt          = 0.0065
dx          = 8*np.pi/2048
dz          = 3*np.pi/1536 
cut_dims    = [64,8,64]
Nx          = 64
Ny          = 8
Nz          = 64
y0          = 239
z0          = 1
x0          = 1024
xe         = x0 + Nx - 1
ye         = y0 + Ny - 1  
ze         = z0 + Nz - 1

# Iterate through snapshots
for tidx in range(150):
    x0p = x0 - round(0.45*dt*tidx/dx)
    xep = xe - round(0.45*dt*tidx/dx)

    # Get velocity field (unfiltered)
    velocity = lJHTDB.getbigCutout(
        data_set = 'channel',
        fields='u',
        t_start=tidx+1,
        t_end=tidx+1,
        start = np.array([x0p, y0, z0], dtype = np.int),
        end   = np.array([xep, ye, ze], dtype = np.int),
        step  = np.array([1, 1, 1], dtype = np.int),
        filter_width = 1)

    # Make the shape of velocity equal to (dims,Nx,Ny,Nz)
    velocity = np.transpose(velocity)

    # Save
    np.save('data/velos.{:02}.npy'.format(tidx), velocity)

    # Get the pressure field (unfiltered)
    velocity = lJHTDB.getbigCutout(
        data_set = 'channel',
        fields='p',
        t_start=tidx+1,
        t_end=tidx+1,
        start = np.array([x0p, y0, z0], dtype = np.int),
        end   = np.array([xep, ye, ze], dtype = np.int),
        step  = np.array([1, 1, 1], dtype = np.int),
        filter_width = 1)

    # Make the shape of velocity equal to (dims,Nx,Ny,Nz)
    velocity = np.transpose(velocity)

    # Save
    np.save('data/press.{:02}.npy'.format(tidx), velocity)
