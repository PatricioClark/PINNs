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
import spherical
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
dx_plus     = dx/visc_length
dz_plus     = dz/visc_length
y_points    = np.loadtxt('y_points.txt')
y_plus      = y_points/visc_length + 1/visc_length
y0          = 250
z0          = 0
x0          = 1024

# Filter
delta_plus = 40
delta      = delta_plus*visc_length
delta_x    = round(delta_plus/dx_plus)
delta_z    = round(delta_plus/dz_plus)
bound_x    = 4 # approx delta/dx
bound_z    = 8 # approx delta/dz
cut_dims   = [120,10,240]
filt_box   = np.full((delta_x,1,delta_z),fill_value=1.0/(delta_x*1*delta_z))
def les_filter(field):
    return scipy.signal.convolve(field, filt_box, mode='same')

# Iterate through snapshots
for tidx in range(150):
    xe = x0 - round(0.45*dt*tidx/dx)

    # Get velocity field (unfiltered)
    velocity = lJHTDB.getCutout(
        data_set = 'channel',
        field='u',
        time=tidx,
        start = np.array([xe, y0, z0], dtype = np.int),
        size  = np.array(cut_dims,  dtype = np.int),
        step  = np.array([1, 1, 1], dtype = np.int),
        filter_width = 1)

    # Make the shape of velocity equal to (dims,Nx,Ny,Nz)
    velocity = np.transpose(velocity)

    # Filter velocity
    filt_velos = np.array([les_filter(velocity[i]) for i in range(dims)])

    # Remove boundaries affected by the filtering
    filt_velos = filt_velos[:,bound_x:-bound_x,:,bound_z:-bound_z]

    # Sample field at LES resolution
    filt_velos  = filt_velos[:,::bound_x//2,:,::bound_z//2]

    # Save
    np.save('data/filt_velos.{:02}.npy'.format(tidx), filt_velos)
