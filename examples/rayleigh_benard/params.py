# Parameters for the 2D Rayleigh-Benard example. Values are illustrative:
# adapt them to your data and problem.

# Data
path = './data'                 # directory read by mod.load_grid / mod.load_snapshot
snapshots = range(0, 100, 5)    # snapshot indices passed to mod.load_snapshot
sample_prob = 0.25              # fraction of points DROPPED at random
valid_snapshot = 50             # snapshot used for validation errors

# Physics (free-fall units, see equations.RB2D)
Ra = 1e6
Pr = 1.0

# Network: (t, x, z) -> (u, w, T, p)
activation = 'siren'
hu = 128                        # units per hidden layer
width = 4                       # number of hidden layers (sic)

# Training
lr = 1e-4
mbsize = 1000                   # batch size
lp = 1e-2                       # physics weight (data weight is 1)
pre_epochs = 2                  # data-only epochs before adding the physics
epochs = 100
