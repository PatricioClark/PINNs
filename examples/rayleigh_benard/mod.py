import os
import numpy as np

from PINNs import PhysicsInformedNN

# -----------------------------------------------------------------------------
# Data loading. These two functions are the only ones that know the data
# format: adapt them to yours. Everything else uses what they return.
# -----------------------------------------------------------------------------

def load_grid(path):
    '''Placeholder: adapt to your data.

    Returns x and z, 1D arrays with the grid points in each direction (the z
    grid may be non-uniform, e.g. Chebyshev).
    '''
    x = np.load(f'{path}/x.npy')
    z = np.load(f'{path}/z.npy')
    return x, z

def load_snapshot(path, ii):
    '''Placeholder: adapt to your data.

    Returns the time t of snapshot ii and the fields u, w and T, each an
    array of shape (nx, nz) on the grid given by load_grid.
    '''
    data = np.load(f'{path}/snapshot.{ii:04}.npz')
    return float(data['t']), data['u'], data['w'], data['T']

# -----------------------------------------------------------------------------

def generate_data(params, path):
    '''Training points: coords (t, x, z) and fields (u, w, T, p).

    The data has no pressure, so p is a zero placeholder that is never used
    (data_mask=[True, True, True, False] in train). A fraction
    params.sample_prob of the points is dropped at random.
    '''
    x, z = load_grid(path)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    xx = xx.flatten()
    zz = zz.flatten()

    coords = []
    fields = []
    for ii in params.snapshots:
        t, u, w, T = load_snapshot(path, ii)
        keep = np.random.rand(len(xx)) > params.sample_prob
        tt = np.full(len(xx), t)
        coords.append(np.stack([tt, xx, zz], 1)[keep])
        fields.append(np.stack([u.flatten(), w.flatten(), T.flatten(),
                                np.zeros(len(xx))], 1)[keep])

    coords = np.concatenate(coords).astype(np.float32)
    fields = np.concatenate(fields).astype(np.float32)
    return coords, fields

def build_model(params, inorm, onorm):
    '''(t, x, z) -> (u, w, T, p)'''
    layers = [3] + params.width*[params.hu] + [4]
    model = PhysicsInformedNN(layers,
                              dest='./odir/',
                              activation=params.activation,
                              norm_in=inorm,
                              norm_out=onorm,
                              )
    return model

def plot_points(path, ii):
    '''Coords (t, x, z) of the full grid at snapshot ii'''
    x, z = load_grid(path)
    t, _, _, _ = load_snapshot(path, ii)
    tt, xx, zz = np.meshgrid(t, x, z, indexing='ij')
    X = np.stack([tt.flatten(), xx.flatten(), zz.flatten()], 1)
    return X.astype(np.float32)

def cte_validation(self, params, path, ii):
    '''Relative L2 errors of u, w and T against snapshot ii, appended to
    validation.dat in dest as "ep err_u err_w err_T".'''
    X = plot_points(path, ii)
    _, u, w, T = load_snapshot(path, ii)
    ref = [u.flatten(), w.flatten(), T.flatten()]

    def validation(ep):
        Y = self.model(X)[0].numpy()
        errs = [np.linalg.norm(Y[:,jj]-ref[jj])/np.linalg.norm(ref[jj])
                for jj in range(3)]
        with open(os.path.join(self.dest, 'validation.dat'), 'a') as fh:
            print(ep, *errs, file=fh)

    return validation
