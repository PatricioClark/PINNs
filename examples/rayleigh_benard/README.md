# 2D Rayleigh-Benard example

A PINN that fits snapshots of a 2D Rayleigh-Benard flow while enforcing the
Oberbeck-Boussinesq equations, with `Ra` and `Pr` known. It is a template:
the data format is a placeholder you are expected to adapt.

## The problem

The network maps `(t, x, z)` to `(u, w, T, p)`. The equations (in
`equations.py`, `RB2D`) are in free-fall units (lengths in units of the cell
height, velocities in units of the free-fall velocity, temperature in units of
the temperature difference), with gravity along `-z`:

```
u_x + w_z = 0
u_t + u u_x + w u_z = -p_x + sqrt(Pr/Ra) (u_xx + u_zz)
w_t + u w_x + w w_z = -p_z + sqrt(Pr/Ra) (w_xx + w_zz) + T
T_t + u T_x + w T_z =        1/sqrt(Ra Pr) (T_xx + T_zz)
```

Only `u`, `w` and `T` are fitted to the data. The pressure has no data
(`data_mask=[True, True, True, False]` in `run_pinn.py`) and is learned through
the equations alone, up to a constant. If your data is in other units, change
the coefficients in `RB2D` accordingly.

## Files

- `params.py`: all parameters (data, `Ra` and `Pr`, network, training).
- `equations.py`: the PDE residuals, `RB2D(model, coords, eq_params)`.
- `mod.py`: data loading, model construction and validation.
- `run_pinn.py`: the training script.

## Setting it up

1. Adapt `load_grid` and `load_snapshot` at the top of `mod.py` to your data
   format. Their docstrings say what they must return: the 1D `x` and `z`
   grids, and the time and the `u`, `w`, `T` fields, each of shape
   `(nx, nz)`, of a given snapshot. Nothing else reads the data files.
2. Set `path`, `snapshots`, `Ra` and `Pr` in `params.py`, and the rest as
   needed. The network and training values are reasonable starting points,
   not tuned.

## Running

From this directory, with the *parent* of the `PINNs` repo on the import path
(TensorFlow >= 2.16, see `requirements.txt` at the repo root):

```bash
PYTHONPATH=/path/to/parent/of/PINNs python run_pinn.py
```

A fresh run first fits the data alone for `pre_epochs` epochs, then trains on
data plus physics (weight `lp`) for `epochs` epochs.

## Outputs

Everything goes to `odir/`:

- `output.dat`: `epoch loss_data loss_phys`.
- `validation.dat`: `epoch err_u err_w err_T`, relative L2 errors against
  snapshot `valid_snapshot`.
- `ckpt/`: checkpoints. Running again resumes from the latest one (and skips
  the data-only epochs); delete `odir/` for a fresh run.

To evaluate the trained model, `PINN.model(X)[0]` with `X` of shape `(N, 3)`
gives `(u, w, T, p)`; `mod.plot_points` builds `X` for the full grid at a
snapshot.

## Notes

- Boundary conditions are not enforced explicitly; they enter only through
  the data. Adding them means adding wall points and residuals to `RB2D`.
- `lp` sets the balance between data and physics and usually needs tuning.
  Alternatively, `alpha > 0` in `train` balances them automatically.
