# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-module library for Physics-Informed Neural Networks (Raissi et al. 2019) in TensorFlow/Keras 3, geared towards inverse problems. All the code is `pinn.py`; it exports one class, `PhysicsInformedNN`. Target stack is TensorFlow >= 2.16 (Keras 3). See `requirements.txt`.

The repo directory itself is the package: `__init__.py` re-exports the class, so client code does `from PINNs import PhysicsInformedNN` with the *parent* of this directory on the import path.

## Running the examples

There is no test framework. `tests/test.py` and `tests/test_func.py` are end-to-end example scripts (a 2D toy problem with two unknown PDE parameters; `test_func` makes one of them a learned function of x_2). To run one:

```bash
pip install -r requirements.txt
mkdir -p tests/odir                     # output dir must exist; it is gitignored
cd tests && PYTHONPATH=../.. MPLBACKEND=Agg python test.py
```

- Run from inside `tests/`: the scripts write to `./odir/` relative to the cwd and read the loss history back from there.
- `PYTHONPATH=../..` makes `PINNs` importable as a package.
- `MPLBACKEND=Agg` avoids the blocking `plt.show()` at the end; figures are saved to `odir/fig_*.png` regardless.
- The scripts train for 400 epochs (`tot_eps`); lower it for a quick smoke run.

`tuner/` and `alternatives/` are kept for reference but are **not maintained**: the tuner scripts depend on modules and data outside this repo and on an older constructor API, and the owner has chosen to leave them as they are. Do not fix or delete them unless asked.

## Architecture

### Composition, not inheritance

`PhysicsInformedNN` *wraps* a Keras functional model (`self.model`) rather than subclassing `keras.Model`. This is a deliberate decision: the training loop needs two separate gradient computations, per-point loss weights, stratified batching and a user-supplied PDE callable, none of which fit `fit()`/`train_step` cleanly, and the thin wrapper limits exposure to Keras API churn. Do not refactor towards a `keras.Model` subclass.

### Model output contract

`self.model(coords)` returns a **list**:

- `out[0]` — the learned fields, shape `(N, dout)`.
- `out[1:]` — one tensor per entry of the `inverse` list, in order, each shape `(N, 1)`. If `inverse` is `None`, a single untrained `dummy` output is appended instead so the list structure is stable.

Every PDE callable relies on this. The PDE function has signature `pde(model, coords, eq_params) -> list_of_residuals`, is called inside the training step, and typically opens its own `tf.GradientTape` on `coords` to get derivatives of `out[0]`. See `tests/test.py::some_eqs` for the canonical example.

Inverse constants are implemented as a `Dense(1)` layer applied to zeroed coordinates, so the layer's *bias* is the learned constant (initialised from `value`). Inverse functions get their own sub-network built by `_generate_network`, optionally with a `mask` that zeroes the coordinates they must not depend on. Input normalisation and feature expansion are applied *before* both the main network and the inverse networks.

### Training step (`_training_step`, `@tf.function`)

1. One persistent tape computes `loss_data` (masked MSE on `out[0]`) and `loss_phys` (MSE of the PDE residuals), each weighted by per-point `lambda_data` / `lambda_phys`.
2. Gradients of the two losses are taken **separately**, then combined as `g_data + bal_phys * g_phys`.
3. `None` gradients are replaced with zeros by hand. This is a workaround for a TF >= 2.16 change that broke `UnconnectedGradients.ZERO`; the dummy output and the kernel of the constant-inverse `Dense` always have unconnected gradients, so **do not revert this to the `ZERO` flag.**
4. If `alpha > 0`, `bal_phys` is updated as an exponential moving average of the ratio of mean absolute gradients (Wang, Teng & Perdikaris 2020). `bal_phys` lives in a `tf.Variable` that is checkpointed, so the balance survives restarts.

### Batching

`train(..., batch_size, flags=...)` computes `batches = len_data // batch_size` and calls `get_mini_batch` once per batch with the **batch count**, not the batch size. The sampler draws `len(group) // batches` points from *each* flag group per batch, so an epoch covers every group once (in ordered mode) and batches preserve the global group ratios. Consequences: the actual batch size can be a point or two below `batch_size` due to integer division, and a group with fewer points than `batches` contributes nothing.

### Persistence and outputs

Everything goes under `dest`:

- `output.dat` — `epoch loss_data loss_phys`, appended once per `print_freq` epochs.
- `inverse.dat` — epoch followed by the current value of each `'const'` inverse parameter.
- `balance.dat` — epoch and `bal_phys`, only when `alpha > 0`.
- `ckpt/` — `tf.train.Checkpoint` of model, optimizer, `bal_phys` and an epoch counter, via a `CheckpointManager`.

`restore=True` is the **default**: constructing the class with an existing `ckpt/` in `dest` silently resumes from it, and the epoch counter continues across successive `train()` calls. Pass `restore=False` for a fresh run. The normalisation layers are `Lambda` closures, so `model.save()` will not serialise them cleanly; the checkpoint is the supported persistence path.

### Activations

`activation` may be a string (`'tanh'`, `'relu'`, `'elu'`, `'siren'`) or a dict `{'type': ..., **overrides}` where any extra key overrides the defaults built in `_generate_activation` (e.g. `kinit`, or `first_omega0` / `hidden_omega0` for SIREN). SIREN re-derives the kernel initialiser per layer from `omega0` inside `_generate_network`.
