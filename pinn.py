# PINN general class

# Requires Python 3.* and Tensorflow 2.0

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time

tf.keras.backend.set_floatx('float64')
class PhysicsInformedNN:
    """
    General PINN class

    Dimensional and problem agnostic implementatation of a Physics Informed
    Neural Netowrk (PINN). Dimensionality is set by specifying the dimensions
    of the inputs and the outputs. The Physiscs is set by writing the partial
    differential equations of the problem.

    This implementation assumes that both args and data are normalized.

    The instantiated model is soted in self.model and it takes coords as inputs
    and outputs a list where the first contains the learned fields and the rest
    of the entries contains the different learned parameters when running an
    inverse PINN or a dummy output that should be disregarded when not.

    A few definitions before proceeding to the initialization parameters:
    
    din   = input dims
    dout  = output dims
    dpar  = number of parameters used by the pde

    Parameters
    ----------

    layers : list
        Shape of the NN. The first element must be din, and the last one must
        be dout.
    dest : str [optional]
        Path for output files.
    eq_params : list [optional]
        List of parameters to be used in pde.
    inverse : list [optional]
        If a list is a supplied the PINN will run the inverse problem, where
        one or more of the paramters of the pde are to be found. The list must
        be of the same length as eq_params and its entries can be False, if that
        parameters is fixed, 'const' if the parameters to be learned is a
        constant (in this case the value provided in eq_params will be used to
        initialize the variable), or a list with two elements, the first a
        tuple indicating which arguments the parameter depends on, and the
        second a list with the shape of the NN to be used to model the hidden
        parameter.
    restore : bool [optional]
        If True, it checks if a checkpoint exists in dest. If a checkpoint
        exists it restores the modelfrom there. Default is True.
    """
    # Initialize the class
    def __init__(self,
                 layers,
                 dest='./',
                 activation='tanh',
                 optimizer=keras.optimizers.Adam(lr=5e-4),
                 normalize=False,
                 eq_params=[],
                 inverse=False,
                 restore=True):

        # Numbers and dimensions
        self.din  = layers[0]
        self.dout = layers[-1]
        depth     = len(layers)-2
        width     = layers[1]

        # Extras
        self.dpar        = len(eq_params)
        self.dest        = dest
        self.eq_params   = eq_params
        self.eval_params = copy.copy(eq_params)
        self.inverse     = inverse
        self.restore     = restore
        self.normalize   = normalize
        self.optimizer   = optimizer
        self.activation  = activation

        # Activation function
        if activation=='tanh':
            self.act_fn = keras.activations.tanh
        elif activation=='relu':
            self.act_fn = keras.activations.relu
        elif activation == 'adaptive_global':
            self.act_fn = AdaptiveAct()

        # Input definition
        coords = keras.layers.Input(self.din, name='coords')

        # Normalzation
        if normalize:
            xmin   = normalize[0]
            xmax   = normalize[1]
            norm   = keras.layers.Lambda(lambda x: 2*(x-xmin)/(xmax-xmin) - 1)
            normed = norm(coords)
            hidden = normed
            self.norm = norm
        else:
            hidden  = coords

        # Hidden layers
        for ii in range(depth):
            hidden = keras.layers.Dense(width)(hidden)
            if activation=='adaptive_layer':
                self.act_fn = AdaptiveAct()
            hidden = self.act_fn(hidden)

        # Output definition
        fields = keras.layers.Dense(self.dout, name='fields')(hidden)

        # Check if inverse problem
        if inverse:
            self.inv_outputs = self.generate_inverse(coords)
        else:
            cte   = keras.layers.Lambda(lambda x: 0*x[:,0:1]+1)(coords)
            dummy = keras.layers.Dense(1, use_bias=False)(cte)
            self.inv_outputs = [dummy]

        # Create model
        model = keras.Model(inputs=coords, outputs=[fields]+self.inv_outputs)
        self.model = model

        # Create save checkpoints / Load if existing previous
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.dest, max_to_keep=5)
        if self.restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

    def generate_inverse(self, coords):
        """
        Generate networks for inverse problem

        This function returns a list whose entries are either False if the
        parameter is fixed, or a neural network that ouputs the parameter to be
        learned.

        Parameters
        ----------

        coords : keras Input layer
            The input layer of the PINN.
        """

        # Create list
        inv_nets = [False]*self.dpar
        inv_outputs = []

        # Iterate through equation parameters
        for pp in range(self.dpar):

            # Hidden parameter is a constant
            if   self.inverse[pp]=='const':
                cte = keras.layers.Lambda(lambda x: 0*x[:,0:1]+1)(coords)
                ini = keras.initializers.Constant(value=self.eq_params[pp])
                hid = keras.layers.Dense(1,
                                         kernel_initializer=ini,
                                         use_bias=False)(cte)
                inv_outputs.append(hid)

            # Hidden parameter is a field
            elif self.inverse[pp]:
                inps   = keras.layers.concatenate(
                           [coords[:,ii:ii+1] for ii in self.inverse[pp][0]])
                if self.normalize:
                    inps = self.norm(inps)
                hidden = inps
                for ii in range(self.inverse[pp][1]):
                    hidden = keras.layers.Dense(self.inverse[pp][2])(hidden)
                    if activation=='adaptive_layer':
                        self.act_fn = AdaptiveAct()
                    hidden = self.act_fn(hidden)
                func = keras.layers.Dense(1)(hidden)
                inv_outputs.append(func)
        return inv_outputs

    def train(self,
              X_data, Y_data,
              pde,
              epochs, batch_size,
              lambda_data=1.0,
              lambda_phys=1.0,
              verbose=False,
              print_freq=1,
              save_freq=1,
              timer=False,
              data_mask=None):
        """
        Train function

        Loss functions are written to output.dat

        Parameters
        ----------

        X_data : ndarray
            Coordinates where the data constraint is going to be enforced.
            Must have shape (:, din).
        Y_data : ndarray
            Data used for the data constraint.
            Must have shape (:, dout). First dimension must be the same as
            X_data. If data constraint is not to be enforced on a certain field
            then the data_mask option should be used.
        pde : function
            Function specifying the equations of the problem. Takes as a
            PhysicsInformedNN class instance, coords and eq_params as inputs.  The
            output of the function must be a list containing all equations.
        epochs : int
            Number of epochs to train
        batch_size : int
            Size of batches
        lambda_data : float or array [optional]
            Weight of the data part of the loss function. If it is an array, it
            should be the same length as X_data, each entry will correspond to
            the particular lambda_data of the corresponding data point.
        lambda_phys : float or array [optional]
            Weight of the physics part of the loss function. If it is an, array
            it should be the same length as X_data, each entry will correspond
            to the particular lambda_phys of the corresponding data point.
        verbose : bool [optional]
            Verbose output or not. Default is False.
        print_freq : int [optional]
            Print status frequency. Default is 1.
        save_freq : int [optional]
            Save model frequency. Default is 1.
        timer : bool [optional]
            If True, print time per batch for the first 10 batches. Default is
            False.
        data_mask : list [optional]
            Determine which output fields to use in the data constraint. Must have
            shape (dout,) with either True or False in each element. Default is
            all True.
        """

        # Check data_mask
        if data_mask is None:
            data_mask = [True for _ in range(self.dout)]

        # Run epochs
        ep0     = int(self.ckpt.step)
        batches = X_data.shape[0] // batch_size
        idxs    = np.arange(X_data.shape[0])
        for ep in range(ep0, ep0+epochs):
            np.random.shuffle(idxs)
            for ba in range(batches):
                sl_ba = slice(ba*batch_size, (ba+1)*batch_size)

                # Create batches and cast to TF objects
                X_batch = X_data[idxs[sl_ba]]
                Y_batch = Y_data[idxs[sl_ba]]
                X_batch = tf.convert_to_tensor(X_batch)
                Y_batch = tf.convert_to_tensor(Y_batch)
                try:
                    l_data = lambda_data[idxs[sl_ba]]
                    l_phys = lambda_phys[idxs[sl_ba]]
                except TypeError:
                    l_data = lambda_data
                    l_phys = lambda_phys
                l_data = tf.constant(l_data, dtype='float64')
                l_phys = tf.constant(l_phys, dtype='float64')

                if timer: t0 = time.time()
                (loss_data,
                 loss_phys,
                 inv_outputs) = self.training_step(X_batch, Y_batch,
                                                   pde,
                                                   l_data,
                                                   l_phys,
                                                   data_mask)
                if timer:
                    print("Time per batch:", time.time()-t0)
                    if ba>10: timer = False

            # Print status
            if ep%print_freq==0:
                self.print_status(ep,
                                  loss_data,
                                  loss_phys,
                                  inv_outputs,
                                  verbose=verbose)
            # Save progress
            self.ckpt.step.assign_add(1)
            if ep%save_freq==0:
                self.manager.save()

    @tf.function
    def training_step(self, X_batch, Y_batch,
                      pde, lambda_data, lambda_phys,
                      data_mask):
        with tf.GradientTape() as tape:
            # Data part
            output = self.model(X_batch)
            Y_pred = output[0]
            p_pred = output[1:]
            aux = [tf.reduce_mean(
                   lambda_data*tf.square(Y_batch[:,ii]-Y_pred[:,ii]))
                   for ii in range(self.dout)
                   if data_mask[ii]]
            loss_data = tf.add_n(aux)

            # Grab inverse coefs
            if self.inverse:
                qq = 0
                for pp in range(self.dpar):
                    if self.inverse[pp]:
                        self.eval_params[pp] = p_pred[qq][:,0]
                        qq += 1

            # Physics part
            equations = pde(self.model, X_batch, self.eval_params)
            aux       = [tf.reduce_mean(
                         lambda_phys*tf.square(eq))
                         for eq in equations]
            loss_phys = tf.add_n(aux)

            # Total loss function
            loss = loss_data + loss_phys

        # Calculate and apply gradients
        gradients = tape.gradient(loss,
                    self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                    self.model.trainable_variables))

        return loss_data, loss_phys, [param[0][0] for param in p_pred]

    def print_status(self, ep, lu, lf, inv_outputs, verbose=False):
        """ Print status function """

        # Loss functions
        output_file = open(self.dest + 'output.dat', 'a')
        print(ep, f'{lu}', f'{lf}',
              file=output_file)
        output_file.close()
        if verbose:
            print(ep, f'{lu}', f'{lf}')

        # Inverse coefficients
        if self.inverse:
            output_file = open(self.dest + 'inverse.dat', 'a')
            if self.inverse:
                print(ep, *[pp.numpy() for pp in inv_outputs],
                      file=output_file)
            output_file.close()

    def grad(self, coords):
        """
        Neural network gradient

        Returns the gradient of the Neural Network stored in model according to its
        inputs.

        Parameters
        ----------

        coords : Input tensor
            Tensor with the coordinates in which to evaluate the gradient

        The returned gradient is such that
            df_i/dx_j = gradient[i][:,j]
        where the final length is equal to coords.shape[0] (i.e, the number of
        points where the gradient is evaluated).

        Returns
        -------

        df : Tensor or array
            Gradient of the NN
        d2f : Tensor or array
            Hessian of the NN
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(coords)
            Yp = self.model(coords)[0]
            fields = [Yp[:,ii] for ii in range(self.din)]
        df = [tape.gradient(fields[jj], coords) for jj in range(self.dout)]
        del tape

        return Yp, df

    def grad_and_hess(self, coords):
        """
        Neural network gradient and hessian

        Returns the gradient and hessian of the Neural Network stored in model
        according to its inputs.

        The returned gradient is such that
            df_i/dx_j = gradient[i][:,j]
        where the final length is equal to coords.shape[0] (i.e, the number of
        points where the gradient is evaluated).

        The returned hessian is such that
            d^2f_i/(dx_j dx_k) = hessian[i][j][:,k]
        where the final length is equal to coords.shape[0] (i.e, the number of
        points where the gradient is evaluated).

        Parameters
        ----------

        coords : Input tensor
            Tensor with the coordinates in which to evaluate the gradient

        Returns
        -------

        df : Tensor or array
            Gradient of the NN
        d2f : Tensor or array
            Hessian of the NN
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(coords)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(coords)
                Yp = self.model(coords)[0]
                fields = [Yp[:,ii] for ii in range(self.din)]
            df = [tape1.gradient(fields[jj], coords)  for jj in range(self.dout)]
            gr = [[df[jj][:,ii] for ii in range(self.din)] for jj in range(self.dout)]
        d2f = [[tape2.gradient(gr[jj][ii], coords)
                for ii in range(self.din)]
                for jj in range(self.dout)]
        del tape1
        del tape2

        return Yp, df, d2f

class AdaptiveAct(keras.layers.Layer):
    """ Adaptive activation function """
    def __init__(self, activation=keras.activations.tanh, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def build(self, batch_input_shape):
        self.a = self.add_weight(name='activation', shape=[1])
        super().build(batch_input_shape)

    def call(self, X):
        return self.activation(self.a * X)

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
