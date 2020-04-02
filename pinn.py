# PINN general class

# Requires Python 3.* and Tensorflow 2.0

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time

tf.keras.backend.set_floatx('float32')
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

    Training can be done usuing the dynamic balance described in
    "Understanding and mitigating gradient pathologies in physics-informed
    neural networks" by Wang, Teng & Perdikaris (2020).

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
    activation : str [optional]
        Activation function to be used. Default is 'tanh'.
    optimizer : keras.optimizer instance [optional]
        Optimizer to be used in the gradient descent. Default is Adam with
        fixed learning rate equal to 5e-4.
    normalize : float or array [optional]
        If a number or an array of size din is supplied, the first layer of the
        networks normalizes the inputs uniformly between -1 and 1. Default is
        False.
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
            norm   = lambda x: 2*(x-xmin)/(xmax-xmin) - 1
            hidden = keras.layers.Lambda(norm)(coords)
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
        self.num_trainable_vars = np.sum([np.prod(v.shape)
                                          for v in self.model.trainable_variables])

        # Parameter for dyncamic balance
        # Can be modified from the outside before calling PINN.train
        self.balance = tf.Variable(1.0, name='balance')

        # Create save checkpoints / Load if existing previous
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           balance=self.balance,
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
                if len(self.inverse[pp][0])==1:
                    ii   = self.inverse[pp][0][0]
                    inps = coords[:,ii:ii+1]
                else:
                    inps   = keras.layers.concatenate(
                           [coords[:,ii:ii+1] for ii in self.inverse[pp][0]])
                if self.normalize:
                    inps = keras.layers.Lambda(self.norm)(inps)
                hidden = inps
                for ii in range(self.inverse[pp][1]):
                    hidden = keras.layers.Dense(self.inverse[pp][2])(hidden)
                    if self.activation=='adaptive_layer':
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
              alpha=0.0,
              full_eqs=True,
              flags=None,
              rnd_order_training=True,
              verbose=False,
              print_freq=1,
              valid_freq=0,
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
        alpha : float [optional]
            If non-zero, performs adaptive balance of the physics and data part
            of the loss functions. See comment above for reference. Default is
            zero.
        full_eqs: bool [optional]
            If True the physics term of the loss is treated as whole. If False,
            each equation is normalized by the means of their gradients.
            Default is True.
        flags: ndarray of ints [optional]
            If supplied, different flags will be used to group different
            points, and then each batch will be formed by picking points from
            each different group, respecting the global ratio of points between
            groups. Default is all points have the same flag.
        rnd_order_training: bool [optional]
            If True points are taking randomly from each group when forming a
            batch, if False points are taking in order, following the order the
            data was supplied. Default is True.
        verbose : bool [optional]
            Verbose output or not. Default is False.
        print_freq : int [optional]
            Print status frequency. Default is 1.
        valid_freq : int [optional]
            Validation check frequency. If zero, no validation is performed.
            Default is 0.
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

        len_data = X_data.shape[0]
        batches = len_data // batch_size

        # Check data_mask
        if data_mask is None:
            data_mask = [True for _ in range(self.dout)]

        # Expand lambdas if necessary
        if not np.shape(lambda_data):
            lambda_data = np.array([lambda_data for _ in range(len_data)])
            lambda_phys = np.array([lambda_phys for _ in range(len_data)])

        # Expand flags
        if flags is None:
            flags = [1 for _ in range(len_data)]
        flags     = np.array(flags)
        flag_idxs = [np.where(flags==f)[0] for f in np.unique(flags)]

        # Cast balance
        balance = tf.constant(self.balance.numpy(), dtype='float32')

        # Run epochs
        ep0     = int(self.ckpt.step)
        for ep in range(ep0, ep0+epochs):
            for ba in range(batches):

                # Create batches and cast to TF objects
                (X_batch,
                 Y_batch,
                 l_data,
                 l_phys) = get_mini_batch(X_data,
                                          Y_data,
                                          lambda_data,
                                          lambda_phys,
                                          ba,
                                          batches,
                                          flag_idxs,
                                          random=rnd_order_training)
                X_batch = tf.convert_to_tensor(X_batch)
                Y_batch = tf.convert_to_tensor(Y_batch)
                l_data = tf.constant(l_data, dtype='float32')
                l_phys = tf.constant(l_phys, dtype='float32')
                ba_counter = tf.constant(ba)

                if timer: t0 = time.time()
                (loss_data,
                 loss_phys,
                 inv_outputs,
                 balance) = self.training_step(X_batch, Y_batch,
                                                   pde,
                                                   l_data,
                                                   l_phys,
                                                   data_mask,
                                                   balance,
                                                   alpha,
                                                   full_eqs,
                                                   ba_counter)
                if timer:
                    print("Time per batch:", time.time()-t0)
                    if ba>10: timer = False

            # Print status
            if ep%print_freq==0:
                self.print_status(ep,
                                  loss_data,
                                  loss_phys,
                                  inv_outputs,
                                  alpha,
                                  verbose=verbose)

            # Perform validation check
            if valid_freq and ep%valid_freq==0:
                self.validation(ep)

            # Save progress
            self.ckpt.step.assign_add(1)
            self.ckpt.balance.assign(balance.numpy())
            if ep%save_freq==0:
                self.manager.save()

    @tf.function
    def training_step(self, X_batch, Y_batch,
                      pde, lambda_data, lambda_phys,
                      data_mask, balance, alpha, full_eqs, ba):
        with tf.GradientTape(persistent=True) as tape:
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
            loss_eqs  = [tf.reduce_mean(
                         lambda_phys*tf.square(eq))
                         for eq in equations]
            loss_phys = tf.add_n(loss_eqs)

            # Total loss function
            loss = loss_data + loss_phys

        # Calculate gradients of data part
        gradients_data = tape.gradient(loss_data,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Calculate gradients of physics part
        if full_eqs:
            gradients_phys = tape.gradient(loss_phys,
                        self.model.trainable_variables,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
        else:
            gradients_eqs  = [tape.gradient(leq,
                        self.model.trainable_variables,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
                        for leq in loss_eqs]
            means = [get_mean_grad(gr, self.num_trainable_vars)
                     for gr in gradients_eqs]
            maxme = tf.reduce_max(means)
            bal_eqs = [maxme/mg for mg in means]
            gradients_eqs  = [[tf.multiply(x,lay) for x,lay in zip(bal_eqs, greq)]
                    for greq in gradients_eqs]
            gradients_phys = [tf.add_n(x) for x in zip(*gradients_eqs)]

        # Delete tape
        del tape

        # If doing dynamic balance, calculate balance
        if alpha>0.0:
            mean_grad_data = get_mean_grad(gradients_data, self.num_trainable_vars)
            mean_grad_phys = get_mean_grad(gradients_phys, self.num_trainable_vars)
            lhat = mean_grad_phys/mean_grad_data
            # if max_grad_phys>0:
            #     lhat = max_grad_phys/mean_grad_data
            # else:
            #     lhat = balance
            balance = (1.0-alpha)*balance + alpha*lhat

        # Apply gradients
        gradients = [x + balance*y for x,y in zip(gradients_phys, gradients_data)]
        self.optimizer.apply_gradients(zip(gradients,
                    self.model.trainable_variables))

        return loss_data, loss_phys, [param[0][0] for param in p_pred], balance

    def print_status(self, ep, lu, lf, inv_outputs, alpha, verbose=False):
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

        # Balance lambda
        if alpha:
            output_file = open(self.dest + 'balance.dat', 'a')
            print(ep, self.balance.numpy(),
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

@tf.function
def get_mean_grad(grads, n):
    ''' Get the mean of the absolute values of the gradient '''
    sum_over_layers = [tf.reduce_sum(tf.abs(gr)) for gr in grads]
    total_sum       = tf.add_n(sum_over_layers)
    return total_sum/n

@tf.function
def get_max_grad(grads):
    max_of_layers = [tf.reduce_max(tf.abs(gr)) for gr in grads]
    total_max       = tf.reduce_max(max_of_layers)
    return total_max

def get_mini_batch(X, Y, ld, lf, ba, batches, flag_idxs, random=True):
    idxs = []
    for fi in flag_idxs:
        if random:
            sl = np.random.choice(fi, len(fi)//batches)
            idxs.append(sl)
        else:
            flag_size = len(fi)//batches
            sl = slice(ba*flag_size, (ba+1)*flag_size)
            idxs.append(fi[sl])
    idxs = np.concatenate(idxs)
    return X[idxs], Y[idxs], ld[idxs], lf[idxs]

class AdaptiveAct(keras.layers.Layer):
    """ Adaptive activation function """
    def __init__(self, activation=keras.activations.tanh, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def build(self, batch_input_shape):
        self.a = self.add_weight(name='activation', shape=[1])
        super().build(batch_input_shape)

    def call(self, X):
        aux = tf.multiply(10.0,  self.a)
        return self.activation(tf.multiply(aux, X))

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
