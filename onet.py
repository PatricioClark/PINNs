# Deep ONet general class

# Requires Python 3.* and Tensorflow 2.0

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time

tf.keras.backend.set_floatx('float64')

class DeepONet:
    """
    General Deep Onet class

    Parameters
    ----------

    dim_f : int [optional]
        Dimensions of input function function (first dimension of branch
        network's input). If value is 1 dimension can be omitted in
        datashape in that case.
    m : int
        Number of sensors (second dimension of branch network's input)
    dim_y : int
        Dimension of y (trunk network's input)
    depth_branch : int
        depth of branch network
    depth_trunk : int
        depth of branch network
    p : int
        Width of branch and trunk networks
    dest : str [optional]
        Path for output files.
    activation : str [optional]
        Activation function to be used. Default is 'relu'.
    optimizer : keras.optimizer instance [optional]
        Optimizer to be used in the gradient descent. Default is Adam with
        fixed learning rate equal to 1e-3.
    norm_in : float or array [optional]
        If a number or an array of size din is supplied, the first layer of the
        network normalizes the inputs uniformly between -1 and 1. Default is
        False.
    norm_out : float or array [optional]
        If a number or an array of size dout is supplied, the layer layer of the
        network normalizes the outputs using z-score. Default is
        False.
    restore : bool [optional]
        If True, it checks if a checkpoint exists in dest. If a checkpoint
        exists it restores the modelfrom there. Default is True.
    """
    # Initialize the class
    def __init__(self,
                 m,
                 dim_f,
                 dim_y,
                 depth_branch,
                 depth_trunk,
                 p,
                 dest='./',
                 regularizer=None,
                 p_drop=0.0,
                 activation='relu',
                 optimizer=keras.optimizers.Adam(lr=1e-3),
                 norm_in=False,
                 norm_out=False,
                 norm_out_type='z-score',
                 restore=True):

        # Numbers and dimensions
        self.dim_f        = dim_f
        self.m            = m
        self.dim_y        = dim_y
        self.depth_branch = depth_branch
        self.depth_trunk  = depth_trunk
        self.width        = p

        # Extras
        self.dest        = dest
        self.regu        = regularizer
        self.norm_in     = norm_in
        self.norm_out    = norm_out
        self.optimizer   = optimizer
        self.activation  = activation

        # Activation function
        if activation=='tanh':
            self.act_fn = keras.activations.tanh
        elif activation=='relu':
            self.act_fn = keras.activations.relu
        elif activation == 'adaptive_global':
            self.act_fn = AdaptiveAct()

        # Inputs definition
        funct = keras.layers.Input((dim_f, m), name='funct')
        point = keras.layers.Input(dim_y,      name='point')

        # Normalize input
        if norm_in:
            fmin   = norm_in[0][0]
            fmax   = norm_in[0][1]
            pmin   = norm_in[1][0]
            pmax   = norm_in[1][1]
            norm_f   = lambda x: 2*(x-fmin)/(fmax-fmin) - 1
            norm_p   = lambda x: 2*(x-pmin)/(pmax-pmin) - 1
            hid_b = keras.layers.Lambda(norm_f)(funct)
            hid_t = keras.layers.Lambda(norm_p)(point)
        else:
            hid_b = funct
            hid_t = point

        # Branch network
        for ii in range(self.depth_branch-1):
            if activation=='adaptive_layer':
                self.act_fn = AdaptiveAct()
            hid_b = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       activation=self.act_fn)(hid_b)
            if p_drop:
                hid_b = keras.layers.Dropout(p_drop)(hid_b)
        hid_b = keras.layers.Dense(self.width,
                                   kernel_regularizer=self.regu)(hid_b)

        # Trunk network
        for ii in range(self.depth_trunk):
            if activation=='adaptive_layer':
                self.act_fn = AdaptiveAct()
            hid_t = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       activation=self.act_fn)(hid_t)
            if p_drop and ii<self.depth_trunk-1:
                hid_t = keras.layers.Dropout(p_drop)(hid_t)

        # Output definition
        output = keras.layers.Dot(axes=-1)([hid_b, hid_t])
        output = BiasLayer()(output)

        if norm_out:
            if norm_out_type=='z_score':
                mm = norm_out[0]
                sg = norm_out[1]
                out_norm = lambda x: sg*x + mm 
            elif norm_out_type=='min_max':
                ymin = norm_out[0]
                ymax = norm_out[1]
                out_norm = lambda x: 0.5*(x+1)*(ymax-ymin) + ymin
            output = keras.layers.Lambda(out_norm)(output)

        # Create model
        model = keras.Model(inputs=[funct, point], outputs=output)
        self.model = model
        self.num_trainable_vars = np.sum([np.prod(v.shape)
                                          for v in self.model.trainable_variables])

        # Create save checkpoints / Load if existing previous
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  self.dest + '/ckpt',
                                                  max_to_keep=5)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

    def train(self,
              Xf, Xp, Y,
              epochs, batch_size,
              loss_fn='mse',
              verbose=False,
              print_freq=1,
              valid_freq=0,
              valid_func=False,
              Xf_test=None, Xp_test=None, Y_test=None,
              save_freq=1,
              timer=False):
        """
        Train function

        Loss functions are written to output.dat

        Parameters
        ----------

        Xf : ndarray
            Input for branch network. Must have shape (:, dim_f, m).
        Xp : ndarray
            Input for trunk network. Must have shape (:, dim_y).
        Y : ndarray
            Data used for training, G(u)(y)
            Must have shape (:, dim_f).
        epochs : int
            Number of epochs to train.
        batch_size : int
            Size of batches.
        loss_fn : str [optional]
            Loss function to be used for training and validation.
        verbose : bool [optional]
            Verbose output or not. Default is False.
        print_freq : int [optional]
            Print status frequency. Default is 1.
        valid_freq : int [optional]
            Validation check frequency. If zero, no validation is performed.
            Default is 0.
        Xf_test : ndarray
            Input for branch network used for testing. Must have shape (:, dim_f, m).
        Xp_test : ndarray
            Input for trunk network used for testing. Must have shape (:, dim_y).
        Y_test : ndarray
            Data used for testing, G(u)(y)
            Must have shape (:, dim_f).
        save_freq : int [optional]
            Save model frequency. Default is 1.
        timer : bool [optional]
            If True, print time per batch for the first 10 batches. Default is
            False.
        """

        len_data = Y.shape[0]
        batches = len_data // batch_size
        idx_arr = np.arange(len_data)

        # Define loss function
        if loss_fn=='mse':
            loss_fn = MSE_loss

        # Run epochs
        ep0 = int(self.ckpt.step)
        for ep in range(ep0, ep0+epochs):
            # Print status
            if ep%print_freq==0:
                try:
                    self.print_status(ep, [loss.numpy()], verbose=verbose)
                except:
                    Y_pred = self.model((Xf_test, Xp_test))
                    loss   = loss_fn(Y_test, Y_pred)
                    self.print_status(ep, [loss.numpy()], verbose=verbose)

                # Print adaptive weights evol
                if self.activation=='adaptive_layer':
                    adps = [v.numpy()[0] for v in self.model.trainable_variables if 'adaptive' in v.name]
                    self.print_status(ep, adps, fname='adpt')

            # Perform validation check
            if valid_freq and ep%valid_freq==0:
                Y_pred = self.model((Xf_test, Xp_test))
                valid  = loss_fn(Y_test, Y_pred)
                self.print_status(ep, [valid.numpy()], fname='valid')

                if valid_func:
                    self.validation(ep)

            # Save progress
            self.ckpt.step.assign_add(1)
            if ep%save_freq==0:
                self.manager.save()
            
            # Loop through batches
            for ba in range(batches):

                # Create batches and cast to TF objects
                Xf_batch, Xp_batch, Y_batch = get_mini_batch(Xf, Xp, Y,
                                                             idx_arr, batch_size)
                Xf_batch = tf.convert_to_tensor(Xf_batch)
                Xp_batch = tf.convert_to_tensor(Xp_batch)
                Y_batch  = tf.convert_to_tensor(Y_batch)

                if timer: t0 = time.time()
                loss = self.training_step(Xf_batch, Xp_batch, Y_batch, loss_fn)
                if timer:
                    print("Time per batch:", time.time()-t0)
                    if ba>10 or ep>5: timer = False

    @tf.function
    def training_step(self, Xf_batch, Xp_batch, Y_batch, loss_fn):
        with tf.GradientTape() as tape:
            Y_pred = self.model((Xf_batch, Xp_batch), training=True)
            loss   = loss_fn(Y_batch, Y_pred)

        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients,
                       self.model.trainable_variables))

        return loss

    def print_status(self, ep, loss, verbose=False, fname='loss'):
        """ Print status function """

        # Loss functions
        output_file = open(self.dest + f'{fname}.dat', 'a')
        print(ep, *loss, file=output_file)
        output_file.close()

        if verbose:
            print(ep, f'{loss}')

def get_mini_batch(X1, X2, Y, idx_arr, batch_size):
    idxs = np.random.choice(idx_arr, batch_size)
    return X1[idxs], X2[idxs], Y[idxs]

class BiasLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

@tf.function
def MSE_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_true - y_pred))

class AdaptiveAct(keras.layers.Layer):
    """ Adaptive activation function """
    def __init__(self, activation=keras.activations.tanh, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def build(self, batch_input_shape):
        ini = keras.initializers.Constant(value=1./10)
        self.a = self.add_weight(name='activation',
                                 initializer=ini,
                                 shape=[1])
        super().build(batch_input_shape)

    def call(self, X):
        aux = tf.multiply(tf.cast(10.0, 'float64'),  self.a)
        return self.activation(tf.multiply(aux, X))

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
