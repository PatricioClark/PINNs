#!/usr/bin/python3
# -*- coding: utf-8 -*-

# DeepONet general class
# Written by Patricio Clark Di Leoni at Johns Hopkins University
# December 2020

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time

tf.keras.backend.set_floatx('float64')

class DeepONet:
    """
    General DeepONet class

    The class creates a keras.Model with a branch and trunk networks.

    While the class constains a custom train method, it's adivised to use the
    builtin fit method from keras.Model.

    The class also creates a checkpoint and a logger callback to be used during
    training.

    Parameters
    ----------

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
    adaptive : str [optional]
        If activated uses adaptive activation functions, based on the function
        specified in `activation`. Options are False, 'global' and
        'layer'. Defaulta if False.
    feature_expansion: func or None [optional]
        If not None, then the trunk inputs are feature expanded using the
        function provided. Default is None.
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
    norm_out_type : str [optional]
        Type of output normalization to use. Default is 'z-score'.
    save_freq : int [optional]
        Save model frequency. Default is 1.
    restore : bool [optional]
        If True, it checks if a checkpoint exists in dest. If a checkpoint
        exists it restores the modelfrom there. Default is True.
    """
    # Initialize the class
    def __init__(self,
                 m,
                 dim_y,
                 depth_branch,
                 depth_trunk,
                 p,
                 dest='./',
                 regularizer=None,
                 p_drop=0.0,
                 activation='relu',
                 adaptive=False,
                 slope_recovery=False,
                 optimizer=keras.optimizers.Adam(lr=1e-3),
                 norm_in=False,
                 norm_out=False,
                 norm_out_type='z-score',
                 save_freq=1,
                 restore=True):

        # Numbers and dimensions
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
        self.save_freq   = save_freq
        self.activation  = activation
        self.adaptive    = adaptive

        # Activation function
        if activation=='tanh':
            self.act_fn = keras.activations.tanh
            self.kinit  = 'glorot_normal'
        elif activation=='relu':
            self.act_fn = keras.activations.relu
            self.kinit  = 'he_normal'
        elif activation=='elu':
            self.act_fn = keras.activations.elu
            self.kinit  = 'he_normal'
        elif activation=='selu':
            self.act_fn = keras.activations.selu
            self.kinit  = 'lecun_normal'

        # Adaptive global
        if   adaptive == 'global':
            self.act_fn = AdaptiveAct(activation=self.act_fn)
        elif adaptive == 'layer':
            self.act_f0 = self.act_fn

        # Inputs definition
        funct = keras.layers.Input(m,     name='funct')
        point = keras.layers.Input(dim_y, name='point')

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

        # Expand time
        if feature_expansion is not None:
            hid_t = keras.layers.Lambda(feature_expansion)(hid_t)

        # Branch network
        for ii in range(self.depth_branch-1):
            if adaptive=='layer':
                self.act_fn = AdaptiveAct(activation=self.act_f0)
            hid_b = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       kernel_initializer=self.kinit,
                                       activation=self.act_fn)(hid_b)
            if p_drop:
                hid_b = keras.layers.Dropout(p_drop)(hid_b)
        hid_b = keras.layers.Dense(self.width,
                                   kernel_initializer=self.kinit,
                                   kernel_regularizer=self.regu)(hid_b)

        # Trunk network
        for ii in range(self.depth_trunk):
            if adaptive=='layer':
                self.act_fn = AdaptiveAct(activation=self.act_f0)
            hid_t = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       kernel_initializer=self.kinit,
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

        # Add slope recovery
        if adaptive and slope_recovery:
            adps   = [tf.math.exp(v) for v in self.model.trainable_variables
                      if 'adaptive' in v.name]
            srecov = lambda: tf.reduce_mean(adps)
            self.model.add_loss(srecov)


        # Create save checkpoints, managers and callbacks
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  self.dest + '/ckpt',
                                                  max_to_keep=5)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

        self.ckpt_cb = self.ckpt_cb_creator(self.manager, self.ckpt, self.save_freq)

        # Creater logger
        self.logger = keras.callbacks.CSVLogger('output.dat',
                                                append=True,
                                                separator=' ')

    def ckpt_cb_creator(self, manager, ckpt, save_freq):
        ''' Create Checkpoint Callback that works like TF manager '''
        class CkptCb(keras.callbacks.Callback):
            def __init__(self_cb, ckpt):
                super(CkptCb, self_cb).__init__()
                self_cb.step = ckpt.step
            def on_epoch_end(self_cb, epoch, logs):
                if self_cb.step%save_freq==0:
                    manager.save()
                self_cb.step.assign_add(1)
        return CkptCb(ckpt)

    def train(self,
              Xf, Xp, Y, W=None,
              epochs=10,
              batch_size=32,
              loss_fn='mse',
              data_in_dataset=False,
              buffer_size=None,
              verbose=False,
              print_freq=1,
              valid_freq=0,
              early_stopping=False,
              val_threshold=np.inf,
              valid_func=False,
              Xf_test=None, Xp_test=None, Y_test=None, W_test=None,
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
        W : ndarray [optional]
            Weights used for the loss function. If None then ones are used.
            Default is None. Must have the same shape as Y.
        epochs : int [optional]
            Number of epochs to train.
        batch_size : int [optional]
            Size of batches.
        loss_fn : str [optional]
            Loss function to be used for training and validation.
        data_in_dataset : bool [optional]
            If True, input data is a tf.data.Dataset. Default is False.
        buffer_size : int [optional]
            Buffer size used in dataset shuffle method. If None the cardinality
            of the train_data is used. Default is None.
        verbose : bool [optional]
            Verbose output or not. Default is False.
        print_freq : int [optional]
            Print status frequency. Default is 1.
        valid_freq : int [optional]
            Validation check frequency. If zero, no validation is performed.
            Default is 0.
        early_stopping : bool [optional]
            If True only saves the model Checkpoint when the validation is
            decreasing. Default is False.
        Xf_test : ndarray
            Input for branch network used for testing. Must have shape (:, m).
        Xp_test : ndarray
            Input for trunk network used for testing. Must have shape (:, dim_y).
        Y_test : ndarray
            Data used for testing, G(u)(y)
            Must have shape (:, 1).
        W_test : ndarray
            Weights used in the loss function.
            Must have shape (:, 1).
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

        # Make weights
        if W is None:
            W = np.ones(np.shape(Y))

        # Run epochs
        ep0 = int(self.ckpt.step)
        best_val = np.inf
        for ep in range(ep0, ep0+epochs):
            # Print status
            if ep%print_freq==0:
                try:
                    self.print_status(ep, [loss.numpy()], verbose=verbose)
                except:
                    Y_pred = self.model((Xf_test, Xp_test))
                    loss   = loss_fn(Y_test, Y_pred, W_test)
                    self.print_status(ep, [loss.numpy()], verbose=verbose)

                # Print adaptive weights evol
                if self.adaptive=='layer':
                    adps = [v.numpy()[0]
                            for v in self.model.trainable_variables
                            if 'adaptive' in v.name]
                    self.print_status(ep, adps, fname='adpt')

            # Perform validation check
            if valid_freq and ep%valid_freq==0:
                Y_pred = self.model((Xf_test, Xp_test))
                valid  = loss_fn(Y_test, Y_pred, W_test)
                self.print_status(ep, [valid.numpy()], fname='valid')

                if valid_func:
                    self.validation(ep)

            # Save progress
            self.ckpt.step.assign_add(1)
            if ep%save_freq==0 and valid.numpy()<best_val:
                self.manager.save()
                
                if early_stopping and valid.numpy()<val_threshold:
                    best_val = valid.numpy()
            
            # Loop through batches
            for ba in range(batches):

                # Create batches and cast to TF objects
                (Xf_batch, Xp_batch,
                  Y_batch, W_batch)  = get_mini_batch(Xf, Xp, Y, W,
                                                      idx_arr, batch_size)
                Xf_batch = tf.convert_to_tensor(Xf_batch)
                Xp_batch = tf.convert_to_tensor(Xp_batch)
                Y_batch  = tf.convert_to_tensor(Y_batch)
                W_batch  = tf.convert_to_tensor(W_batch)

                if timer: t0 = time.time()
                loss = self.training_step(Xf_batch, Xp_batch, Y_batch, W_batch, loss_fn)
                if timer:
                    print("Time per batch:", time.time()-t0)
                    if ba>10 or ep>5: timer = False

    @tf.function
    def training_step(self, Xf_batch, Xp_batch, Y_batch, W_batch, loss_fn):
        with tf.GradientTape() as tape:
            Y_pred = self.model((Xf_batch, Xp_batch), training=True)
            loss   = loss_fn(Y_batch, Y_pred, W_batch)
            applied = tf.add_n([loss] + self.model.losses)

        # Calculate gradients
        gradients = tape.gradient(applied, self.model.trainable_variables)

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

def get_mini_batch(X1, X2, Y, W, idx_arr, batch_size):
    idxs = np.random.choice(idx_arr, batch_size)
    return X1[idxs], X2[idxs], Y[idxs], W[idxs]

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
def MSE_loss(y_true, y_pred, weights):
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

# ###############################################
# Functions for dataset processing and generation
# ###############################################

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Used to generate the records
def serialize_example(Xf, Xp, Y):
    # After creating each array save it by doing:
    # with tf.io.TFRecordWriter(f'data/{label}.tfrecord') as writer:
    #     for ii in range(num):
    #         serialized = serialize_example(combs_in[ii], p_in[ii], evol_out[ii])
    #         writer.write(serialized)
    feature = {
        'Xf': _float_feature(Xf.flatten()),
        'Xp': _float_feature(Xp.flatten()),
        'Y':  _float_feature(Y.flatten()),
    }
    example    = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()

    return serialized

# Used to generate the records
def serialize_example_with_weights(Xf, Xp, Y, W):
    feature = {
        'Xf': _float_feature(Xf.flatten()),
        'Xp': _float_feature(Xp.flatten()),
        'Y':  _float_feature(Y.flatten()),
        'W':  _float_feature(W.flatten()),
    }
    example    = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()

    return serialized

# Parse data
def proto_wrapper(branch_sensors):
    def parse_proto(example_proto):
        features = {
            'Xf': tf.io.FixedLenFeature([branch_sensors], tf.float32),
            'Xp': tf.io.FixedLenFeature([2], tf.float32),
            'Y':  tf.io.FixedLenFeature([], tf.float32),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return (parsed_features['Xf'], parsed_features['Xp']), parsed_features['Y']
    return parse_proto

# Parse data
def proto_wrapper_with_weights(branch_sensors):
    def parse_proto(example_proto):
        features = {
            'Xf': tf.io.FixedLenFeature([branch_sensors], tf.float32),
            'Xp': tf.io.FixedLenFeature([2], tf.float32),
            'Y':  tf.io.FixedLenFeature([], tf.float32),
            'W':  tf.io.FixedLenFeature([], tf.float32),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return ((parsed_features['Xf'], parsed_features['Xp']),
                 parsed_features['Y'],
                 parsed_features['W'])
    return parse_proto

# Load dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
def load_dataset(filepaths, branch_sensors, batch_size,
                 use_weights=True,
                 preads=1,
                 shuffle_buffer=0):
    # Read records
    dataset = tf.data.TFRecordDataset(filepaths,
                                      num_parallel_reads=preads)

    # Disable order
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = dataset.with_options(ignore_order)

    # Parse proto
    if use_weights:
        dataset = dataset.map(proto_wrapper_with_weights(branch_sensors),
                              num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(proto_wrapper(branch_sensors),
                              num_parallel_calls=AUTOTUNE)

    # Shuffle, prefetch and batch
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset
