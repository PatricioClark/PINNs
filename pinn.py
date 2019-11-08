# PINN general class

# Requires Python 3.* and Tensorflow 2.0

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras

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
    pde : function
        Function specifying the equations of the problem. Takes as a
        PhysicsInformedNN class instance, coords and eq_params as inputs.  The
        output of the function must be a list containing all equations.
    dest : str [optional]
        Path for output files.
    lambda_data : int [optional]
        Weight of the data part of the loss function.
    lambda_phys : int [optional]
        Weight of the physiscs part of the loss function.
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
                 pde,
                 dest='./',
                 lambda_data=1,
                 lambda_phys=1,
                 activation='tanh',
                 optimizer=keras.optimizers.Nadam(lr=0.01),
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
        self.pde         = pde
        self.eq_params   = eq_params
        self.eval_params = copy.copy(eq_params)
        self.inverse     = inverse
        self.restore     = restore
        self.normalize   = normalize
        self.optimizer   = optimizer
        self.activation  = activation
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys

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
              X_data, Y_data, X_phys,
              epochs, batch_size,
              verbose=False,
              save_freq=1,
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
        X_phys : ndarray
            Coordinates where the physics constraint is going to be enforced.
            Must have shape (:, din). Can be of different length from X_data.
        epochs : int
            Number of epochs to train
        batch_size : int
            Size of batches
        verbose : bool [optional]
            Verbose output or not. Default is False.
        save_freq : int [optional]
            Save model frequency. Default is 1.
        data_mask : list [optional]
            Determine which output fields to use in the data constraint. Must have
            shape (dout,) with either True or False in each element. Default is
            all True.
        """

        # Check data_mask
        if data_mask is None:
            data_mask = [True for _ in range(self.dout)]

        # Metrics
        mean_data = keras.metrics.Mean()
        mean_phys = keras.metrics.Mean()

        # Run epochs
        ep0     = int(self.ckpt.step)
        batches = np.max([X_data.shape[0], X_phys.shape[0]]) // batch_size
        for ep in range(ep0, ep0+epochs):
            for ba in range(batches):

                # Create batches
                X_batch, Y_batch = random_batch(X_data, Y_data,
                                                batch_size=batch_size)
                X_eqenf, _       = random_batch(X_phys, X_phys,
                                                batch_size=batch_size)

                with tf.GradientTape(persistent=True) as tape:
                    # Data part
                    Y_pred, dummy = self.model(X_batch)
                    aux = [tf.reduce_mean(tf.square(Y_batch[:,ii]-Y_pred[:,ii]))
                           if data_mask[ii] else 0 for ii in range(self.dout)]
                    loss_data = sum(aux)

                    # Grab inverse coefs
                    if self.inverse:
                        param_pred = self.model(X_eqenf)[1:]
                        qq = 0
                        for pp in range(self.dpar):
                            if self.inverse[pp]:
                                self.eval_params[pp] = param_pred[qq][:,0]
                                self.eq_params[pp]   = self.eval_params[pp][0].numpy()
                                qq += 1
                    else:
                        loss_data = loss_data + 0*dummy

                    # Physics part
                    X_eqenf   = tf.convert_to_tensor(X_eqenf)
                    equations = self.pde(self, X_eqenf, self.eval_params)
                    aux       = [tf.reduce_mean(tf.square(eq))
                                 for eq in equations]
                    loss_phys = sum(aux)

                    # Total loss function
                    loss = self.lambda_data*loss_data + self.lambda_phys*loss_phys

                # Get mean of losses
                mean_data(loss_data)
                mean_phys(loss_phys)

                # Calculate and apply gradients
                gradients = tape.gradient(loss,
                            self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients,
                            self.model.trainable_variables))

                # Free tape
                del tape

            # Print status
            self.print_status(ep,
                              mean_data.result(),
                              mean_phys.result(),
                              verbose=verbose)
            mean_data.reset_states()
            mean_phys.reset_states()

            # Save progress
            self.ckpt.step.assign_add(1)
            if ep%save_freq==0:
                self.manager.save()

    def print_status(self, ep, lu, lf, verbose=False):
        """ Print status function """
        output_file = open(self.dest + 'output.dat', 'a')
        print(ep,
              '{}'.format(lu),
              '{}'.format(lf),
              file=output_file)
        output_file.close()
        if verbose:
            print(ep,
                  '{}'.format(lu),
                  '{}'.format(lf))

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

def random_batch(X, Y, batch_size=32):
    """ Generate random batch of data and inputs """
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], Y[idx]

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

def LES3D(pinn, coords, params):
    """ LES 3D Smagorinsky equations """

    x = coords[:,0:1]
    y = coords[:,1:2]
    z = coords[:,2:3]
    t = coords[:,3:4]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x,y,z,t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x,y,z,t])
            X  = tf.concat((x,y,z,t),1)
            Yp = pinn.model(X)[0]
            u  = Yp[:,0]
            v  = Yp[:,1]
            w  = Yp[:,2]
            p  = Yp[:,3]

        # First derivatives
        u_t = tape1.gradient(u, t)[:,0]
        v_t = tape1.gradient(v, t)[:,0]
        w_t = tape1.gradient(w, t)[:,0]

        u_x = tape1.gradient(u, x)[:,0]
        v_x = tape1.gradient(v, x)[:,0]
        w_x = tape1.gradient(w, x)[:,0]
        p_x = tape1.gradient(w, x)[:,0]

        u_y = tape1.gradient(u, y)[:,0]
        v_y = tape1.gradient(v, y)[:,0]
        w_y = tape1.gradient(w, y)[:,0]
        p_y = tape1.gradient(w, y)[:,0]

        u_z = tape1.gradient(u, z)[:,0]
        v_z = tape1.gradient(v, z)[:,0]
        w_z = tape1.gradient(w, z)[:,0]
        p_z = tape1.gradient(w, z)[:,0]

        S11 = u_x
        S12 = 0.5*(u_y+v_x)
        S13 = 0.5*(u_z+w_x)
        S22 = v_y
        S23 = 0.5*(v_z+w_y)
        S33 = w_z

        vl    = 1.006e-3
        delta = 40*vl
        c_s   = params[0]
        eddy_viscosity = (c_s*delta)**2*tf.sqrt(2*(S11**2+2*S12**2+2*S13**2+
                                                            S22**2+2*S23**2+
                                                                     S33**2))
        tau11 = -2*eddy_viscosity*S11
        tau12 = -2*eddy_viscosity*S12
        tau13 = -2*eddy_viscosity*S13
        tau22 = -2*eddy_viscosity*S22
        tau23 = -2*eddy_viscosity*S23
        tau33 = -2*eddy_viscosity*S33
        del tape1

    # Second derivatives
    u_xx = tape2.gradient(u_x, x)[:,0]
    v_xx = tape2.gradient(v_x, x)[:,0]
    w_xx = tape2.gradient(w_x, x)[:,0]

    u_yy = tape2.gradient(u_y, y)[:,0]
    v_yy = tape2.gradient(v_y, y)[:,0]
    w_yy = tape2.gradient(w_y, y)[:,0]

    u_zz = tape2.gradient(u_z, z)[:,0]
    v_zz = tape2.gradient(v_z, z)[:,0]
    w_zz = tape2.gradient(w_z, z)[:,0]
    
    tau11_x = tape2.gradient(tau11, x)[:,0]
    tau21_x = tape2.gradient(tau12, x)[:,0]
    tau31_x = tape2.gradient(tau13, x)[:,0]

    tau12_y = tape2.gradient(tau12, y)[:,0]
    tau22_y = tape2.gradient(tau22, y)[:,0]  
    tau32_y = tape2.gradient(tau23, y)[:,0]  

    tau13_z = tape2.gradient(tau13, z)[:,0]
    tau23_z = tape2.gradient(tau23, z)[:,0]  
    tau33_z = tape2.gradient(tau33, z)[:,0]  
    del tape2

    # Equations to be enforced
    nu = 5e-5
    f0 = u_x+v_y+w_z
    f1 = (u_t + u*u_x + v*u_y + w*u_z +
            p_x - nu*(u_xx+u_yy+u_zz) + tau11_x + tau12_y + tau13_z)
    f2 = (v_t + u*v_x + v*v_y + w*v_z +
            p_y - nu*(v_xx+v_yy+v_zz) + tau21_x + tau22_y + tau23_z)
    f3 = (w_t + u*w_x + v*w_y + w*w_z +
            p_z - nu*(w_xx+w_yy+w_zz) + tau31_x + tau32_y + tau33_z)
        
    return [f0, f1, f2, f3]

def NS3D(pinn, coords, params):
    """ NS 3D equations """

    x = coords[:,0:1]
    y = coords[:,1:2]
    z = coords[:,2:3]
    t = coords[:,3:4]
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x,y,z,t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x,y,z,t])
            X  = tf.concat((x,y,z,t),1)
            Yp = pinn.model(X)[0]
            u  = Yp[:,0]
            v  = Yp[:,1]
            w  = Yp[:,2]
            p  = Yp[:,3]

        # First derivatives
        u_t = tape1.gradient(u, t)[:,0]
        v_t = tape1.gradient(v, t)[:,0]
        w_t = tape1.gradient(w, t)[:,0]

        u_x = tape1.gradient(u, x)[:,0]
        v_x = tape1.gradient(v, x)[:,0]
        w_x = tape1.gradient(w, x)[:,0]
        p_x = tape1.gradient(w, x)[:,0]

        u_y = tape1.gradient(u, y)[:,0]
        v_y = tape1.gradient(v, y)[:,0]
        w_y = tape1.gradient(w, y)[:,0]
        p_y = tape1.gradient(w, y)[:,0]

        u_z = tape1.gradient(u, z)[:,0]
        v_z = tape1.gradient(v, z)[:,0]
        w_z = tape1.gradient(w, z)[:,0]
        p_z = tape1.gradient(w, z)[:,0]

        del tape1

    # Second derivatives
    u_xx = tape2.gradient(u_x, x)[:,0]
    v_xx = tape2.gradient(v_x, x)[:,0]
    w_xx = tape2.gradient(w_x, x)[:,0]

    u_yy = tape2.gradient(u_y, y)[:,0]
    v_yy = tape2.gradient(v_y, y)[:,0]
    w_yy = tape2.gradient(w_y, y)[:,0]

    u_zz = tape2.gradient(u_z, z)[:,0]
    v_zz = tape2.gradient(v_z, z)[:,0]
    w_zz = tape2.gradient(w_z, z)[:,0]
    
    del tape2

    # Equations to be enforced
    nu = params[0]
    PX = params[1]
    f0 = u_x+v_y+w_z
    f1 = (u_t + u*u_x + v*u_y + w*u_z + p_x + PX - nu*(u_xx+u_yy+u_zz))
    f2 = (v_t + u*v_x + v*v_y + w*v_z + p_y      - nu*(v_xx+v_yy+v_zz))
    f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz))
        
    return [f0, f1, f2, f3]
