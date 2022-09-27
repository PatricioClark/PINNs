"""Implementation of Physics-Informed Neural Networks (PINNs)

Original references:
    - Raissi et al

2022 Patricio Clark Di Leoni
     Departamento de Ingeniería
     Universidad de San Andrés
     e-mail: pclarkdileoni@udesa.edu.ar
"""

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time

tf.keras.backend.set_floatx('float32')


class PhysicsInformedNN:
    """General PINN class

    Dimensional and problem agnostic implementatation of a Physics Informed
    Neural Netowrk (PINN), geared towards solving inverse problems.
    Dimensionality is set by specifying the dimensions of the inputs and the
    outputs. The Physics is set by supplying the partial differential equations
    of the problem.

    The instantiated model is stored in self.model and it takes coords as inputs
    and outputs a list where the first contains the learned fields and the rest
    of the entries contains the different learned parameters when running an
    inverse PINN or a dummy output that should be disregarded when not.

    Training can be done using the dynamic balance methods described in
    "Understanding and mitigating gradient pathologies in physics-informed
    neural networks" by Wang, Teng & Perdikaris (2020) (alpha option in
    training).

    A few definitions before proceeding to the initialization parameters:
    
    din   = input dims
    dout  = output dims


    Args:
        layers : list
            Shape of the NN. The first element must be din, and the last one
            must be dout.
        dest : str [optional]
            Path for output files.
        activation : str [optional]
            Activation function to be used. Default is 'tanh'.
        resnet : bool [optional]
            If True turn PINN into a residual network with two layers per
            block.
        optimizer : keras.optimizer instance [optional]
            Optimizer to be used in the gradient descent. Default is Adam with
            fixed learning rate equal to 5e-4.
        norm_in : float or array [optional]
            If a number or an array of size din is supplied, the first layer of
            the network normalizes the inputs uniformly between -1 and 1.
            Default is False.
        norm_out : float or array [optional]
            If a number or an array of size dout is supplied, the layer layer
            of the network normalizes the outputs using z-score. Default is
            False.
        inverse : list [optional]
            If a list is a supplied the PINN will run the inverse problem,
            where one or more of the paramters of the pde are to be found. The
            list must be of the same length as eq_params and its entries can be
            False, if that parameters is fixed, 'const' if the parameters to be
            learned is a constant (in this case the value provided in eq_params
            will be used to initialize the variable), or a list with two
            elements, the first a tuple indicating which arguments the
            parameter depends on, and the second a list with the shape of the
            NN to be used to model the hidden parameter.
        restore : bool [optional]
            If True, it checks if a checkpoint exists in dest. If a checkpoint
            exists it restores the modelfrom there. Default is True.
    """
    # Initialize the class
    def __init__(self,
                 layers,
                 dest='./',
                 activation='elu',
                 resnet=False,
                 optimizer=keras.optimizers.Adam(lr=5e-4),
                 norm_in=None,
                 norm_out=None,
                 norm_out_type='z-score',
                 inverse=None,
                 restore=True):

        # Numbers and dimensions
        self.din    = layers[0]
        self.dout   = layers[-1]
        self.layers = layers

        # Extras
        self.dest        = dest
        self.resnet      = resnet
        self.inverse     = inverse
        self.norm_in     = norm_in
        self.norm_out    = norm_out
        self.optimizer   = optimizer
        self.restore     = restore
        self.activation  = activation

        # Input definition and normalization
        coords = keras.layers.Input(self.din, name='coords')

        if norm_in is not None:
            x1     = tf.Variable(norm_in[0], name='x1')
            x2     = tf.Variable(norm_in[1], name='x2')
            def norm(x):
                return 2*(x-x1)/(x2-x1) - 1
        else:
            x1   = tf.Variable(1.0,  name='x1')
            x2   = tf.Variable(-1.0, name='x2')
            def norm(x):
                return x

        self.norm = norm
        coords = keras.layers.Lambda(norm)(coords)

        # Generate main network
        fields = self._generate_network(coords, layers, activation, resnet)

        # Normalize main network output
        if norm_out is not None:
            y1 = tf.Variable(norm_out[0], name='y1')
            y2 = tf.Variable(norm_out[1], name='y2')
            if norm_out_type=='z-score':
                def out_norm(x):
                    return y2*x + y1
            elif norm_out_type=='min_max':
                def out_norm(x):
                    return 0.5*(x+1)*(y2-y1) + y1
        else:
            y1   = tf.Variable(0.0, name='y1')
            y2   = tf.Variable(1.0, name='y2')
            def out_norm(x):
                return x
        fields  = keras.layers.Lambda(out_norm)(fields)

        # Generate inverse parts
        if self.inverse is not None:
            inv_outputs = self._generate_inverse(coords)
        else:
            inv_outputs = [coords]  # Use coords as dummy outputs
        outputs = [fields] + inv_outputs

        # Create model
        model = keras.Model(inputs=coords, outputs=outputs)
        self.model = model
        self.num_trainable_vars = np.sum([np.prod(v.shape)
                                          for v in self.model.trainable_variables])
        self.num_trainable_vars = tf.cast(self.num_trainable_vars, tf.float32)

        # Parameters for dynamic balance
        self.bal_phys = tf.Variable(1.0, name='bal_phys')

        # Create save checkpoints / Load if existing previous
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           bal_phys=self.bal_phys,
                                           optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  self.dest + '/ckpt',
                                                  max_to_keep=5)
        if self.restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

    def _generate_network(self, coords, layers, activation, resnet, mask=None, out_name='fields'):
        ''' Generate network '''
        din   = layers[0]
        dout  = layers[-1]
        depth = len(layers)-2
        width = layers[1]
        if din != self.din:
            raise ValueError('The input dimensions of all networks must be equal!')

        act_dict = self._generate_activation(activation)

        # Apply mask if provided
        if mask is not None:
            coords = keras.layers.Lambda(lambda x: mask*x)(coords)

        # Hidden layers
        hidden = coords
        first_layer = True
        for ii in range(depth):
            new_layer = keras.layers.Dense(width,
                                           kernel_initializer=act_dict['kinit'])(hidden)

            # Update params depending on arch choices
            if act_dict['type'] == 'siren':
                if first_layer:
                    omega0 = act_dict['first_omega0']
                else:
                    omega0 = act_dict['hidden_omega0']
                act_dict['kinit'] = tf.keras.initializers.RandomUniform(-tf.sqrt(6.0/width)/omega0,
                                                                         tf.sqrt(6.0/width)/omega0)
                act_dict['act_fn'] = SirenAct(omega0=omega0)

            # Apply activation function
            new_layer   = act_dict['act_fn'](new_layer)

            if resnet and not first_layer:
                aux_layer = keras.layers.Dense(width,
                                               kernel_initializer=act_dict['kinit'])(new_layer)
                aux_layer = act_dict['act_fn'](aux_layer)

                hidden = 0.5*(hidden + aux_layer)
            else:
                hidden = new_layer

            first_layer = False

        # Output definition
        fields = keras.layers.Dense(dout,
                                    kernel_initializer=act_dict['kinit'],
                                    name=out_name)(hidden)

        return fields

    def _generate_activation(self, activation):
        ''' Generate dictionary with activation function properties '''
        act_dict = {}

        # Check type
        if type(activation) is str:
            act_dict['type'] = activation
        elif isinstance(activation, dict):
            act_dict['type'] = activation['type']

        # Load default values
        if act_dict['type'] == 'tanh':
            act_fn = keras.activations.tanh
            kinit  = 'glorot_normal'

        elif act_dict['type'] == 'relu':
            act_fn = keras.activations.relu

        elif act_dict['type'] == 'elu':
            act_fn = keras.activations.elu
            kinit  = 'glorot_normal'

        elif act_dict['type'] == 'siren':
            act_fn = SirenAct()
            kinit  = tf.keras.initializers.RandomUniform(-1.0/self.din,
                                                          1.0/self.din)
            act_dict['first_omega0']  = 30.0
            act_dict['hidden_omega0'] = 30.0

        else:
            raise ValueError(f"Activation type '{act_dict['type']}' not implemented")

        act_dict['act_fn'] = act_fn
        act_dict['kinit']  = kinit

        # Look for new values
        if isinstance(activation, dict):
            for key in activation:
                act_dict[key] = activation[key]

        return act_dict

    def _generate_inverse(self, coords):
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
        inv_outputs = []

        # Iterate through equation parameters
        for ii, inv in enumerate(self.inverse):
            name = f'inv-{ii}'

            # Inverse parameter is a constant
            if   inv['type'] == 'const':
                cte = keras.layers.Lambda(lambda x: 0*x)(coords)
                ini = keras.initializers.Constant(value=inv['value'])
                out = keras.layers.Dense(1,
                                         bias_initializer=ini,
                                         name=name,
                                         use_bias=True)(cte)

            # Inverse parameter is a field
            elif inv['type'] == 'func':
                if 'activation' not in inv:
                    inv['activation'] = self.activation
                if 'resnet' not in inv:
                    inv['resnet'] = self.resnet
                if 'mask' not in inv:
                    inv['mask'] = None
                out = self._generate_network(coords,
                                             inv['layers'],
                                             inv['activation'],
                                             inv['resnet'],
                                             mask=inv['mask'],
                                             name=name)

            # Append inverse
            inv_outputs.append(out)

        return inv_outputs

    def train(self,
              x_data,
              y_data,
              pde,
              epochs,
              batch_size,
              eq_params=None,
              lambda_data=1.0,
              lambda_phys=1.0,
              alpha=0.0,
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

        x_data : ndarray
            Coordinates where the data constraint is going to be enforced.
            Must have shape (:, din).
        y_data : ndarray
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

        len_data = x_data.shape[0]
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

        num_flags = len(flag_idxs)
        if num_flags > batches:
            batches = num_flags
            batch_size = len_data//num_flags

        # Cast balance
        bal_phys = tf.constant(self.bal_phys.numpy(), dtype='float32')

        # Run epochs
        ep0     = int(self.ckpt.step)
        for ep in range(ep0, ep0+epochs):
            for ba in range(batches):

                # Create batches and cast to TF objects
                (x_batch,
                 y_batch,
                 l_data,
                 l_phys) = get_mini_batch(x_data,
                                          y_data,
                                          lambda_data,
                                          lambda_phys,
                                          ba,
                                          batch_size,
                                          flag_idxs,
                                          random=rnd_order_training)
                x_batch = tf.convert_to_tensor(x_batch)
                y_batch = tf.convert_to_tensor(y_batch)
                l_data  = tf.constant(l_data, dtype='float32')
                l_phys  = tf.constant(l_phys, dtype='float32')
                ba_counter  = tf.constant(ba)

                if timer:
                    t0 = time.time()
                (loss_data,
                 loss_phys,
                 inv_outputs,
                 bal_phys) = self._training_step(x_batch,
                                                 y_batch,
                                                 pde,
                                                 eq_params,
                                                 l_data,
                                                 l_phys,
                                                 data_mask,
                                                 bal_phys,
                                                 alpha,
                                                 ba_counter)
                if timer:
                    print("Time per batch:", time.time()-t0)
                    if ba>10:
                        timer = False


            # Print status
            if ep%print_freq==0:
                self._print_status(ep,
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
            self.ckpt.bal_phys.assign(bal_phys.numpy())
            if ep%save_freq==0:
                self.manager.save()

    @tf.function
    def _training_step(self, x_batch, y_batch,
                      pde, eq_params, lambda_data, lambda_phys,
                      data_mask, bal_phys, alpha, ba):
        with tf.GradientTape(persistent=True) as tape:
            # Data part
            output = self.model(x_batch, training=True)
            y_pred = output[0]
            aux = [tf.reduce_mean(
                   lambda_data*tf.square(y_batch[:,ii]-y_pred[:,ii]))
                   for ii in range(self.dout)
                   if data_mask[ii]]
            loss_data = tf.add_n(aux)

            # Physics part
            equations = pde(self.model, x_batch, eq_params)
            loss_eqs  = [tf.reduce_mean(
                         lambda_phys*tf.square(eq))
                         for eq in equations]
            loss_phys = tf.add_n(loss_eqs)
            equations = tf.convert_to_tensor(equations)

        # Calculate gradients of data part
        gradients_data = tape.gradient(loss_data,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Calculate gradients of physics part
        gradients_phys = tape.gradient(loss_phys,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Delete tape
        del tape

        # alpha-based dynamic balance
        if alpha > 0.0:
            mean_grad_data = get_mean_grad(gradients_data, self.num_trainable_vars)
            mean_grad_phys = get_mean_grad(gradients_phys, self.num_trainable_vars)
            lhat = mean_grad_data/mean_grad_phys
            bal_phys = (1.0-alpha)*bal_phys + alpha*lhat

        # Apply gradients to the total loss function
        gradients = [g_data + bal_phys*g_phys
                     for g_data, g_phys in zip(gradients_data, gradients_phys)]
        self.optimizer.apply_gradients(zip(gradients,
                    self.model.trainable_variables))

        # Save inverse constants for output
        inv_ctes = []
        if self.inverse is not None:
            for ii, inv in enumerate(self.inverse, start=1):
                if inv['type'] == 'const':
                    inv_ctes.append(output[ii][0])

        return (loss_data,
                loss_phys,
                inv_ctes,
                bal_phys)

    def _print_status(self, ep, lu, lf,
                     inv_ctes, alpha, verbose=False):
        """ Print status function """

        # Loss functions
        output_file = open(self.dest + 'output.dat', 'a')
        print(ep, f'{lu}', f'{lf}',
              file=output_file)
        output_file.close()
        if verbose:
            print(ep, f'{lu}', f'{lf}')

        # Inverse coefficients
        if self.inverse and len(inv_ctes) > 0:
            output_file = open(self.dest + 'inverse.dat', 'a')
            print(ep, *[pp.numpy()[0] for pp in inv_ctes], file=output_file)
            output_file.close()

        # Balance lambda with alpha
        if alpha:
            output_file = open(self.dest + 'balance.dat', 'a')
            print(ep, self.bal_phys.numpy(),
                  file=output_file)
            output_file.close()

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

@tf.function
def get_tr_k(grads):
    sum_over_layers = [tf.reduce_sum(tf.square(gr)) for gr in grads]
    total_sum       = tf.add_n(sum_over_layers)
    return total_sum

def get_mini_batch(X, Y, ld, lf, ba, batch_size, flag_idxs, random=True):
    ''' New separted version for this problem '''
    which = np.random.randint(np.shape(flag_idxs)[0])
    fi    = flag_idxs[which]
    idxs  = np.random.choice(fi, batch_size)
    return X[idxs], Y[idxs], ld[idxs], lf[idxs]

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
        aux = tf.multiply(10.0,  self.a)
        return self.activation(tf.multiply(aux, X))

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

class SirenAct(keras.layers.Layer):
    """ Siren activation function """
    def __init__(self, activation=tf.math.sin, omega0=30.0, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.omega0     = omega0

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def call(self, X):
        return self.activation(tf.multiply(self.omega0, X))

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
