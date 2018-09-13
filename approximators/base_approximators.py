from abc import ABCMeta
from abc import abstractmethod
from protos.approximators import helpers_pb2
from functools import partial
from utils.proto_parser import proto_to_dict
import tensorflow as tf

""" Approximators for Approximate Reinforcement Learning
    Allows the User to specify their own function approximators or
    utilize out-of-the-box Architectures configured via protobuf configs.
"""

OPTIMIZERS = {
    "AdamOptimizer": tf.train.AdamOptimizer,
    "GradientDescentOptimizer": tf.train.GradientDescentOptimizer
}


class DeepApproximator(object):
    """ Interface for deep approximators for value, policies, meta-losses, etc...
    """
    __metaclass__ = ABCMeta

    def __init__(self, graph, config):
        self._config = config
        self._graph = graph
        self._name_scope = config.name_scope
        self._reuse = config.reuse
        self._network = None
        self._var_scope_obj = None
        self._inputs_placeholders = []
        self._feed_dict = {} # dict of inputs names without ":0" and placeholders as values

        self._update_target_plhs_dict = {} # placeholders for targets in parameter updates


        self._learning_rate = config.learning_rate

        self._optimizer = OPTIMIZERS[self.enum_optimizer_to_str(config.optimizer)](self._learning_rate)

        self._loss = None
        self._applied_gradients = None
        self._init_op = None


    @property
    def inputs_placeholders(self):
        return self._inputs_placeholders

    @property
    def feed_dict_keys(self):
        return list(self._feed_dict.keys())

    @property
    def feed_dict(self):
        return self._feed_dict

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def applied_gradients(self):
        return self._applied_gradients

    @property
    def var_scope_obj(self):
        return self._var_scope_obj

    @property
    def var_scope_name(self):
        return self._name_scope
    @property
    def network(self):
        return self._network

    @property
    def trainable_parameters(self): #TODO add support for selecting variables to train
        return self._trainable_variables

    @property
    def trainable_parameters_dict(self):
        return {v.name: v for v in self._trainable_variables}

    @property
    def init_op(self):
        return self._init_op

    @staticmethod
    def parse_specific_model_config(config):
        """ Parses the parameters specific to the actual Approximator class
                Args:
                    config: the approximators_pb2 object
                Returns:
                    the "model" field from it
        """

        return getattr(config, config.WhichOneof("model"))

    @staticmethod
    def parse_output_proto_to_fn(config):
        """ Parses out the Output for this Approximator
                Args:
                    config: the approximators_pb2 object
                Returns:
                    the partial fn for the output to be applied to the
                    base network output
        """
        output = config.WhichOneof("output")
        return partial(eval(output), **proto_to_dict(getattr(config, output)))

    @staticmethod
    def enum_activation_to_str(enum_value):
        return helpers_pb2._ACTIVATION.values_by_number[enum_value].name

    @staticmethod
    def enum_optimizer_to_str(enum_value):
        return helpers_pb2._OPTIMIZER.values_by_number[enum_value].name

    @staticmethod
    def enum_initializer_to_str(enum_value):
        return helpers_pb2._INITIALIZER.values_by_number[enum_value].name

    def add_target_placeholder(self, placeholder):
        """ Let's network know about placeholders used for specifying targets in loss """

        self._update_target_plhs_dict[placeholder.name.split()[0]] = placeholder

    def copy(self, session, runtime_params):
        """Runs operations to copy runtime values to this models params
            Args:
                session: tf.Session object
                runtime_params: dict with full param names as keys and runtime values as values
            Raises:
                ValueError: for wrong args
        """
        if not isinstance(session, tf.Session):
            raise ValueError("Must pass in tf.Session")
        if not isinstance(runtime_params, dict):
            raise ValueError("runtime_params must a be dict with param names and values")

        ops = [v[1] for v in self._copy_ops_dict.values()]
        feed_dict = { v[0]: runtime_params[k] for k,v in self._copy_ops_dict.items() }
        #with session.as_default():

        with self._graph.as_default():
            session.run(ops, feed_dict=feed_dict)


    def set_up(self, tensor_inputs, inputs_placeholders, **kwargs):
        """ TensorFlow construction of the approximator network
                Args:
                    tensor_inputs: Tensor inputs to the network (used when overriding)
                    inputs_placeholders: list, the required placeholders to fill before running
                    kwargs:
                        var_scope_obj: scope object for network params
                        last_block: the end of the network without output head

                Raises:
                    NotImplementedError: if method is not overriden or no variables in graph
                    ValueError: missing required kwargs
        """
        if "var_scope_obj" not in kwargs:
            raise ValueError("Expects var_scope_obj as kwarg")
        self._var_scope_obj = kwargs["var_scope_obj"]

        if "last_block" not in kwargs:
            raise ValueError("Expects last_block as kwarg")
        last_block = kwargs["last_block"]

        if  not isinstance(inputs_placeholders, list):
            raise ValueError("Expects inputs_placeholders as list")
        self._inputs_placeholders = inputs_placeholders

        self._feed_dict = {v.name.split(":")[0]: v for v in self._inputs_placeholders}



        with self._graph.as_default():

            # pre-fetch to check if caller has constructed blocks in graph
            self._trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._var_scope_obj.name)

            if len(self._trainable_variables) == 0:
                # reset
                self._var_scope_obj = None
                self._network = None
                self._inputs_placeholders = []
                self._feed_dict = {}
                raise NotImplementedError("No variables in graph")

            # build optimizer and add output
            with tf.variable_scope(self._var_scope_obj):
                self._network = self.parse_output_proto_to_fn(self._config)(last_block)

            # setup operations to copy parameters from runtime values
            self._trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._var_scope_obj.name)

            self._copy_ops_dict = {}
            for param in self._trainable_variables:
                param_value_plh = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
                assign_op = tf.assign(param, param_value_plh)
                self._copy_ops_dict[param.name] = (param_value_plh, assign_op)



    def initialize(self, session):
        """ Initialize all network variables
                Args:
                    session: tf.Session

                Raises:
                    ValueError: for not tf.Session
        """
        if not isinstance(session, tf.Session):
            raise ValueError("Must pass in tf.Session")
        with self._graph.as_default():
            #with session.as_default():\
            self._init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._var_scope_obj.name))
        session.run(self._init_op)


    def __produce_feed_dict(self, runtime_tensor_inputs, in_feed_dict):
        """Converts dict of containing runtime inputs into proper
            feed_dict
                Args:
                    runtime_tensor_inputs: dict containing runtime_values as values and keys from
                        feed_dict_keys as keys
                    in_feed_dict: the dictionary of placeholders to look at

                Returns:
                    feed_dict: dictionary containg placeholders as key and runtime values

                Raises:
                    ValueError: on missing keys
        """
        if not isinstance(runtime_tensor_inputs, dict):
            raise ValueError("runtime_tensor_inputs, must be a dict with keys according to \
                    feed_dict_keys as runtime values as values")

        feed_dict = {}
        for k in in_feed_dict.keys():
            if not k in runtime_tensor_inputs:
                raise ValueError("runtime_tensor_inputs dict missing %s" % k)
            feed_dict[k + ":0"] = runtime_tensor_inputs[k] # TODO: this only accounts for output 0... for now

        return feed_dict


    def _produce_feed_dict(self, runtime_tensor_inputs):
        return self.__produce_feed_dict(runtime_tensor_inputs, self._feed_dict)

    def _produce_update_target_dict(self, runtime_tensor_targets):
        return self.__produce_feed_dict(runtime_tensor_targets, self._update_target_plhs_dict)


    @abstractmethod
    def inference(self, session, runtime_tensor_inputs):
        """ NOTE: Make sure to call _produce_feed_dict
        Performs Runtime inference on the network. Usually setups a Session
                Args:
                    session: current runtime session
                    runtime_tensor_inputs: dict containing runtime_values as values and keys from
                        feed_dict_keys as keys

                Returns:
                    runtime graph output

                Raises:
                    ValueError: on bad arguments
        """
        raise NotImplementedError()

    def gradients(self, loss):
        """ Compute the network gradients for parameter update

            Returns:
                gradients tensor
        """
        with self._graph.as_default():
            self._loss = loss
            return self._optimizer.compute_gradients(loss, self.trainable_parameters)

    def apply_gradients(self, gradients):
        """ Applied gradients to network optimizer and creates train operation
                Args:
                    gradients in proper tensorflow format
        """
        with self._graph.as_default():
            self._applied_gradients = gradients
            self._train_op = self._optimizer.apply_gradients(gradients)


    def update(self, session, runtime_inputs, runtime_targets):
        """ Perform a network parameter update
                Args:
                    runtime_inputs: usually training batch inputs containg placeholders and values
                    runtime_targets: training batch targets
        """
        runtime_batch = {}
        runtime_batch.extend(self._produce_feed_dict(runtime_inputs))
        runtime_batch.extend(self._produce_update_target_dict(runtime_targets))
        #with session.as_default():
        session.run(self._train_op, feed_dict=runtime_batch)



def multinomial(tensor_inputs, axis):
    """Useful for constructing output layers for a multinomial policy settings
            Args:
                tensor_inputs: output of network to pass in
                axis: axis to softmax on
            Returns:
                output of multinomial policy
    """
    return tf.nn.softmax(tensor_inputs, axis=axis)


def binomial(tensor_inputs, **kwargs):
    """Useful for constructing output layers for a binomial policy setting
            Args:
                tensor_inputs: output of network to pass in
            Returns:
                output of binomial policy
    """
    return tf.nn.sigmoid(tensor_inputs)


def value(tensor_inputs, **kwargs):
    """Useful for constructing output of state-value function or action-value function
            Args:
                tensor_inputs: output of network to pass in
                **kwargs:
                    num_actions: number of actions in a determinstic settings or 1 for value function
            Returns:
                regression layer output
    """
    shape = tensor_inputs.get_shape()
    if len(shape) != 4:
        for _ in range(4 - len(shape)):
            tensor_inputs = tf.expand_dims(tensor_inputs, axis=1)
    shape = tensor_inputs.get_shape()
    kernel_h = int(shape[1])
    kernel_w = int(shape[2])
    conv = tf.layers.conv2d(tensor_inputs, filters=kwargs['num_actions'], kernel_size=[kernel_h, kernel_w], activation=None, name="value")
    return tf.squeeze(conv, axis=[1, 2])

def gaussian(tensor_inputs, **kwargs):
    """Useful for constructing output layers for continuous stochastic policy
            Args:
                tensor_inputs: output of network to pass in
                **kwargs:
                    num_actions: shape of action tensor output
            Returns:
                mean and sigma gaussian policy
    """
    mean = tf.layers.dense(tensor_inputs, kwargs['num_actions'], activation=None)
    sigma = tf.layers.dense(tf.ones([1, kwargs['num_actions']]), activation=None, use_bias=False)
    return mean, sigma

ACTIVATIONS = {
    "NONE": tf.identity,
    "RELU" : tf.nn.relu6,
    "SIGMOID": tf.nn.sigmoid,
    "ELU": tf.nn.elu
}


INITIALIZERS = {
    "ones_initializer": tf.ones_initializer(),
    "variance_scaling_initializer": tf.variance_scaling_initializer()
}
