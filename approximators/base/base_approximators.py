from abc import ABCMeta
from abc import abstractmethod
import os
import tensorflow as tf
from advantage.utils.proto_parsers import parse_which_one
from advantage.utils.tf_utils import build_init_uninit_op, strip_and_replace_scope
from advantage.approximators.base.utils import parse_optimizer
from advantage.approximators.base.output import Output
from advantage.exception import AdvantageError


""" Approximators for Approximate Reinforcement Learning
    Allows the User to specify their own function approximators or
    utilize out-of-the-box Architectures configured via protobuf configs.
"""


def _produce_feed_dict(runtime_tensor_inputs, placeholders):
    """Converts dict of containing runtime inputs into proper
        feed_dict
            Args:
                runtime_tensor_inputs: dict containing runtime_values as values
                    and keys from feed_dict_keys as keys
                in_feed_dict: the dictionary of placeholders to look at

            Returns:
                feed_dict: dictionary containg placeholders as
                    key and runtime values

            Raises:
                ValueError: on missing keys
    """
    if not isinstance(runtime_tensor_inputs, dict):
        raise ValueError("runtime_tensor_inputs, must be a dict with keys according to \
                feed_dict_keys as runtime values as values")

    feed_dict = {}
    get_plh_name = lambda plh: plh.name if hasattr(plh, "name") else str(plh)

    for k in runtime_tensor_inputs.keys():
        plh_name = get_plh_name(k)

        matches = list(filter(lambda x, y=plh_name: y in get_plh_name(x), placeholders))

        if not matches:
            raise ValueError("`runtime_tensor_inputs` dict missing %s" % k)

        if len(matches) != 1:
            raise ValueError("found more than one matches placeholders for %s" % k)

        # add back `:tensor_number` since TF feed_dict uses it
        feed_dict[matches[0]] = runtime_tensor_inputs[k]

    if len(feed_dict.keys()) != len(placeholders):
        raise ValueError("`feed_dict` missing required inputs")

    return feed_dict


def deep_approximator(cls):
    """ Wraps the DeepApproximator interface
        providing extended functionality
        and wrapping implemented methods
    """

    # pylint: disable=too-many-public-methods
    # reason-disabled: most of the public methods are properties
    # pylint: disable=too-many-instance-attributes
    # reason-disabled: all attrs are needed
    class _DeepApproximator:
        """ Interface for deep approximators to be used for value, policies, meta-losses, etc...
        """

        def __init__(self, graph, config, approximator_scope):
            self._config = config
            self._graph = graph
            self._network = None # output of TF sub-graph network

            self._approximator_scope = approximator_scope

            self._inputs_placeholders = []

            self._update_target_placeholders = [] # placeholders for targets in parameter updates

            self._trainable_variables = []

            self._copy_ops = {}

            self._learning_rate = config.learning_rate

            name_scope = approximator_scope.name_scope

            optimizer = parse_optimizer(config.optimizer)

            self._optimizer = optimizer(name_scope)(self._learning_rate)

            self._loss = None
            self._applied_gradients = None
            self._init_op = None
            self._train_op = None


            self._wrapped = cls()
            self._wrapped.config = None

            self.__class__.__name__ = self._wrapped.__class__.__name__
            self.__class__.__doc__ = self._wrapped.__class__.__doc__

        def __getattr__(self, attr):
            return getattr(self._wrapped, attr)

        @property
        def config(self):
            """ propety for `_config`
            """
            return self._config

        @property
        def inputs_placeholders(self):
            """ property `_inputs_placeholders`
            """
            return self._inputs_placeholders

        @property
        def learning_rate(self):
            """ property `_learning_rate `
            """
            return self._learning_rate

        @property
        def applied_gradients(self):
            """ property `_applied_gradients`
            """
            return self._applied_gradients

        @property
        def approximator_scope(self):
            """ property `_approximator_scope`
            """
            return self._approximator_scope

        @property
        def network(self):
            """ property `_network`
            """
            return self._network

        @property
        def trainable_parameters(self): #TODO add support for selecting variables to train
            """ property `_trainable_parameters`
            """
            return self._trainable_variables

        @property
        def trainable_parameters_dict(self):
            """ property `_trainable_parameters` as dict
            """
            return {v.name: v for v in self._trainable_variables}

        @property
        def init_op(self):
            """ property `_init_op `
            """
            return self._init_op

        @property
        def name_scope(self):
            """ property to get name_scope
            """
            return self._approximator_scope.name_scope

        def _post_set_up(self, inputs_placeholders, last_block):
            """ Completes network setup procedures after tf has built
                model.
                    Args:
                        inputs_placeholders: list, the required placeholders to fill before running
                        last_block: the end of the network without output head

                    Raises:
                        ValueError: missing required kwargs
            """

            if  not isinstance(inputs_placeholders, list):
                raise ValueError("Expects `inputs_placeholders` as list")

            self._inputs_placeholders = inputs_placeholders

            # build optimizer and add output
            with self._approximator_scope():

                self._network = Output.from_config(self._config)(last_block)

                # setup operations to copy parameters from runtime values
                self._trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                              scope=self.name_scope)


                for param in self._trainable_variables:
                    param_value_plh = tf.placeholder(shape=param.get_shape(), dtype=tf.float32)
                    assign_op = tf.assign(param, param_value_plh)
                    self._copy_ops[param.name] = (param_value_plh, assign_op)

        def _produce_input_feed_dict(self, runtime_tensor_inputs):
            """ Form feed dict to run forward pass
            """
            return _produce_feed_dict(runtime_tensor_inputs, self._inputs_placeholders)

        def _produce_target_feed_dict(self, runtime_tensor_targets):
            """ Form feed dict for supervised targets
            """
            return _produce_feed_dict(runtime_tensor_targets, self._update_target_placeholders)


        def set_up(self, tensor_inputs, inputs_placeholders):
            """ Wraps around the implemented set_up method in `DeepApproximator`
                interface. Provides access to the specific model config
                via the `config` attribute of `_wrapped` set by this method.
                    Args:
                        tensor_inputs: inputs to the network
                        inputs_placeholders: list, the required placeholders to fill before running
                            [used by _set_up]

            """

            model_config = getattr(self._config,
                                   parse_which_one(self._config, "approximator"))

            self._wrapped.config = model_config

            with self._approximator_scope():
                last_block = self._wrapped.set_up(tensor_inputs,
                                                  inputs_placeholders)

            self._post_set_up(inputs_placeholders,
                              last_block)


        def add_target_placeholder(self, placeholder):
            """ Let's network know about placeholders used for specifying targets in loss """

            self._update_target_placeholders.append(placeholder)

        def make_copy_op(self, session, network):
            """Runs operations to copy runtime values to this models params
                Args:
                    session: tf.Session object
                    network: approximator to copy from
                Raises:
                    ValueError: for wrong args
            """
            if not isinstance(session, tf.Session):
                raise ValueError("Must pass in tf.Session")

            ops = [v[1] for v in self._copy_ops.values()]

            suffix_start = len(self.name_scope.split("/"))

            network_params = network.trainable_parameters_dict

            network_params_runtime = session.run(network_params)

            replaced_scope = {strip_and_replace_scope(self.name_scope, k, suffix_start) :
                              v for k, v in network_params_runtime.items()}
            try:
                feed_dict = {v[0] : replaced_scope[k] for k, v in self._copy_ops.items()}
            except KeyError:
                raise ValueError("There was a problem attempting to copy network params!"
                                 " Are the networks both the same structure?")

            return lambda: session.run(ops, feed_dict=feed_dict)


        def initialize(self, session):
            """ Initialize all network variables
                    Args:
                        session: tf.Session

                    Raises:
                        ValueError: for not tf.Session
            """
            if not isinstance(session, tf.Session):
                raise ValueError("Must pass in tf.Session")

            with self._approximator_scope():
                vars_in_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=self.name_scope)

                self._init_op = build_init_uninit_op(session, vars_in_scope)

                if self._init_op:
                    session.run(self._init_op)
                else:
                    self._init_op = tf.no_op()

        def gradients(self, loss):
            """ Compute the network gradients for parameter update

                Returns:
                    gradients tensor
            """
            with self._approximator_scope():
                self._loss = loss
                return self._optimizer.compute_gradients(loss, self.trainable_parameters)

        def apply_gradients(self, gradients):
            """ Applied gradients to network optimizer and creates train operation
                    Args:
                        gradients in proper tensorflow format
            """
            with self._approximator_scope():
                self._applied_gradients = gradients
                global_step = tf.train.get_or_create_global_step()

                self._train_op = self._optimizer.apply_gradients(gradients,
                                                                 global_step=global_step)

        def minimize(self, loss):
            """ Short for minimizing loss
                    Args:
                        loss: loss to minimize
            """
            with self._approximator_scope():
                self._loss = loss
                global_step = tf.train.get_or_create_global_step()
                self._train_op = self._optimizer.minimize(loss,
                                                          var_list=self.trainable_parameters,
                                                          global_step=global_step)


        def update(self, session, runtime_inputs, runtime_targets):
            """ Perform a network parameter update
                    Args:
                        runtime_inputs: usually training batch inputs containg
                           placeholders and values runtime_targets: training batch targets
            """
            runtime_batch = {}
            runtime_batch.update(self._produce_input_feed_dict(runtime_inputs))
            runtime_batch.update(self._produce_target_feed_dict(runtime_targets))

            session.run(self._train_op, feed_dict=runtime_batch)

        def inference(self, session, runtime_tensor_inputs):
            """ Performs inference on runtime_tensor_inputs
                    Args:
                        session: current runtime session
                        runtime_tensor_inputs: dict containing runtime_values as
                           values and keys from feed_dict_keys as keys

                    Returns:
                        runtime graph output

                    Raises:
                        ValueError: on bad arguments
            """
            if not isinstance(session, tf.Session):
                raise ValueError("Must pass in tf.Session")

            feed_dict = self._produce_input_feed_dict(runtime_tensor_inputs)

            return session.run(self._network, feed_dict=feed_dict)

    return _DeepApproximator

# pylint: disable=too-few-public-methods
# reason-disabled: represents just an interface
class DeepApproximator(metaclass=ABCMeta):
    """ Interface for Deep Approximators
    """
    config = None # set by _DeepApproximator set_up method

    @abstractmethod
    def set_up(self, tensor_inputs, inputs_placeholders):
        """ TensorFlow construction of the approximator network
                Args:
                    tensor_inputs: actual input to the network
                    inputs_placeholders: list, the required placeholders to fill before running.
                        These are the placholders that the tensor_inputs depend on.

                Returns:
                    the last block in the network

                Raises:
                    NotImplementedError: must be implemented
        """
        raise NotImplementedError()
