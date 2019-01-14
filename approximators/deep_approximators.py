from abc import ABCMeta
from abc import abstractmethod
import tensorflow as tf
import gin
from advantage.utils.tf_utils import build_init_uninit_op, strip_and_replace_scope
from advantage.exception import AdvantageError
import advantage.loggers as loggers


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


def _avg_loss_summary(loss):
    """ Creates average loss summary

            Args:
                loss: loss tensor
    """
    loss_avg = tf.train.ExponentialMovingAverage(0.9, name="moving_avg")

    #get the moving average ops (create shadow variables)
    loss_avg_op = loss_avg.apply([loss])

    #log loss and shadow variables for avg loss
    raw_sum = tf.summary.scalar(loss.op.name + " (raw)", loss)
    avg_sum = tf.summary.scalar(loss.op.name, loss_avg.average(loss))
    summary = tf.summary.merge([avg_sum, raw_sum])
    return (summary, loss_avg_op)

def deep_approximator(cls):
    """ Wraps the DeepApproximator interface
        providing extended functionality
        and wrapping implemented methods
    """

    # pylint: disable=too-many-public-methods
    # reason-disabled: most of the public methods are properties
    # pylint: disable=too-many-instance-attributes
    # reason-disabled: all attrs are needed
    @gin.configurable
    class _DeepApproximator:
        """ Interface for deep approximators to be used for value, policies, meta-losses, etc...
        """

        def __init__(self,
                     scope,
                     architecture,
                     tensor_inputs,
                     inputs_placeholders,
                     optimizer=None):
            self._scope = scope

            self._architecture = architecture

            self._network = None # output of TF sub-graph network

            self._tenor_inputs = tensor_inputs

            self._inputs_placeholders = inputs_placeholders

            self._update_target_placeholders = [] # placeholders for targets in parameter updates

            self._trainable_variables = []

            self._copy_ops = {}

            self._optimizer = None

            self._loss = None

            self._init_op = None
            self._train_op = None

            if not hasattr(cls, "set_up"):
                raise NotImplementedError("%s must implement"
                                          " `set_up(self, architecture, tensor_inputs, inputs_placeholders)`")


            self._wrapped = cls()

            self.optimizer_fn = optimizer

            self.__class__.__name__ = self._wrapped.__class__.__name__
            self.__class__.__doc__ = self._wrapped.__class__.__doc__

        def __getattr__(self, attr):
            return getattr(self._wrapped, attr)

        @property
        def architecture(self):
            """ property for `_architecture`
            """
            return self._architecture

        @property
        def tensor_inputs(self):
            """ property for `_tensor_inputs`
            """
            return self._tenor_inputs

        @property
        def inputs_placeholders(self):
            """ property `_inputs_placeholders`
            """
            return self._inputs_placeholders

        @property
        def scope(self):
            """ property `_approximator_scope`
            """
            return self._scope

        @property
        def network(self):
            """ property `_network`
            """
            return self._network

        @property
        def trainable_parameters(self):
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
            with self._scope():
                if self._optimizer_fn:
                    self._optimizer = self._optimizer_fn()

                self._network = last_block

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

        def set_up(self):
            """ Wraps around the implemented set_up method in `DeepApproximator`
                interface. Provides access to the architecture,
                tensor_inputs (inputs to the network) and
                inputs_placeholders (list, the required placeholders to fill before running
                            [used by _set_up])

            """

            with self._scope():

                last_block = self._wrapped.set_up(self._architecture,
                                                  self._tensor_inputs,
                                                  self._inputs_placeholders)

            self._post_set_up(self._inputs_placeholders,
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

            def copy_op():
                network_params_runtime = session.run(network_params)

                replaced_scope = {strip_and_replace_scope(self.name_scope, k, suffix_start) :
                                  v for k, v in network_params_runtime.items()}
                try:
                    feed_dict = {v[0] : replaced_scope[k] for k, v in self._copy_ops.items()}
                except KeyError:
                    raise ValueError("There was a problem attempting to copy network params!"
                                     " Are the networks both the same structure?")
                session.run(ops, feed_dict=feed_dict)

            return copy_op

        def initialize(self, session):
            """ Initialize all network variables
                    Args:
                        session: tf.Session

                    Raises:
                        ValueError: for not tf.Session
            """
            if not isinstance(session, tf.Session):
                raise ValueError("Must pass in tf.Session")

            with self._scope():
                vars_in_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope=self.name_scope)

                self._init_op = build_init_uninit_op(session, vars_in_scope)

                if self._init_op:
                    session.run(self._init_op)
                else:
                    self._init_op = tf.no_op()

        def from_gradient_func(self, gradient_func):
            """ Compute the network gradients for parameter update
            given a function that computes the gradient given
            the parameter to take the gradient with respect to
                Args:
                    gradient_func: (i.e.) lambda param: tf.gradient(..., param)

                Returns:
                    gradients tensor
            """

            with self._scope():
                self._loss = None
                return [(gradient_func(param), param) for param in self.trainable_parameters]

        def gradients(self, loss):
            """ Compute the network gradients for parameter update
                    Args:
                        loss: loss function

                    Returns:
                        list(tuple(gradient_op, variable_with_respect_to))
            """

            with self._approximator_scope():
                self._loss = loss
                return self._optimizer.compute_gradients(loss, self.trainable_parameters)


        def apply_gradients(self, gradients):
            """ Applied gradients to network optimizer and creates train operation
                    Args:
                        gradients in proper tensorflow format
            """
            with self._scope():
                global_step = tf.train.get_or_create_global_step()
                summary, loss_avg = _avg_loss_summary(self._loss)
                with tf.control_dependencies([loss_avg]):
                    self._train_op = [summary, self._optimizer.apply_gradients(gradients,
                                                                               global_step=global_step)]

        def minimize(self, loss):
            """ Short for minimizing loss
                    Args:
                        loss: loss to minimize
            """
            with self._scope():
                self._loss = loss
                global_step = tf.train.get_or_create_global_step()
                summary, loss_avg = _avg_loss_summary(loss)
                with tf.control_dependencies([loss_avg]):
                    self._train_op = [summary, self._optimizer.minimize(loss,
                                                                        var_list=self.trainable_parameters,
                                                                        global_step=global_step)]



        @loggers.value(loggers.LogVarType.RETURNED_VALUE,
                       stdout=False,
                       tensorboard=True)
        def update(self, session, runtime_inputs, runtime_targets):
            """ Perform a network parameter update
                    Args:
                        runtime_inputs: usually training batch inputs containg
                           placeholders and values
                        runtime_targets: training batch targets

                    Returns:
                        tf summary
            """
            runtime_batch = {}
            runtime_batch.update(self._produce_input_feed_dict(runtime_inputs))
            runtime_batch.update(self._produce_target_feed_dict(runtime_targets))

            results = session.run(self._train_op, feed_dict=runtime_batch)
            return results[0] # summary

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
