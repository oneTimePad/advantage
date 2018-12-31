from abc import ABCMeta, abstractmethod
from enum import Enum
import tensorflow as tf
import numpy as np
import gin
from advantage.utils.tf_utils import ScopeWrap

""" Contains common policies/functions for RL agents.
"""

def _value_layer(tensor_inputs, num_actions):
    """Useful for constructing output of state-value function or action-value function
            Args:
                tensor_inputs: output of network to pass in

                num_actions: number of actions in a determinstic settings or
                    1 for value function
            Returns:
                regression layer output
    """
    shape = tensor_inputs.get_shape()
    # conv2d needs 4D input
    if len(shape) != 4:
        for _ in range(4 - len(shape)):
            tensor_inputs = tf.expand_dims(tensor_inputs, axis=1)
    shape = tensor_inputs.get_shape()
    kernel_h = int(shape[1])
    kernel_w = int(shape[2])
    conv = tf.layers.conv2d(tensor_inputs,
                            filters=num_actions,
                            kernel_size=[kernel_h, kernel_w],
                            activation=None,
                            name="value")
    # remove extra dimensions added
    return tf.squeeze(conv, axis=[1, 2])

class RLFunction(metaclass=ABCMeta):
    """ Represents any approximate RL
    function
    """

    def __init__(self,
                 scope,
                 approximator):

        self._scope = scope
        self._approximator = approximator
        self._eval_func = None
        self._func = approximator.network

    def __getattr__(self, attr):
        return getattr(self._approximator, attr)

    @property
    def scope(self):
        """ property for _scope
        """
        return self._scope

    @property
    def approximator(self):
        """ property for `_func`
        """
        return self._approximator

    @property
    def func(self):
        """ property for `_func`
        """
        return self._func

    @abstractmethod
    def set_up(self, session, **kwargs):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
                kwargs: args for policy
        """
        raise NotImplementedError()

@gin.configurable(blacklist=["scope"])
class ValueFunction(RLFunction):
    """ Represents a Value or Action-Value
    function
    """

    name_scope = "value_function"

    def __init__(self,
                 scope,
                 approximator,
                 state_shape,
                 num_of_actions):


        with scope():
            state_plh = tf.placeholder(shape=state_shape,
                                       dtype=tf.float32,
                                       name="state")
            approx_scope = ScopeWrap.build(scope, approximator.name_scope)
            approx_inst = approximator(approx_scope, state_plh, [state_plh])

        self._num_of_actions = num_of_actions
        super().__init__(self, scope, approx_inst)

    def __call__(self,
                 session,
                 states):

        return self._eval_func(session, states)

    def set_up(self,
               session,
               **kwargs):
        """ SetUp method for ValueFunction
        builds all necessary TF Graph Elements

            Args:
                session: tf.Session
        """
        self._approximator.set_up()
        with self._scope():
            num_of_actions = self._num_of_actions
            self._func = _value_layer(self._approximator.network, num_of_actions)

            if num_of_actions and num_of_actions > 1:
                # the actions taken by the policy leading to the bellman transition
                action_taken_plh = tf.placeholder(shape=[None, 1],
                                                  dtype=tf.int32,
                                                  name="action_taken")
                self._approximator.add_target_placeholder(action_taken_plh)

                # extract the Q-value for the action taken
                self._func = tf.reduce_sum(tf.one_hot(indices=action_taken_plh,
                                                      depth=num_of_actions) * self._func,
                                           axis=1)

            self._eval_func = lambda session, states: self._approximator.inference(session,
                                                                                   {"state": states})

            if num_of_actions and num_of_actions > 1:
                self._eval_func = lambda session, states: np.amax(self._eval_func(session, states))

        self._approximator.initialize(session)

class ContinousActionValueFunction(RLFunction):
    """ Represents an Action-Value Function (Q)
    that has a continuous action as it's input
    as well as a state. Examples are Continuous-Q
    Learning and DDPG
    """

    name_scope = "continuous_action_value_function"

    def __init__(self,
                 scope,
                 approximator,
                 has_action_source=False):

        self._has_action_source = has_action_source
        super().__init__(self, scope, approximator)

    def __call__(self,
                 session,
                 states,
                 actions=None):

        if self._has_action_source and actions:
            self._eval_func(session, states, actions)
        elif self._has_action_source and not actions:
            raise ValueError("when built without an action source"
                             " kwarg `actions` specifying selected"
                             " actions is required")
        if actions:
            raise ValueError("actions must not be passed when"
                             " when a action source is pressent")

        return self._eval_func(session, states)

    def set_up(self,
               session,
               **kwargs):
        """ SetUp method for ContinuousValueFunction
        builds all necessary TF Graph Elements

            Args:
                session: tf.Session
        """

        with self._scope():

            self._func = _value_layer(self._approximator.network, 1)

        if self._has_action_source:
            self._eval_func = lambda session, states: self._approximator.inference(session,
                                                                                   {"state": states})
        else:
            self._eval_func = lambda session, states, actions: self._approximator.inference(session,
                                                                                            {"state": states,
                                                                                             "action": actions})

        self._approximator.initialize(session)

    @gin.configurable(blacklist=["scope", "state_shape", "action_shape"])
    @classmethod
    def build(cls,
              scope,
              approximator,
              state_shape,
              action_shape):
        """ Constructs the Value Function

            Args:
                upper_scope : the upper ScopeWrap
                approximator_config: config to build approximator
                state_shape: shape of state input to approximator
                action_shape: shape of action input to approximator

            Returns:
                ContinousActionValueFunction
        """

        with scope():
            state_plh = tf.placeholder(shape=state_shape,
                                       dtype=tf.float32,
                                       name="state")

            action_plh = tf.placeholder(shape=action_shape,
                                        dtype=tf.float32,
                                        name="action")

            concat = tf.concat([state_plh, action_plh], axis=1)

        approx_scope = ScopeWrap.build(scope, approximator.name_scope)
        approx_inst = approximator(approx_scope,
                                   concat,
                                   [state_plh, action_plh])

        return cls(scope,
                   approx_inst)

    @gin.configurable(blacklist=["scope", "state_shape"])
    @classmethod
    def build_from_action_source(cls,
                                 scope,
                                 approximator,
                                 state_shape,
                                 action_func):
        """ Constructs the Value Function with an action
        source function. This makes computing the gradient
        with respect to the action source parameters
        easier. In addition, this function only needs to
        take the state as input as opposed to the state
        and action.

            Args:
                scope : ScopeWrap
                approximator: approximator from gin
                state_shape: shape of state input to approximator
                action_func: action RLFunction

            Returns:
                ContinousActionValueFunction
        """

        with scope():
            state_plh = tf.placeholder(shape=state_shape,
                                       dtype=tf.float32,
                                       name="state")

            concat = tf.concat([state_plh, action_func.network], axis=1)

        approx_scope = ScopeWrap.build(scope, approximator.name_scope)
        approx_inst = approximator(approx_scope,
                                   concat,
                                   [state_plh])

        return cls(scope,
                   approx_inst,
                   has_action_source=True)


@gin.configurable(blacklist=["scope", "state_shape", "action_shape"])
class ContinuousRealPolicy(RLFunction):
    """ Policy that outputs a Real Number
    useful for parameterizing means
    """

    name_scope = "continuous_real_policy_function"

    def __init__(self,
                 scope,
                 approximator,
                 state_shape,
                 action_shape):

        self._action_shape = action_shape

        with scope():
            state_plh = tf.placeholder(shape=state_shape,
                                       dtype=tf.float32,
                                       name="state")
        approx_scope = ScopeWrap.build(scope, approximator.name_scope)
        approx_inst = approximator(approx_scope,
                                   state_plh,
                                   [state_plh])

        super().__init__(self, scope, approx_inst)


    def __call__(self,
                 session,
                 states):

        return self._eval_func(session, states)

    def set_up(self, session, **kwargs):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
                kwargs: args for policy
        """
        with self._scope():

            self._func = _value_layer(self._approximator.network, self._action_shape)

        self._eval_func = lambda session, states: self._approximator.inference(session,
                                                                               {"state": states})

        self._approximator.initialize(session)

class ProbabilisticPolicy(RLFunction):
    """ Represents a
    policy that selects actions
    given a state based on a probability
    distribution
    """
    name_scope = None

    def __init__(self,
                 scope,
                 approximator,
                 state_shape):
        with scope():
            state_plh = tf.placeholder(shape=state_shape,
                                       dtype=tf.float32,
                                       name="state")
        approx_scope = ScopeWrap.build(scope, approximator.name_scope)
        approx_inst = approximator(approx_scope,
                                   state_plh,
                                   [state_plh])

        super().__init__(self, scope, approx_inst)

    def __call__(self,
                 session,
                 states):

        return self._eval_func(session, states)

    @abstractmethod
    def _policy(self, network, **kwargs):
        """ Appends Policy to
        network

            Args:
                network output

            Returns:
                network with output appended
        """
        raise NotImplementedError()

    def set_up(self, session, **kwargs):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
                kwargs: args for policy
        """
        self._func = self._policy(self._approximator.network, **kwargs)
        self._eval_func = lambda session, states: self._approximator.inference(session,
                                                                               {"state": states})

        self._approximator.initialize(session)

@gin.configurable(blacklist=["scope", "state_shape"])
class MultinomialPolicy(ProbabilisticPolicy):
    """ Policy on N-actions
    """

    name_scope = "multinomial_policy"

    @staticmethod
    def _sample_multinomial(distribution):
        """ Sample from Normal given
        mean and sigma
            Args:
                distribution: np ndarray of probabilities

            Returns:
                action
        """

        return np.amax(distribution, axis=1)

    def _policy(self, network, **kwargs):
        """Useful for constructing output layers for a multinomial policy settings
                Args:
                    network: output of network to pass in
                    kwargs["num_of_actions"] : number of actions
                Returns:
                    output of multinomial policy
        """
        if not kwargs or "num_of_actions" not in kwargs:
            raise ValueError("MultinomialPolicy expects `num_of_actions` kwargs")

        return tf.nn.softmax(network, axis=kwargs["num_of_actions"])

    def set_up(self,
               session,
               **kwargs):
        """
            Args:
                session: tf.Session
                kwargs["num_of_actions"]: number of policy actions
        """
        if "num_of_actions" not in kwargs:
            raise ValueError("MultinomialPolicy expects `num_of_actions` kwargs")

        super().set_up(session, num_of_actions=kwargs["num_of_actions"])

        def sample(session, states):
            distribution = self._eval_func(session, states)

            return self._sample_multinomial(distribution)

        self._eval_func = sample

@gin.configurable(blacklist=["scope", "state_shape"])
class BernoulliPolicy(ProbabilisticPolicy):
    """ Policy on 2 actions.
        An alternative to MultinomialPolicy
        on two actions.
    """

    name_scope = "bernoulli_policy"

    @staticmethod
    def _sample_binomial(distribution):
        """ Sample from Normal given
        mean and sigma
            Args:
                distribution: np ndarray of probabilities

            Returns:
                action
        """

        return np.amax(distribution, axis=1)

    def _policy(self, network, **kwargs):
        """Useful for constructing output layers for a multinomial policy settings
                Args:
                    network: output of network to pass in
                Returns:
                    output of binomial policy
        """

        return tf.nn.sigmoid(network)

    def set_up(self,
               session,
               **kwargs):
        """
            Args:
                session: tf.Session
        """

        super().set_up(session)

        def sample(session, states):
            distribution = self._eval_func(session, states)

            return self._sample_binomial(distribution)

        self._eval_func = sample

@gin.configurable(blacklist=["scope", "state_shape"])
class GaussianPolicy(ProbabilisticPolicy):
    """ Continuous Gaussian Policy
    """

    name_scope = "gaussian_policy"

    def __init__(self,
                 scope,
                 approximator):
        self._mean = None
        self._sigma = None

        super().__init__(self, scope, approximator)

    @property
    def mean(self):
        """property for _mean
        """
        return self._mean

    @property
    def sigma(self):
        """property for _sigma
        """
        return self._sigma

    @staticmethod
    def _sample_gaussian(mean, sigma):
        """ Sample from Normal given
        mean and sigma
            Args:
                mean: mean as np.ndarray
                sigma: stddev as np.ndarray

            Returns:
                action
        """

        covariance = np.diag(np.exp(sigma))

        return np.random.multivariate_normal(mean=mean, cov=covariance)


    def _policy(self, network, **kwargs):
        """Useful for constructing output layers for a multinomial policy settings
                Args:
                    network: output of network to pass in
                    kwargs["action_shape"]: shape of action space
                Returns:
                    mean and sigma gaussian policy
        """
        if not kwargs or "action_shape" not in kwargs:
            raise ValueError("MultinomialPolicy expects `action_shape` kwargs")

        mean = tf.layers.dense(network, kwargs["action_shape"], activation=None)
        sigma = tf.layers.dense(tf.ones([1, kwargs["action_shape"]]),
                                kwargs["num_of_actions"],
                                activation=None,
                                use_bias=False)
        self._mean = mean
        self._sigma = sigma

        return mean, sigma

    def set_up(self,
               session,
               **kwargs):
        """
            Args:
                session: tf.Session
                kwargs["action_shape"]: shape of action space
        """
        if "action_shape" not in kwargs:
            raise ValueError("MultinomialPolicy expects `action_shape` kwargs")

        super().set_up(session, num_of_actions=kwargs["action_shape"])

        def sample(session, states):
            mean_value, sigma_value = self._eval_func(session, states)

            return self._sample_gaussian(mean_value, sigma_value)

        self._eval_func = sample

class Policies(Enum):
    """Possible Policies to select
    """
    VALUE = ValueFunction
    CONT_ACTION_VALUE = ContinousActionValueFunction
    CONT_REAL = ContinuousRealPolicy
    MULTINOMIAL = MultinomialPolicy
    BERNOULLI = BernoulliPolicy
    GAUSSIAN = GaussianPolicy
