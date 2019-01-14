from abc import ABCMeta, abstractmethod
from enum import Enum
import tensorflow as tf
import numpy as np
import gin
import random
import attr
from advantage.utils.tf_utils import ScopeWrap
from advantage.utils.gin_utils import gin_classmethod
import advantage.loggers as loggers

""" Contains common policies/functions for RL agents.
"""

@attr.s
class Epsilon:
    """ Holds attrs for defining
    decaying epsilon
    """
    init = attr.ib(kw_only=True)
    min = attr.ib(kw_only=True)
    decay_steps = attr.ib(kw_only=True)
    decay_rate = attr.ib(kw_only=True)

def decayed_epsilon(scope,
                    epsilon):
    """ Constructs a callable that returns
    the decayed epsilon

        Args:
            agent: agent who has session
            scope: scope to make vars/ops in
            epsilon: epsilon config

        Returns:
            callable that returns epsilon value
    """
    with scope():
        eps = tf.train.exponential_decay(epsilon.init,
                                         scope.improve_step,
                                         epsilon.decay_steps,
                                         epsilon.decay_rate,
                                         staircase=True,
                                         name="decayed_epsilon")
    min_epsilon = epsilon.min

    @loggers.value(loggers.LogVarType.RETURNED_VALUE,
                   "epsilon",
                   "Agent current epsilon is %.2f",
                   tensorboard=False)
    def fetch_eps(session):

        eps_runtime = session.run(eps)

        return eps_runtime if eps_runtime > min_epsilon else min_epsilon

    return fetch_eps

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
                 approximator,
                 approximator_fn,
                 state_shape):

        self._scope = scope
        self._approximator = approximator
        self._approximator_fn = approximator_fn
        self._eval_func = None # represent func used for evaluation or inference
        self._func = approximator.network # represents tf network output
        self._state_shape = state_shape

    def __getattr__(self, attribute):
        return getattr(self._approximator, attribute)

    @property
    def scope(self):
        """ property for _scope
        """
        return self._scope

    @property
    def approximator_fn(self):
        """ property for `_approximator_fn`
        """
        return self._approximator_fn

    @property
    def approximator(self):
        """ property for `_approximator`
        """
        return self._approximator

    @property
    def func(self):
        """ property for `_func`
        """
        return self._func

    @property
    def state_shape(self):
        """ property for `_state_shape`
        """
        return self._state_shape

    @abstractmethod
    def set_up(self, session):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
                kwargs: args for policy
        """
        raise NotImplementedError()

    @abstractmethod
    @classmethod
    def copy_obj(cls, scope):
        """ Perform a copy from `copy_from`
                Args:
                    scope: the scope for the new copy

                Returns:
                    copy of RLFunction
                    of the same type
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
                 num_of_actions=None,
                 use_epsilon=False):

        with scope():
            state_plh = tf.placeholder(shape=state_shape,
                                       dtype=tf.float32,
                                       name="state")
            approx_scope = ScopeWrap.build(scope, approximator.name_scope)
            approx_inst = approximator(approx_scope, state_plh, [state_plh])

        self._num_of_actions = num_of_actions

        self._fetch_eps = None

        if scope.is_training and use_epsilon:
            self._fetch_eps = decayed_epsilon(scope, Epsilon())

        super().__init__(self,
                         scope,
                         approximator,
                         approx_inst)

    def __call__(self,
                 session,
                 states):
        sampled_action = self._eval_func(session, states)

        if self._fetch_eps:
            prob = random.random()
            eps = self._fetch_eps(session)

            if prob > eps:
                sampled_action = sampled_action
            else:
                sampled_action = random.randint(0, self._num_of_actions)

        return sampled_action

    @property
    def num_of_actions(self):
        """ property for `_num_of_actions`
        """
        return self._num_of_actions

    def set_up(self,
               session):
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

    def copy_obj(self, scope):
        approx_scope = ScopeWrap.build(scope, self.approximator_fn.name_scope)

        return ValueFunction(approx_scope,
                             self.approximator_fn,
                             self.state_shape,
                             self.num_of_actions)

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
                 approximator_fn,
                 has_action_source=False):

        self._has_action_source = has_action_source

        self.action_func = None
        self.action_shape = None

        super().__init__(self,
                         scope,
                         approximator,
                         approximator_fn)

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

    @property
    def has_action_source(self):
        """ property for `_has_action_source`
        """
        return self._has_action_source

    def set_up(self,
               session):
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

    @gin_classmethod(blacklist=["scope", "state_shape", "action_shape"])
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

        inst = cls(scope,
                   approx_inst,
                   approximator)

        inst.action_shape = action_shape

        return inst

    @gin_classmethod(blacklist=["scope", "state_shape"])
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

        inst = cls(scope,
                   approx_inst,
                   approximator,
                   has_action_source=True)

        inst.action_func = action_func

        return inst

    def copy_obj(self, scope):

        if self.has_action_source:
            return self.build_from_action_source(scope,
                                                 gin.REQUIRED,
                                                 self.state_shape,
                                                 gin.REQUIRED)
        return self.build(scope,
                          gin.REQUIRED,
                          self.state_shape,
                          self.action_shape)


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

        super().__init__(self,
                         scope,
                         approx_inst,
                         approximator)


    def __call__(self,
                 session,
                 states):

        return self._eval_func(session, states)

    def set_up(self, session):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
        """
        with self._scope():

            self._func = _value_layer(self._approximator.network, self._action_shape)

            if self.scope.is_training:
                self._func += tf.random_normal(self._action_shape)

        self._eval_func = lambda session, states: self._approximator.inference(session,
                                                                               {"state": states})

        self._approximator.initialize(session)

    def copy_obj(self, scope):
        approx_scope = ScopeWrap.build(scope, self.approximator_fn.name_scope)

        return ContinuousRealPolicy(approx_scope,
                                    self.approximator_fn,
                                    self.state_shape,
                                    self.num_of_actions)

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
                 state_shape,
                 action_shape=None):
        with scope():
            state_plh = tf.placeholder(shape=state_shape,
                                       dtype=tf.float32,
                                       name="state")
            if action_shape: # used by liklehood
                self._action_taken_plh = tf.placeholder(shape=action_shape,
                                                        dtype=tf.int32,
                                                        name="action_taken")
        approx_scope = ScopeWrap.build(scope, approximator.name_scope)
        approx_inst = approximator(approx_scope,
                                   state_plh,
                                   [state_plh])
        super().__init__(self, scope, approx_inst)

        self._old = None
        self._copy = None
        self._action_shape = action_shape
        self._liklehood = None

    def __call__(self,
                 session,
                 states):

        return self._sample(session, states)

    @property
    def old(self):
        """ property for `_old`
        """
        return self._old

    @property
    def liklehood(self):
        """ property for `_liklehood`
        """
        return self._liklehood

    @property
    def action_shape(self):
        """ property for `_action_shape`
        """
        return self._action_shape

    @property
    def action_taken_plh(self):
        """ property for `_action_taken_plh`
        """
        return self._action_taken_plh

    @abstractmethod
    def _sample(self, session, states):
        """ Builds sampling function
                Args:
                    session: tf.Session
                    states: states to evaluate

                Returns:
                    function for sampling
        """
        raise NotImplementedError()

    @abstractmethod
    def _policy(self, network):
        """ Appends Policy to
        network

            Args:
                network output

            Returns:
                network with output appended
        """
        raise NotImplementedError()

    @abstractmethod
    def _make_liklehood(self):
        """ Makes the liklehood
        tensor.
        """
        raise NotImplementedError()

    def _make_action_taken_plh(self, action_shape):
        """ Make the `action_taken_plh`
        if doesn't exist.
            Args:
                action_shape: shape of placeholder
        """
        if not self._action_taken_plh:
            with self._scope():
                self._action_taken_plh = tf.placeholder(shape=action_shape,
                                                        dtype=tf.int32,
                                                        name="action_taken")

    def eval_liklehood(self, session, states, actions=None):
        """ Computes liklehood
                Args:
                    session: tf.Session
                    states: states to eval on
                    actions: actions taken at states
                Returns:
                    liklehoods
        """
        if self._action_taken_plh:
            if not actions:
                raise ValueError("Expects actions to not be None")
            feed_dict = {"state": states, self._action_taken_plh: actions}
        else:
            feed_dict = {"state": states}

        return session.run(self.liklehood, feed_dict=feed_dict)

    def set_up(self, session):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
        """
        self._func = self._policy(self._approximator.network)
        self._eval_func = lambda session, states: self._approximator.inference(session,
                                                                               {"state": states})
        self._make_liklehood()
        self._approximator.initialize(session)

    def make_old(self, session):
        """ Constructs a copy of the policy
        termed "old". It is common for ProbabilisticPolicy's
        to have an old copy they sync with. Only allowed
        if training
            Args:
                session: tf.Session
        """

        if self.scope.is_training:
            old_scope = ScopeWrap.build(self.scope,
                                        "old/{0}".format(self.name_scope))

            self._old = self.copy_obj(old_scope)

            self._copy = self._old.make_copy_op(session,
                                                self._func)

    def sync_old(self):
        """ Sync and save params to `old`
        """
        self._copy()

@gin.configurable(blacklist=["scope", "state_shape"])
class MultinomialPolicy(ProbabilisticPolicy):
    """ Policy on N-actions
    """

    name_scope = "multinomial_policy"

    def __init__(self,
                 scope,
                 approximator,
                 state_shape,
                 num_of_actions):
        self._num_of_actions = num_of_actions

        super().__init__(self, scope, approximator, state_shape, [None, 1])

    @property
    def num_of_actions(self):
        """ property of _num_of_actions
        """
        return self._num_of_actions

    def _sample(self, session, states):
        distribution = self._eval_func(session, states)
        return np.amax(distribution, axis=1)

    def _policy(self, network):
        """Useful for constructing output layers for a multinomial policy settings
                Args:
                    network: output of network to pass in
                Returns:
                    output of multinomial policy
        """

        return tf.nn.softmax(network, axis=self._num_of_actions)

    def _make_liklehood(self):
        self._liklehood = -tf.log(self._func)[:, self._action_taken_plh]

    def copy_obj(self, scope):
        approx_scope = ScopeWrap.build(scope, self.approximator_fn.name_scope)

        return MultinomialPolicy(approx_scope,
                                 self.approximator_fn,
                                 self.state_shape,
                                 self.num_of_actions)


@gin.configurable(blacklist=["scope", "state_shape"])
class BernoulliPolicy(ProbabilisticPolicy):
    """ Policy on 2 actions.
        An alternative to MultinomialPolicy
        on two actions.
    """

    name_scope = "bernoulli_policy"

    def __init__(self,
                 scope,
                 approximator,
                 state_shape):

        super().__init__(self, scope, approximator, state_shape)

    def _sample(self, session, states):
        distribution = self._eval_func(session, states)
        return np.amax(distribution, axis=1)

    def _policy(self, network):
        """Useful for constructing output layers for a multinomial policy settings
                Args:
                    network: output of network to pass in
                Returns:
                    output of binomial policy
        """

        return tf.nn.sigmoid(network)

    def _make_liklehood(self):
        self._liklehood = -tf.log(self._func)


    def copy_obj(self, scope):
        approx_scope = ScopeWrap.build(scope, self.approximator_fn.name_scope)

        return BernoulliPolicy(approx_scope,
                               self.approximator_fn,
                               self.state_shape)

@gin.configurable(blacklist=["scope", "state_shape", "action_shape"])
class GaussianPolicy(ProbabilisticPolicy):
    """ Continuous Gaussian Policy
    """

    name_scope = "gaussian_policy"

    def __init__(self,
                 scope,
                 approximator,
                 state_shape,
                 action_shape):
        self._mean = None
        self._sigma = None

        super().__init__(self, scope, approximator, state_shape, action_shape)

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

    def _sample(self, session, states):

        mean, sigma = self._eval_func(session, states)
        covariance = np.diag(np.exp(sigma))

        return np.random.multivariate_normal(mean=mean, cov=covariance)

    def _policy(self, network):
        """Useful for constructing output layers for a multinomial policy settings
                Args:
                    network: output of network to pass in
                Returns:
                    mean and sigma gaussian policy
        """

        mean = tf.layers.dense(network, self._action_shape, activation=None)
        sigma = tf.layers.dense(tf.ones([1, self._action_shape]),
                                self._action_shape,
                                activation=None,
                                use_bias=False)
        self._mean = mean
        self._sigma = sigma

        return mean, sigma

    def _make_liklehood(self):
        action_dim = self.action_shape[1]
        first_add = -(action_dim/ 2) * tf.log(2 * np.pi)
        sec_add = -tf.reduce_sum(self.sigma, axis=1)
        third_add = -0.5 * tf.reduce_sum(((self._action_taken_plh - self.mean) / tf.exp(self.sigma)) ** 2, axis=1)

        return first_add + sec_add + third_add

    def copy_obj(self, scope):
        approx_scope = ScopeWrap.build(scope, self.approximator_fn.name_scope)

        return GaussianPolicy(approx_scope,
                              self.approximator_fn,
                              self.state_shape,
                              self.action_shape)

class Policies(Enum):
    """Possible Policies to select
    """
    VALUE = ValueFunction
    CONT_ACTION_VALUE = ContinousActionValueFunction
    CONT_REAL = ContinuousRealPolicy
    MULTINOMIAL = MultinomialPolicy
    BERNOULLI = BernoulliPolicy
    GAUSSIAN = GaussianPolicy
