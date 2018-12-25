from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from advantage.buffers import ReplayBuffer
from advantage.utils.tf_utils import ScopeWrap
from advantage.elements import NStepSarsa

""" This module consists of common objectives utilized in Deep
RL. These objectives are pretty much plug-and-play allowing
them to adjust to many popular RL algorithms.

    The main two forms of model-free RL algorithms are:
        Value-Gradient
        Actor-Critic: an approximation to the full return Policy Gradient
"""

class Objective(metaclass=ABCMeta):
    """ Objectives represent the goal/loss
    an Agent is looking to optimization utilizing
    partial trajectory samples
    """

    pass


class NStepAdvantageObjective(Objective):
    """ Used for an agent
    that need to minimize the Advantage
    or Bellman error. The policy converges
    when it can accurately predict the expected
    reward given a state, value pairs.
    This is also known as the `Value Gradient`
    """
    def __init__(self,
                 replay_buffer,
                 element_cls,
                 discount_factor,
                 value_func,
                 iteration_of_optimization,
                 steps=None):
        """
            Args:
                replay_buffer: This objective uses
                    a `Buffer` as it's source
                    of `Elements`
                element_cls: The cls for the `Element`
                    stored in the `Buffer`
                discount_factor : factor used for exponential reward sum
                value_function : value function
                    to optimize
                iterations_of_optimization: times to update for one
                    `optimization` call
                steps: number of step for bellman operator,
                    None means until terminal state

        """

        if not isinstance(element_cls, NStepSarsa):
            raise ValueError("NStepAdvantageObjective requires `element_cls`"
                             " to be an instance of `NStepSarsa`")

        self._replay_buffer = replay_buffer
        self._element_cls = element_cls

        self._discount_factor = discount_factor
        self._iterations_of_optimization = iteration_of_optimization
        self._steps = steps
        self._value_func = value_func

        self._objective = None
        self._gradients = None

        self._waiting_buffer = ReplayBuffer(steps)

    @property
    def value_func(self):
        """ property for `_value_func`
        """
        return self._value_func

    @property
    def bootstrap_func(self):
        """ property for bootstrapping func
        """
        return self._value_func

    def _add_reward(self, reward):
        """ Integrates `reward` into the
        discounted sum kept by all `Elements`
        in the `_waiting_buffer`

            Args:
                reward: reward to integrat, from
                    the latest `Element`
        """

        buffer_len = self._waiting_buffer.len

        stacked = self._element_cls.stack(self._waiting_buffer.sample(buffer_len))
        factors = self._discount_factor ** (np.arange(1, buffer_len)[-1:])
        discounted = factors * reward

        stacked["n_step_return"] += discounted

    def set_up(self,
               session,
               regularizer=None):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
                regularizer: regularizer to use
        """
        self._value_func.set_up(session)

        with tf.name_scope("n_step_bellman_objective"):
            # the bellman values for training value_func
            bellman_plh = tf.placeholder(shape=[None, 1],
                                         dtype=tf.float32,
                                         name="bellman_target_plh")

            self._value_func.add_target_placeholder(bellman_plh)

            self._objective = tf.reduce_mean((1/2) * tf.square(bellman_plh - self._value_func.func),
                                             name="objective")

            if regularizer:
                self._objective += regularizer(self._value_func)

            gradients = self._value_func.gradients(self._objective)
            self._gradients = tf.reduce_mean(gradients,
                                             name="reduce_mean_gradients")

            self._value_func.apply_gradients(gradients)

    def push(self, env_dict):
        """ Accounts for new env_dict
        element

            Args:
                env_dict: sampled from `Environment`
        """

        element = self._element_cls.make_element_from_env(env_dict)
        buffer_len = self._waiting_buffer.len

        if buffer_len:
            self._add_reward(element.reward)

        if element.done or buffer_len == self._steps:
            if not element.done:
                boostrap_func = self.bootstrap_func
                self._add_reward(boostrap_func([element.next_state]))

            self._replay_buffer.push(self._waiting_buffer.sample_and_pop(buffer_len))

        else:
            self._waiting_buffer.push(element)

    def fetch_gradient(self,
                       session,
                       batch_size,
                       sample_less=False,
                       pop=True):

        for batch in self._replay_buffer.sample_batches(batch_size,
                                                        self._iterations_of_optimization,
                                                        sample_less=sample_less,
                                                        pop=pop):

            stacked = self._element_cls.stack(batch)

            yield self._value_func.fetch_mean_gradient(session,
                                                       {"state": stacked["state"]},
                                                       {"bellman_target_plh": stacked["n_step_return"]})


    def optimize(self,
                 session,
                 batch_size,
                 sample_less=False,
                 pop=True,
                 fetch_gradient=False):
        """ Performs optimization step
        for `value_func`

            Args:
                session: tf.Session
                batch_size: training batch size
                sample_less: whether sampling less than
                    a batch is allowed
                pop: whether elements should be
                    removed from `_replay_buffer`
                    when training
        """
        for batch in self._replay_buffer.sample_batches(batch_size,
                                                        self._iterations_of_optimization,
                                                        sample_less=sample_less,
                                                        pop=pop):

            stacked = self._element_cls.stack(batch)

            self._value_func.update(session,
                                    {"state": stacked["state"]},
                                    {"bellman_target_plh": stacked["n_step_return"]})


class DecoupledNStepAdvantageObjective(NStepAdvantageObjective):
    """ Represents a Bellman objective
    in which the bootstrapping func is not the same
    value function whose parameters are being
    optimization.
    """

    def __init__(self,
                 bootstrap_func,
                 *args,
                 **kwargs):
        """
            Args:
                bootstrap_func: function for bootstrapping
        """
        self._bootstrap_func = bootstrap_func

        self._copy = None

        super().__init__(*args,
                         name_scope="decoupled_n_step_bellman_objective",
                         **kwargs)

    @property
    def boostrap_func(self):
        """ property for custom boostrap_func
        """
        return self._bootstrap_func

    def set_up(self, session, regularizer=None):
        """ Builds all necessary TF-Graph
                components

                Args:
                    session: tf.Session
                    regularizer: regularizer to use
        """

        super().set_up(session, regularizer=regularizer)

        self._bootstrap_func.set_up(session)

        self._copy = self._bootstrap_func.make_copy_op(session,
                                                       self._value_func)

    def sync(self):
        """ Synchronizes `bootstrap_func`
        and `value_func` parameters
        """
        self._copy()


class PolicyGradientObjective(Objective):
    """ Contains information related
    to specifying an objective for an
    `Actor` or Policy related objective
    """
    def __init__(self,
                 upper_scope,
                 replay_buffer,
                 element_cls,
                 policy_func,
                 policy_return,
                 iteration_of_optimization,
                 from_gradient=False,
                 name_scope=""):

        self._policy_func = policy_func
        self._policy_return = policy_return

        self._replay_buffer = replay_buffer
        self._element_cls = element_cls

        self._iterations_of_optimization = iteration_of_optimization
        self._from_gradient = from_gradient

        self._gradients = None

        scope = ScopeWrap.build(upper_scope, name_scope)
        self._scope = ScopeWrap.build(scope, "policy_gradient_objective")

        self._action_taken_plh = None
        self._next_state_plh = None


    def action_taken_plh(self, action_shape=None):
        """fetching a placeholder for taking the selected
        action in the expected_return
            Args:
                action_shape: shape for action placeholder

            Returns:
                placeholder
        """
        if not self._action_taken_plh:
            with self._scope():
                self._action_taken_plh = tf.placeholder(shape=[None, action_shape],
                                                        dtype=tf.float32,
                                                        name="action_taken")
                self._policy_func.add_target_placeholder(self._action_taken_plh)

        return self._action_taken_plh

    def next_state_plh(self, state_shape=None):
        """fetching a placeholder for taking the next state
        in the expected_return
            Args:
                state_shape: shape for next_state placeholder

            Returns:
                placeholder
        """
        if not self._next_state_plh:
            with self._scope():
                self._next_state_plh = tf.placeholder(shape=[None, state_shape],
                                                      dtype=tf.float32,
                                                      name="next_state")
                self._policy_func.add_target_placeholder(self._next_state_plh)

        return self._next_state_plh

    def set_up(self,
               session,
               regularizer=None):

        self._policy_func.set_up(session)

        with self._scope():

            if self._from_gradient:
                gradients = self._policy_func.from_gradient_func(self._policy_return)
            else:
                expected_return = tf.reduce_mean(self._policy_return,
                                                 name="reduce_mean_return")
                gradients = self._policy_func.gradients(expected_return)

            self._gradients = tf.reduce_mean(gradients,
                                             name="reduce_mean_gradients")

            self._policy_func.apply_gradients(gradients)

    def push(self, env_dict):
        """ Accounts for new env_dict
        element

            Args:
                env_dict: sampled from `Environment`
        """
        pass

    def fetch_gradient(self,
                       session,
                       batch_size,
                       sample_less=False,
                       pop=True):

        for batch in self._replay_buffer.sample_batches(batch_size,
                                                        self._iterations_of_optimization,
                                                        sample_less=sample_less,
                                                        pop=pop):

            stacked = self._element_cls.stack(batch)

            yield self._policy_func.fetch_mean_gradient(session,
                                                        {"state": stacked["state"]},
                                                        {"action_taken": stacked["action"],
                                                         "next_state": stacked["next_state"]})


    def optimize(self,
                 session,
                 batch_size,
                 sample_less=False,
                 pop=True):

        for batch in self._replay_buffer.sample_batches(batch_size,
                                                        self._iterations_of_optimization,
                                                        sample_less=sample_less,
                                                        pop=pop):

            stacked = self._element_cls.stack(batch)

            self._policy_func.update(session,
                                     {"state": stacked["state"]},
                                     {"action_taken": stacked["action"],
                                      "next_state": stacked["next_state"]})

class MultiActorCriticObjective(Objective):
    """ Represents the objective for
    and Actor-Critic Agent. Objectives
    must be optimized with respect
    to `Approximator` parameters.

    An Objective can have a `Buffer`
    attached to it to use as the source
    for computing the current objective values
    to determine direction for optimization (Gradient)

    This is commonly used to implement `Policy Gradient`
    methods. Substituing N-Step returns for the
    full return.
    """

    def __init__(self,
                 adv_objective,
                 policy_grad_objectives,
                 iteration_of_optimization):
        """
            Args:
                replay_buffer: This objective uses
                    a `Buffer` as it's source
                    of `Elements`
                element_cls: The cls for the `Element`
                    stored in the `Buffer`
                discount_factor : factor used for exponential reward sum
                value_function : critic state-value function
                policy_grad_objectives: list of
                    PolicyGradientObjective objects representing
                    `Actors`
                update_cond: condition specifying when function
                    updates should occur. For example this is a function
                    of the form `lambda step_num, in_term_state: ...`
                    The `step_num` is the number of steps passed
                    and the `in_term_state` is whether last state
                    resulted in a transition to a terminal state
                iterations_of_optimization: times to update for one
                    `optimization` call
                max_steps: number of steps to run policy for
        """

        self._adv_objective = adv_objective
        self._policy_grad_objectives = policy_grad_objectives

        self._iterations_of_optimization = iteration_of_optimization
        self._gradients = None

        self._in_term_state = False
        self._num_steps = 0

    def set_up(self,
               session,
               regularizer=None):
        """ Builds all necessary TF-Graph
        components

            Args:
                session: tf.Session
                regularizer: regularizer to use
        """

        self._adv_objective.set_up(session)

        for policy_grad_objective in self._policy_grad_objectives:
            policy_grad_objective.set_up(session)

    def push(self, env_dict):
        """ Accounts for new env_dict
        element

            Args:
                env_dict: sampled from `Environment`
        """

        self._adv_objective.push(env_dict)

    def optimize(self,
                 session,
                 batch_size,
                 sample_less=False,
                 pop=True):
        """ Perform an optimization step
        for the Value Function with `batch`

            Args:
                session: tf.Session
        """

        self._adv_objective.optimize(session,
                                     batch_size,
                                     sample_less=sample_less,
                                     pop=False)

        for policy_grad_objective in self._policy_grad_objectives[:-1]:
            policy_grad_objective.optimize(session,
                                           batch_size,
                                           sample_less=sample_less,
                                           pop=False)

        self._policy_grad_objectives[-1].optimize(session,
                                                  batch_size,
                                                  sample_less=sample_less,
                                                  pop=True)

class DistributedMultiActorCriticObjective(MultiActorCriticObjective):
    """ Represents the objective used by Distributed Asynchronous
    Actor-Critic agents. Gradients are updated in a `master-slave`
    fashion
    """
    pass

class MetaLearningObjective(Objective):
    """ Represents the task of `Meta-Learning`
    in RL. This is where the agents learn
    a more general loss/reward function that can be
    reused.
    """
    pass
