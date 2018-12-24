from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from advantage.buffers import ReplayBuffer
from advantage.utils.tf_utils import ScopeWrap
from advantage.elements import NStepSarsa

""" Objectives are minimized/maximized by an `Agent`
as is the typical task of an Machine Learning model
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
    reward given a state, value pairs
    """
    def __init__(self,
                 replay_buffer,
                 element_cls,
                 discount_factor,
                 value_func,
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
                steps: number of step for bellman operator,
                    None means until terminal state

        """

        if not isinstance(element_cls, NStepSarsa):
            raise ValueError("NStepAdvantageObjective requires `element_cls`"
                             " to be an instance of `NStepSarsa`")

        self._replay_buffer = replay_buffer
        self._element_cls = element_cls

        self._discount_factor = discount_factor
        self._steps = steps
        self._value_func = value_func

        self._objective = None
        self._gradients = None

        self._waiting_buffer = ReplayBuffer(steps)

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

        stacked["n_step_reward"] += discounted

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

            self._objective = tf.reduce_mean(tf.square(bellman_plh - self._value_func.func),
                                             name="objective")

            if regularizer:
                self._objective += regularizer(self._value_func)

            self._gradients = tf.reduce_mean(self._value_func.gradients(self._objective),
                                             name="reduce_mean_gradients")

            self._value_func.apply_gradients(self._gradients)

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
                       sample_less=True):
        """
        """
        #stacked = self._element_cls.stack(batch)
        pass

    def optimize(self,
                 session,
                 batch,
                 iterations):
        """ Perform an optimization step
        for the Value Function with `batch`

            Args:
                session: tf.Session
                batch: list of `element_cls`
                iterations : iterations of optimization
        """

        stacked = self._element_cls.stack(batch)

        self._value_func.update(session,
                                {"state": stacked["state"]},
                                {"bellman_target_plh": stacked["n_step_reward"]})


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



class ActorCriticObjective(Objective):
    """ Represents the objective for
    and Actor-Critic Agent. Objectives
    must be optimized with respect
    to `Approximator` parameters.

    An Objective can have a `Buffer`
    attached to it to use as the source
    for computing the current objective values
    to determine direction for optimization (Gradient)
    """

    def __init__(self,
                 replay_buffer,
                 element_cls,
                 discount_factor,
                 value_func,
                 prob_policy,
                 steps=None):
        """
            Args:
                replay_buffer: This objective uses
                    a `Buffer` as it's source
                    of `Elements`
                element_cls: The cls for the `Element`
                    stored in the `Buffer`
                discount_factor : factor used for exponential reward sum
                value_function : critic state-value function
                prob_policy : probabilitic policy
                steps: number of step for bellman operator,
                    None means until terminal state
        """

        self._replay_buffer = replay_buffer
        self._element_cls = element_cls
        self._discount_factor = discount_factor
        self._value_func = value_func
        self._prob_policy = prob_policy

        self._adv_objective = NStepAdvantageObjective
