import gym
import numpy as np
from abc import ABCMeta
from abc import abstractmethod
from utils.buffers import Sarsa


class LearningAgent(object):
    __metaclass__ = ABCMeta
    """ Represents the general Learning Reinforcement Learning
    agent. An agent has an environment in which it acts in
    and a policy which it follows. The reward observed
    is then used to modify the policy.

    All Learning RL Agents follow the General Policy Iteration
    formulation.
    """

    def __init__(self, policy, environment, **kwargs):
        self._policy = policy
        self._environment = environment
        self._done = True
        self._state = None
        self._steps = 0 # total number of steps gone
        self._total_reward = 0


    @property
    def total_reward(self):
        return self._total_reward

    @property
    def steps(self):
        return self._steps

    @property
    def policy(self):
        return self._policy

    @property
    def environment(self):
        return self._environment

    @property
    def action_space(self):
        return self.environment.action_space

    @abstractmethod
    def set_up(self):
        """Performs setup operations for the Agents"""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_policy(self, state):
        """ Evaluation of a policy. The probability
        distribution of actions given the current state. Information
        about policy evaluation is stored to be used upon improvement.

            Args:
                state -> the state representation

            Returns:
                pi -> the conditional probability distribution of actions
                given current state
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_action(self, conditional_policy, training):
        """ Samples an action from the policy distribution given the action-space.

            Args:
                conditional_policy -> policy conditioned on a state
                training -> whether agent is training
            Returns:
                action -> an action within the action-space.
        """
        raise NotImplementedError()


    @abstractmethod
    def improve_policy(self, sarsa_samples):
        """ Improves the current policy based on Information
        observed from evaluation.
            Args:
                sarsa_samples: samples to improve using
        """
        raise NotImplementedError()

    def act_in_env(self, training):
        """ Agent acts in the environment
            Args:
                training: whether the agent is training

            Return:
                sarsa element
        """
        if self._done:
            s = self._environment.reset()
        conditional_policy = self.evaluate_policy(s)
        a = self.sample_action(conditional_policy, training)
        self._state, reward, self._done, _ = self._environment.step(a)
        self._steps += 1
        self._total_reward += reward
        return Sarsa(state=s, action=a,
                    reward=reward, done=self._done,
                    next_state=self._state, next_action=None)

    def act_for_steps(self, num_steps, training):
        """ Agent acts in environment for num_steps
                Args:
                    num_steps: maximum number of steps to act for
                    training: whether the agent is training

                Returns: step number, return value of act_in_env

                Raises:
                    StopIteration: after num_steps
        """
        for step in range(0, num_steps):
            yield (step, self.act_in_env(training))

        raise StopIteration("Max number of steps exceeded")



class ActionValueAgent(LearningAgent):
    __metaclass__ = ABCMeta
    """ Represents an RL Agent that is Value-function based using Sample-based RL.
    Also known as Temporal-Difference Learning. It computes the expected Reward utilizing
    samples from the environment and boostrapping. The Action-Value is used for control.
    """

    def __init__(self, policy, environment,  maximum_function, **kwargs):
        self._maximum_function = maximum_function
        super().__init__(policy=policy, environment=environment, **kwargs)

    @property
    def maximum_function(self):
        return self._maximum_function

    def sample_action(self, conditional_policy, training):
        """ Action is chosen as the action with maximum action-value
        """
        return self._maximum_function(conditional_policy)

class PolicyGradientAgent(LearningAgent):
    __metaclass__ = ABCMeta
    """ Represents an RL Agent using Policy Gradients. The agent computes the
    gradient of the expected reward function and uses it to directly update the policy
    """

    def __init__(self, policy, environment, expected_reward, **kwargs):
        self._expected_reward = expected_reward # TODO create a special expected reward class

        super(PolicyGradientAgent, self).__init__(policy=policy, environment=environment, **kwargs)

    @property
    def expected_reward(self):
        return self._expected_reward

    @abstractmethod
    def __compute_policy_gradient(self):
        """ Computes the policy Gradient """ # TODO not finished...not sure how this will be used yet.
        raise NotImplementedError()

class DiscreteActionSpaceAgent(LearningAgent):
    __metaclass__ = ABCMeta
    """ Represents an RL Agent with a discrete agent space """


    def __init__(self, policy, environment, **kwargs):
        action_space = environment.action_space

        if not isinstance(action_space, gym.spaces.Discrete):
            raise Exception() # TODO: replace with specific exception

        self._num_of_actions = action_space.n
        self.sample_action = self._action_wrapper(self.sample_action)
        super().__init__(policy=policy, environment=environment, **kwargs)

    @property
    def num_of_actions(self):
        return self._num_of_actions

    def _action_wrapper(self, fn):
        """ It's hard to say where this goes atm, but it's related to the fact
        that np.argmax on a (1, N) ndarray returns a one element array...
        Might need to go in ActionValueAgent, which could technically generalize to Continuous
        Agents, where this would then pose a problem.
        """
        #TODO Look for a better place to put this...
        def wrapper(conditional_policy, training):
            action = fn(conditional_policy, training)
            if isinstance(action, np.ndarray):
                return int(action[0])
            else:
                return int(action)
        return wrapper

class ContinuousActionSpaceAgent(LearningAgent):
    __metaclass__ = ABCMeta
    """ Represents an RL Agent with a continuous action space """

    def __init__(self, policy, environment, **kwargs):

        action_space = self._environment.action

        if not isinstance(action_space, gym.spaces.Box):
            raise Exception() # TODO: replace with specific exception

        self._action_low = action_space.low
        self._action_high = action_space.high
        self._action_shape = action_space.shape

        super(ContinuousActionSpaceAgent, self).__init__(policy=policy, environment=environment, **kwargs)

    @property
    def action_low(self):
        return self._action_low

    @property
    def action_high(self):
        return self._action_high

    @property
    def action_shape(self):
        return self._action_shape

    @abstractmethod
    def _continuous_distribution_sample(self, conditional_policy):
        """ Continuous Policies output a continuous distribution over actions.
            This samples an action from that distribution
        """
        raise NotImplementedError()

    def sample_action(self, conditional_policy, training):
        action = self._continuous_distribution_sample(conditional_policy)

        if not isinstance(conditional_policy, np.ndarray):
            raise Exception() # TODO: replace with specific exception

        if not (conditional_policy < self._action_high).all() or not (conditional_policy > self._action_low).all():
            raise Exception() # TODO: replace with specific exception
