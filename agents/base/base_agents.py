from functools import partial
import gym
import random
import numpy as np
import tensorflow as tf
from abc import ABCMeta
from abc import abstractmethod
import advantage.loggers as loggers

class LearningAgent(metaclass=ABCMeta):
    """ Represents the general Learning (model-free) Reinforcement Learning
    agent. An agent has an environment in which it acts in
    and a policy which it follows. The reward observed
    is then used to modify the policy.

    All Learning RL Agents follow the General Policy Iteration
    formulation.
    """

    def __init__(self,
                 policy,
                 environment,
                 graph,
                 agent_scope,
                 discount_factor):

        self._policy = policy
        self._environment = environment
        self._done = True
        self._state = None
        self._total_steps = 0 # total number of steps gone
        self._traj_steps = 0 # current steps for trajectory
        self._num_traj = 0 # total number of trajectories completed
        self._traj_reward = -1 # current trajectory reward
        self._dis_traj_reward = 0 # current discounted trajectory reward
        self._discount_factor = discount_factor
        self._graph = graph
        self._session = None
        self._agent_scope = agent_scope
        self.info_log_frequency = None

    @property
    def discount_factor(self):
        """ _discount_factor property
        """
        return self._discount_factor

    @property
    def traj_reward(self):
        """ _traj_reward property
        """
        return self._traj_reward

    @property
    def num_traj(self):
        """ _num_traj property
        """
        return self._num_traj

    @property
    def dis_traj_reward(self):
        """ _dis_traj_reward property
        """
        return self._dis_traj_reward

    @property
    def traj_steps(self):
        """ _traj_steps property
        """
        return self._traj_steps

    @property
    def total_steps(self):
        """ _total_steps property
        """
        return self._total_steps

    @property
    def policy(self):
        """ _policy property
        """
        return self._policy

    @property
    def environment(self):
        """ _environment property
        """
        return self._environment

    @property
    def action_space(self):
        """ environment action_space as property
        """
        return self.environment.action_space

    @property
    def session(self):
        """ property for `_session`
        """
        if not isinstance(self._session, tf.Session):
            raise AttributeError("session not set for agent")

        return self._session

    @session.setter
    def session(self, sess):
        """ setter for `_session`
        """
        self._session = sess

    @property
    def graph(self):
        """ property for `_graph`
        """
        return self._graph

    @property
    def agent_scope(self):
        """ property for `_agent_scope`
        """
        return self._agent_scope

    @abstractmethod
    def set_up_train(self):
        """ Performs setup operations for the Agents
        for training
        """
        raise NotImplementedError()

    @abstractmethod
    def set_up(self):
        """ Performs setup operations for the Agents
        for  inference
        """
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
    def improve_policy(self):
        """ Improves the current policy based on Information
        observed from evaluation. Subclasses define other
        attrs to use this.
        """
        raise NotImplementedError()

    def act_in_env(self, training):
        """ Agent acts in the environment
            Args:
                training: whether the agent is training

            Returns:
                dict containing one-step state infor
                {'state':, 'action':, 'reward':, 'done':, 'next_state':}
                values are in whatever format the environment and agent return
        """
        was_done = False
        if self._done:
            was_done = True
            self._state = self._environment.reset()
            self._dis_traj_reward = 0.
            self._traj_reward = 0.
            self._traj_steps = 0
            self._num_traj += 1

        prev_state = self._state
        conditional_policy = self.evaluate_policy(prev_state)
        action = self.sample_action(conditional_policy, training)
        self._state, reward, self._done = self._environment.step(action)
        self._total_steps += 1
        self._traj_steps += 1

        self._dis_traj_reward += (self._discount_factor) * reward if not was_done else reward
        self._traj_reward += reward

        return {"state": prev_state,
                "action": action,
                "reward": reward,
                "done": self._done,
                "next_state": self._state}

    def act_for_steps(self, num_steps, training):
        """ Generator: Agent acts in environment for num_steps
                Args:
                    num_steps: maximum number of steps to act for
                    training: whether the agent is training

                Yields:
                     return value of act_in_env

        """
        for _ in range(1, num_steps + 1):
            yield self.act_in_env(training)

    def act_for_trajs(self, num_traj, training):
        """ Generator Agent acts in environment until
        completing a specified number of trajectories
            Args:
                num_steps: maximum number of trajectories to act
                for
                training: whether training
            Yields:
                return value of act_in_env
        """
        for _ in range(1, num_traj + 1):
            yield from self.run_trajectory(training)

    @loggers.avg("Agent average trajectory reward is %.2f",
                 loggers.LogVarType.INSTANCE_ATTR,
                 "traj_reward",
                 (loggers.LogVarType.RETURNED_DICT, "done", lambda done: done),
                 tensorboard=True)
    def run_trajectory(self, training=False):
        """ Generator: Runs a trajectory. This means the agent
        acts until a termination signal from the environment
        is received. Not for training.
            Yields:
                return env_info from act_in_env
        """
        terminate = False
        while not terminate:
            env_info = self.act_in_env(training)
            terminate = bool(env_info["done"])
            yield env_info

    def run_trajectory_through(self, training=False):
        """ Runs the traject until complete returning no info
        during run.

            Returns:
                number of steps agent went through
        """
        steps = 0
        while not self.act_in_env(training)["done"]:
            steps += 1
        return steps

# pylint: disable=W0223
# reason-disabled: this class is also abstract
class OffPolicyValueAgent(LearningAgent, metaclass=ABCMeta):
    """ Represents an RL Agent that is Value-function based using Sample-based RL.
    Also known as Temporal-Difference Learning. It computes the expected Reward utilizing
    samples from the environment and boostrapping. The Action-Value is used for control.
    The agent usually is used in a Discrete Action space
    """

    def __init__(self, environment, *args, **kwargs):
        action_space = environment.action_space
        self._num_of_actions = action_space.n

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("environment.action_space must"
                             " be of type gym.spaces.Discrete ")

        self._maximum_function = partial(np.argmax, axis=1)

        # calculates epsilon during training
        # set by subclass (only for training)
        self.epsilon_func = lambda: None

        super().__init__(environment=environment,
                         *args,
                         **kwargs)

    @property
    def num_of_actions(self):
        """ _num_of_actions property
        """
        return self._num_of_actions

    @property
    def maximum_function(self):
        """ property for `_maximum_function`
        """
        return self._maximum_function

    def _act_epsilon_greedily(self, sampled_action):
        """ Follows an epsilon-greedy policy
        for off-policy training

            Args:
                sampled_action: action sampled from p(a|s)

            Returns:
                possible-randomly chosen action
        """

        eps = self.epsilon_func()

        if not eps:
            raise AttributeError("Subclass must set epsilon_func to valid callable")

        if not 0. < eps <= 1.:
            raise ValueError("Epsilon must be in (0, 1.0]")

        prob = random.random()
        return sampled_action if prob > eps else self.action_space.sample()

    def sample_action(self, conditional_policy, training):
        """ Samples an action from the conditional policy
        distribution

            Args:
                conditional_policy : pi(a|s)
                training: whether agent is training

            Returns:
                sampled action
        """
        sampled = self._maximum_function(conditional_policy)[0]
        return sampled if not training else self._act_epsilon_greedily(sampled)

class PolicyGradientAgent(LearningAgent, metaclass=ABCMeta):
    """ Represents an RL Agent using Policy Gradients. The agent computes the
    gradient of the expected reward function and uses it to directly update the policy
    """

    def __init__(self, policy,
                 environment,
                 discount_factor,
                 expected_reward,
                 **kwargs):
        self._expected_reward = expected_reward

        super().__init__(policy=policy,
                         environment=environment,
                         discount_factor=discount_factor,
                         **kwargs)

    @property
    def expected_reward(self):
        """ expected_reward property
        """
        return self._expected_reward

    @abstractmethod
    def _compute_policy_gradient(self):
        """ Computes the policy Gradient
        """
        raise NotImplementedError()


class ContinuousActionSpaceAgent(LearningAgent, metaclass=ABCMeta):
    """ Represents an RL Agent with a continuous action space """

    def __init__(self, policy,
                 environment,
                 discount_factor,
                 **kwargs):

        action_space = self._environment.action

        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("Environment for ContinuousActionSpaceAgent must "
                             "have a continuous aciton-space!")

        self._action_low = action_space.low
        self._action_high = action_space.high
        self._action_shape = action_space.shape

        super().__init__(policy=policy,
                         environment=environment,
                         discount_factor=discount_factor,
                         **kwargs)

    @property
    def action_low(self):
        """ action_low property
        """
        return self._action_low

    @property
    def action_high(self):
        """ action_high property
        """
        return self._action_high

    @property
    def action_shape(self):
        """ action_shape property
        """
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

        # pylint: disable=line-too-long
        # reason-disabled: if statement cond length ok
        if not (conditional_policy < self._action_high).all() or not (conditional_policy > self._action_low).all():
            raise Exception() # TODO: replace with specific exception

        return action
