import gym
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

class LearningAgent(metaclass=ABCMeta):
    """ Represents the general Learning (model-free) Reinforcement Learning
    agent. An agent has an environment in which it acts in
    and a policy which it follows. The reward observed
    is then used to modify the policy.

    All Learning RL Agents follow the General Policy Iteration
    formulation.
    """

    # pylint: disable=unused-argument
    # reason-disabled: kwargs used in multiple inheritance super calls
    def __init__(self, policy, environment, discount_factor, **kwargs):
        self._policy = policy
        self._environment = environment
        self._done = True
        self._state = None
        self._total_steps = 0 # total number of steps gone
        self._traj_steps = 0 # current steps for trajectory
        self._traj_reward = 0 # current trajectory reward
        self._dis_traj_reward = 0 # current discounted trajectory reward
        self._discount_factor = discount_factor

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
    def improve_policy(self, samples):
        """ Improves the current policy based on Information
        observed from evaluation.
            Args:
                samples: samples to improve using
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
                    step number [starting at 1], return value of act_in_env

        """
        for step in range(1, num_steps + 1):
            yield (step, self.act_in_env(training))

    def run_trajectory(self):
        """ Generator: Runs a trajectory. This means the agent
        acts until a termination signal from the environment
        is received. Not for training.
            Yields:
                return env_info from act_in_env
        """
        terminate = False
        while not terminate:
            env_info = self.act_in_env(False)
            terminate = bool(env_info["done"])
            yield env_info

    def run_trajectory_through(self):
        """ Runs the traject until complete returning no info
        during run.

            Returns:
                number of steps agent went through
        """
        steps = 0
        while not self.act_in_env(False)["done"]:
            steps += 1
        return steps


class ActionValueAgent(LearningAgent, metaclass=ABCMeta):
    """ Represents an RL Agent that is Value-function based using Sample-based RL.
    Also known as Temporal-Difference Learning. It computes the expected Reward utilizing
    samples from the environment and boostrapping. The Action-Value is used for control.
    """

    def __init__(self,
                 policy,
                 environment,
                 discount_factor,
                 maximum_function,
                 **kwargs):

        self._maximum_function = maximum_function
        super().__init__(policy=policy,
                         environment=environment,
                         discount_factor=discount_factor,
                         **kwargs)

    @property
    def maximum_function(self):
        return self._maximum_function

    def sample_action(self, conditional_policy, training):
        """ Action is chosen as the action with maximum action-value
        """
        return self._maximum_function(conditional_policy)

class PolicyGradientAgent(LearningAgent, metaclass=ABCMeta):
    """ Represents an RL Agent using Policy Gradients. The agent computes the
    gradient of the expected reward function and uses it to directly update the policy
    """

    def __init__(self, policy,
                 environment,
                 discount_factor,
                 expected_reward,
                 **kwargs):
        self._expected_reward = expected_reward # TODO create a special expected reward class

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
        # TODO not finished...not sure how this will be used yet.
        raise NotImplementedError()

class DiscreteActionSpaceAgent(LearningAgent, metaclass=ABCMeta):
    """ Represents an RL Agent with a discrete agent space
    """

    def __init__(self, policy,
                 environment,
                 discount_factor,
                 **kwargs):
        action_space = environment.action_space

        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("environment.action_space must"
                             " be of type gym.spaces.Discrete ")

        self._num_of_actions = action_space.n
        self.sample_action = self._action_wrapper(self.sample_action)
        super().__init__(policy=policy,
                         environment=environment,
                         discount_factor=discount_factor,
                         **kwargs)

    @property
    def num_of_actions(self):
        """ _num_of_actions property
        """
        return self._num_of_actions

    def _action_wrapper(self, func):
        """ It's hard to say where this goes atm, but it's related to the fact
        that np.argmax on a (1, N) ndarray returns a one element array...
        Might need to go in ActionValueAgent, which could technically generalize to Continuous
        Agents, where this would then pose a problem.
        """
        #TODO Look for a better place to put this...
        def wrapper(conditional_policy, training):
            action = func(conditional_policy, training)
            if isinstance(action, np.ndarray):
                return int(action[0])
            return int(action)
        return wrapper

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
