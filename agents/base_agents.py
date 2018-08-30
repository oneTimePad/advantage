from abc import ABCMeta
from abc import abstractmethod


class LearningAgent(object):
    __metaclass__ = ABCMeta
    """ Represents the general Learning Reinforcement Learning
    agent. An agent has an environment in which it acts in
    and a policy which it follows. The reward observed
    is then used to modify the policy.

    All Learning RL Agents follow the General Policy Iteration
    formulation.
    """

    def __init__(self, policy, environment):
        self._policy = policy
        self._environment = environment

    @property
    def policy(self):
        return self._policy

    @property
    def environment(self):
        return self._environment

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
    def sample_action(self, conditional_policy):
        """ Samples an action from the policy distribution given the action-space.

            Args:
                conditional_policy -> policy conditioned on a state
            Returns:
                action -> an action within the action-space.
        """

    @abstractmethod
    def improve_policy(self):
        """ Improves the current policy based on Information
        observed from evaluation.
        """
        raise NotImplementedError()


class ActionValueAgent(LearningAgent):
    __metaclass__ = ABCMeta
    """ Represents an RL Agent that is Value-function based using Sample-based RL.
    Also known as Temporal-Difference Learning. It computes the expected Reward utilizing
    samples from the environment and boostrapping. The Action-Value is used for control.
    """

    def __init__(self, policy, environment, value_function, maximum_function):
        self._value_function = value_function
        self._maximum_function = maximum_function

        super(ValueAgent, self).__init__(policy, environment)

    @property
    def value_function(self):
        return self._value_function

    @property
    def maximum_function(self):
        return self._maximum_function

    def sample_action(self, conditional_policy):
        """ Action is chosen as the action with maximum action-value
        """
        return self._maximum_function(policy)

class PolicyGradientAgent(LearningAgent):
    __metaclass__ = ABCMeta
    """ Represents an RL Agent using Policy Gradients. The agent computes the
    gradient of the expected reward function and uses it to directly update the policy
    """

    def __init__(self, policy, environment, expected_reward):
        self._expected_reward = expected_reward # TODO create a special expected reward class

        super(PolicyGradientAgent, self).__init__(policy, environment)

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


    def __init__(self, policy, environment):

        action_space = environment.action_space

        if not isinstance(action_space, gym.Discrete):
            raise Exception() # TODO: replace with specific exception

        self._num_of_actions = action_space.n

        super(DiscreteActionSpaceAgent, self).__init__(policy, environment)

    @property
    def num_of_actions(self):
        return self._num_of_actions

class ContinuousActionSpaceAgent(LearningAgent):
    __metaclass__ = ABCMeta
    """ Represents an RL Agent with a continuous action space """

    def __init__(self, policy, environment):

        action_space = self._environment.action

        if not isinstance(action_space, gym.Box):
            raise Exception() # TODO: replace with specific exception

        self._action_low = action_space.low
        self._action_high = action_space.high
        self._action_shape = action_space.shape

        super(ContinuousActionSpaceAgent, self).__init__(policy, environment)

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

    def sample_action(self, conditional_policy):
        action = self._continuous_distribution_sample(conditional_policy)

        if not isinstance(conditional_policy, np.ndarray):
            raise Exception() # TODO: replace with specific exception

        if not (conditional_policy < self._action_high).all() or not (conditional_policy > self._action_low).all():
            raise Exception() # TODO: replace with specific exception
