import gin
from abc import ABCMeta, abstractmethod
from utils.decorator_utils import parameterized

""" This module represents the main set of agent
class or the `base`. All other agents subclasses
of these
"""

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
    def construct(self, **objectives):
        """ Agent construction hook.
        The agents selected `objectives`
        are passed in and the agent can complete
        any further setup that the objectives
        need
        """
        raise NotImplementedError()

    @abstractmethod
    def pre_iteration(self):
        raise NotImplementedError()

    @abstractmethod
    def post_iteration(self):
        raise NotImplementedError()

    @abstractmethod
    def sample_action(self, state):
        """ The agent acts in the
        environment given the `state`-space
        representation
            Args:
                state: state-space representation
                    from selected `Environment`

            Returns:
                selected action-space element
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
        action = self.sample_action(prev_state)
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

    @loggers.avg(loggers.LogVarType.INSTANCE_ATTR,
                 "traj_reward",
                 "Agent average trajectory reward is %.2f",
                 (loggers.LogVarType.RETURNED_DICT, "done", lambda done: done),
                 tensorboard=False)
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



class ValueGradientAgent(LearningAgent):
    """ This agent minimizes the `Bellman` error.
    This is similar to the variants of TD-Learning/Monte-Carlo
    """
    pass

class DecoupledValueGradientAgent(ValueGradientAgent):
    """ This agent minimizes a decoupled TD-Learning
    error. In which there is a `bootstrapping` and
    `target` value-function. When the function is
    an action-value function, this is a Deep Q-Network
    agent.
    """
    pass

class PolicyGradientAgent(LearningAgent):
    """ This agent maxmizes the `policy` gradient.
    This agent can stand on it's own if utilizes
    the full return
    """
    pass

class MultiGradientAgent(LearningAgent):
    """ Represents a more general implementation
    of `Actor-Critic` style learning. In which,
    which multiple `Gradient` Agents can be
    composed
    """
    pass
