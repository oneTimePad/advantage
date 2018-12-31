from abc import ABCMeta
from abc import abstractmethod

class Environment(metaclass=ABCMeta):
    """ Represents the environment for an RL Agent. The Environment
    maintains the MDP's state and acts as a simulator for the agent to
    learn in. User can define their own environments/simulators. The simulators
    will keep track of their state in their own chosen way.
    """
    @property
    @abstractmethod
    def dims(self):
        """ Dict containg the column dimensions of all environment
            properties

            Returns:
                dict : {'state_col_dim':, 'action_col_dim', ...}
                    key template -> '*_col_dim' [see GymEnvironment]
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_space(self):
        """ Returns the ActionSpace object of the environment """
        raise NotImplementedError()

    @property
    def tf_state_shape(self):
        """Fetch env shape as list
            Args:
                environment: Environment or GymEnv

            Returns:
                shape as list for placeholder
        """
        state_shape = self.reset().shape

        return [None] + list(state_shape)

    @abstractmethod
    def step(self, action):
        """The Agent performs an action in the environment, modifying the MDP
        state and resulting in a transition.

            Args:
                action : the action taken by the agent

            Returns:
                next_state :the state transitioned to
                reward : the reward observed
                done : whether the agent's action resulted in the episode finishing
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Resets the MDP internal state

            Returns:
                state: new initialized MDP state
        """
        raise NotImplementedError()

class ActionSpace(metaclass=ABCMeta):
    """ Defines the action space of the Environment
    """
    @abstractmethod
    def sample(self):
        """ Returns a sampled action from the space """
        raise NotImplementedError()
