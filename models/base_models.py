from abc import ABCMeta
from abc import abstractmethod

class LearningModel(object, metaclass=ABCMeta):
    """ Represents the Reinforcement Learning (model-free) Reinforcement
    Learning Model. A model uses an environment to construct MDPs
    for agents. It controls the agents and chooses when to have them
    improve their policies. The model also allows for inter-agent cooporation (i.e. Multi-Agent and Async methods)
    or embedded training (i.e. Meta-RL).
    """

    def __init__(self, graph,
                    environment,
                    agent,
                    steps_to_run_for,
                    improve_policy_modulo,
                    steps_for_act_iter,
                     **kwargs):
        self._agent = agent
        self._environment = environment
        self._steps_to_run_for = steps_to_run_for
        self._improve_policy_modulo = improve_policy_modulo
        self._steps_for_act_iter = steps_for_act_iter

    @abstractmethod
    def set_up(self):
        """ Builds model and setup all utilities """
        raise NotImplementedError()

    @abstractmethod
    def act_iteration(self):
        """ Single acting iteration of the agent(s) with setup
                Returns info to be passed to train_iteration
         """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration(self, info_dict):
        """ Single training iteration of the agent(s) """
        raise NotImplementedError()

    def run_training(self):
        """ Runs the training process for the model """
        pass
