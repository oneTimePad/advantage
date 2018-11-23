from abc import ABCMeta
from abc import abstractmethod
import tensorflow as tf

class LearningModel(metaclass=ABCMeta):
    """ Represents the Reinforcement Learning (model-free) Reinforcement
    Learning Model. A model uses an environment to construct MDPs
    for agents. It controls the agents and chooses when to have them
    improve their policies. The model also allows for inter-agent cooporation
    (i.e. Multi-Agent and Async methods) or embedded training (i.e. Meta-RL).
    """

    def __init__(self,
                 graph,
                 environment,
                 model_scope,
                 agent,
                 steps_to_run_for,
                 improve_policy_modulo,
                 steps_for_act_iter,
                 **kwargs):
        self._agent = agent
        self._environment = environment
        self._model_scope = model_scope
        self._steps_to_run_for = steps_to_run_for
        self._improve_policy_modulo = improve_policy_modulo
        self._steps_for_act_iter = steps_for_act_iter
        self._graph = graph
        self._sessions = []

    @property
    def graph(self):
        """ property for `_graph`
        """
        return self._graph

    @property
    def environment(self):
        """ property for `_environment`
        """
        return self._environment

    @property
    def model_scope(self):
        """ property for `_model_scope`
        """
        return self._model_scope

    @property
    def name_scope(self):
        """ property for name_scope from `_model_scope`
        """
        return self._model_scope.name_scope

    @property
    @abstractmethod
    def steps(self):
        """ Returns number of steps done in total
            steps is used by the runner to determine when to stop
            It is up to the model to determine it's interpretation
        """
        raise NotImplementedError()

    def add_session(self, agent):
        """ Adds session to agent
        """
        agent.session = tf.Session(graph=self._graph)
        self._sessions.append(agent.session)

    def clean(self):
        """ Cleans up TensorFlow Graph
        and Sessions.
        """
        for sess in self._sessions:
            sess.close()

    @abstractmethod
    def set_up_train(self):
        """ Builds model and setup all utilities for training
        """
        raise NotImplementedError()

    @abstractmethod
    def set_up(self):
        """ Builds model for normal usage
        """
        raise NotImplementedError()

    def run_trajectory(self):
        """ Utilized for inference
        """
        assert self._agent is not None
        return self._agent.run_trajectory()

    def run_trajectory_through(self):
        """ Utilized for inference
        """
        assert self._agent is not None
        return self._agent.run_trajectory_through()

    @abstractmethod
    def act_iteration(self):
        """ Single acting iteration of the agent(s) with setup
                Returns:
                    info to be passed to train_iteration
         """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration(self, info_dict):
        """ Single training iteration of the agent(s)
        """
        raise NotImplementedError()
