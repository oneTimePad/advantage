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
                 agent):

        self._agent = agent
        self._environment = environment
        self._model_scope = model_scope

        self._graph = graph
        self._sessions = []

        self._restore_session = None

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
    def restore_session(self):
        """ property for `restore_session`
        """
        if not isinstance(self._restore_session, tf.Session):
            raise AttributeError("`restore_session must be set by model instance")
        return self._restore_session

    @restore_session.setter
    def restore_session(self, session):
        """ Setter for we can make a property that
        validates. Thus keeping the real value
        protected
        """
        self._restore_session = session

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
                    info to be passed to train_iteration (if needed)
         """
        raise NotImplementedError()

    @abstractmethod
    def improve_iteration(self, info_dict):
        """ Single improve/training iteration of the agent(s)

            returns: True/False whether improve_steps should be incremented
            (did improvement actually happen)
        """
        raise NotImplementedError()
