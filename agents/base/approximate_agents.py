import numpy as np
from functools import partial
from abc import ABCMeta
import tensorflow as tf
from advantage.utils.policy_tools import epsilon_greedy
from advantage.agents.base.base_agents import LearningAgent

class ApproximateAgent(LearningAgent, metaclass=ABCMeta):
    """Approximate Learning Agent"""
    def __init__(self,
                 policy,
                 environment,
                 discount_factor,
                 graph,
                 agent_scope,
                 **kwargs):
        self._graph = graph
        self._session = None
        self._agent_scope = agent_scope
        super().__init__(policy=policy,
                         environment=environment,
                         discount_factor=discount_factor,
                         **kwargs)

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
