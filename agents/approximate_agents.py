import numpy as np
from functools import partial
import tensorflow as tf
from utils.policy_tools import epsilon_greedy
from .base_agents import LearningAgent

class ApproximateAgent(LearningAgent):
    """Approximate Learning Agent"""
    def __init__(self, policy, environment, discount_factor, graph, session, **kwargs):
        self._graph = graph
        self._session = session

        super().__init__(policy=policy,
                        environment=environment,
                        discount_factor=discount_factor,
                        **kwargs)

    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        return self._session
