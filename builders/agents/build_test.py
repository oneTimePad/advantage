import unittest
import os
import gym
import numpy as np
import tensorflow as tf
from advantage.builders import build_agent
from advantage.utils.proto_parsers import parse_obj_from_file
from advantage.protos.agents.base import agents_pb2
from advantage.agents.deep_q_agent import DeepQAgent

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

class TestAgentsBuilder(unittest.TestCase):
    """ Test various builds for Agents """ # This feels useless
    def setUp(self):
        pass

    def test_build(self):
        pass

    def test_build_DeepQAgent(self):
        DQN_CONFIG = os.path.join(__location__,  "../mock_configs/deep_q_agent.config")
        graph = tf.Graph()
        environment = gym.make("CartPole-v0")

        agents_config = parse_obj_from_file(DQN_CONFIG, agents_pb2.Agents)

        dqn_agent = build(agents_config, graph, environment)

        if not isinstance(dqn_agent, DeepQAgent):
            self.assertEqual(1, 0)

        self.assertEqual(1, 1)

unittest.main()
