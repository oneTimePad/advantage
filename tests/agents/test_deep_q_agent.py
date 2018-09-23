import unittest
import os
import gym
import tensorflow as tf
from builders.agents_builder import build
from utils.proto_parser import parse_obj_from_file
from protos.agents import agents_pb2
from agents.approximate_agents import DeepQAgent
from utils.sarsa import Sarsa

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"



class TestDeepQAgent(unittest.TestCase):
    """ Tests the various functionalities of the DeepQAgent """
    def setUp(self):
        DQN_CONFIG = os.path.join(__location__,  "../mock_configs/deep_q_agent.config")
        graph = tf.Graph()
        environment = gym.make("CartPole-v0")

        agents_config = parse_obj_from_file(DQN_CONFIG, agents_pb2.Agents)

        self.dqn_agent = build(agents_config, graph, environment)

    def test_set_up(self):
        self.dqn_agent.set_up()

    def test_act_for_steps_training(self):
        self.dqn_agent.set_up()
        steps, sarsa = list(self.dqn_agent.act_for_steps(1, training=False))[0]
        print(steps, sarsa)

    def test_improve_target(self):
        self.dqn_agent.set_up()

        sarsa_buffer = []

        for step, sarsa in self.dqn_agent.act_for_steps(5, training=True):
            sarsa_buffer.append(sarsa)


        self.dqn_agent.improve_target(Sarsa.split_list_to_np(sarsa_buffer))

    def test_improve_policy(self):
        self.dqn_agent.set_up()

        self.dqn_agent.improve_policy(None)




unittest.main()
