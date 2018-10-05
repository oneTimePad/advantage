import unittest
import os
import gym
import tensorflow as tf
import numpy as np
from builders.agents_builder import build
from utils.proto_parser import parse_obj_from_file
from protos.agents import agents_pb2
from agents.deep_q_agent import DeepQAgent
from elements.sarsa import Sarsa
from environments.gym_environment import GymEnvironment

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"



class TestDeepQAgent(unittest.TestCase):
    """ Tests the various functionalities of the DeepQAgent """
    def setUp(self):
        DQN_CONFIG = os.path.join(__location__,  "../mock_configs/deep_q_agent.config")
        graph = tf.Graph()
        environment = GymEnvironment("CartPole-v0")

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

        sarsa_list = []

        for step, env_dict in self.dqn_agent.act_for_steps(5, training=True):
            state = env_dict["state"].astype(np.float32) if isinstance(env_dict["state"], np.ndarray) else np.array([env_dict["state"]], dtype=np.float32)
            action = env_dict["action"].astype(np.floa32) if isinstance(env_dict["action"], np.ndarray) else np.array([env_dict["action"]], dtype=np.float32)
            reward = env_dict["reward"].astype(np.float32) if isinstance(env_dict["reward"], np.ndarray) else np.array([env_dict["reward"]], dtype=np.float32)
            done = np.array([env_dict["done"]], dtype=np.bool)
            next_state = env_dict["next_state"].astype(np.float32) if isinstance(env_dict["next_state"], np.ndarray) else np.array([env_dict["next_state"]], dtype=np.float32)

            sarsa = Sarsa.make_element(state=state,
                                        action=action,
                                        reward=reward,
                                        done=done,
                                        next_state=next_state)
            sarsa_list.append(sarsa)


        self.dqn_agent.improve_target(Sarsa.reduce(sarsa_list))

    def test_improve_policy(self):
        self.dqn_agent.set_up()

        self.dqn_agent.improve_policy()




unittest.main()
