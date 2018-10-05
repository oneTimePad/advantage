import unittest
import os
from builders.models_builder import build
from protos.models import models_pb2
from utils.proto_parser import parse_obj_from_file
from environments.gym_environment import GymEnvironment

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


class TestDeepQModel(unittest.TestCase):
    """ Tests the various functionalities of the DeepQModel """
    def setUp(self):
        self.env = GymEnvironment("CartPole-v0")
        DQNM_CONFIG = os.path.join(__location__,  "../mock_configs/deep_q_model.config")

        models_config = parse_obj_from_file(DQNM_CONFIG, models_pb2.Models)

        self.dqn_model = build(models_config, self.env)

    def test_set_up(self):
        self.dqn_model.set_up()


    def test_act_iteration(self):
        self.test_set_up()

        self.dqn_model.act_iteration()

        self.assertEqual(self.dqn_model.replay_buffer.len, 12) # test config says 12 steps

    def test_train_iteration_improve_target(self):

        self.test_set_up()

        steps = self.dqn_model.act_iteration()

        self.dqn_model.train_iteration(steps)

    def test_train_iteration_improve_policy(self):
        self.test_set_up()

        steps = self.dqn_model.act_iteration()

        self.dqn_model.train_iteration(steps)

        steps = self.dqn_model.act_iteration()

        self.dqn_model.train_iteration(steps)




unittest.main()
