import unittest
from utils.buffers import Sarsa
from utils.values import compute_q_part_advantage
import numpy as np
import tensorflow as tf

class TestValues(unittest.TestCase):
    """ Tests for special value functions """
    def setUp(self):
        self.sarsa = [Sarsa(np.array([1]), np.array([2]), 1, 0, np.array([3]), None),
                        Sarsa(np.array([1]), np.array([2]),  5, 1, np.array([3]), None)]
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.mock_network_data = np.array([[1.0, 3.0],
                                            [4.0, 2.0]])
        with self.graph.as_default():
            self.init = tf.global_variables_initializer()
            self.network = tf.constant(self.mock_network_data)
            self.placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    def test_compute_q_part_advantage(self): # TODO might have to be refined when tested on real network
        with self.graph.as_default():
            with self.session.as_default():
                self.session.run(self.init)

        values = compute_q_part_advantage(self.session, self.network,
                                            self.placeholder, self.sarsa, 0.01)

        np.testing.assert_array_equal(values, np.array([1 + 0.01 * 3,  5.]))

unittest.main()
