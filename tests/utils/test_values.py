import unittest
import os
from elements.sarsa import Sarsa
from utils.values import apply_bellman_operator
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


class MockNetwork:

    def inference(self, *args, **kargs):
        return np.array([[1.0, 3.0],
                        [4.0, 2.0]])

    @property
    def feed_dict_keys(self):
        return ["test"]

class TestValues(unittest.TestCase):
    """ Tests for special value functions """
    def setUp(self):

        self.sarsa = Sarsa.reduce([
                                Sarsa.make_element(state=np.array([1.],dtype=np.float32),
                                                   action=np.array([2.], dtype=np.float32),
                                                   reward=np.array([1.], dtype=np.float32),
                                                   done=np.array([False], dtype=np.bool),
                                                   next_state=np.array([3.], dtype=np.float32)),
                                Sarsa.make_element(state=np.array([1.],dtype=np.float32),
                                                  action=np.array([2.], dtype=np.float32),
                                                  reward=np.array([5.], dtype=np.float32),
                                                  done=np.array([True], dtype=np.bool),
                                                  next_state=np.array([3.], dtype=np.float32))])
        self.network = MockNetwork()
        """
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.init = tf.global_variables_initializer()
            self.network = tf.constant(self.mock_network_data)
            self.placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        """
    def test_apply_bellman_operator(self): # TODO might have to be refined when tested on real network
        """
        with self.graph.as_default():
            with self.session.as_default():
                self.session.run(self.init)
        """

        _, _, values = apply_bellman_operator(tf.Session(), self.network, self.sarsa, 0.01, "test")

        np.testing.assert_array_equal(values, np.array([[1 + 0.01 * 3],  [5.]]))

unittest.main()
