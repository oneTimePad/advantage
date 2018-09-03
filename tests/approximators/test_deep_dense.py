import unittest
import os
from approximators.ff_approximators import DeepDense
import tensorflow as tf
import numpy as np

class MockDeepDenseBlock:
    def __init__(self,
                 activation,
                 num_units):
        self.activation = activation
        self.num_units = num_units
        self.initializer = 1


class MockOutput:
    def __init__(self, output):
        self.output = output
        self.num_actions = 2

    def WhichOneof(self, string):
        return "num_actions"

class MockDeepDenseConfig:
    def __init__(self, output, blocks):
        self.block = blocks
        self.outputConfig = output

class TestDeepDense(unittest.TestCase):
    """Tests for DeepDense approximator"""

    def setUp(self):
        self.deepDenseConfig = MockDeepDenseConfig(
                                    MockOutput(0),
                                    [MockDeepDenseBlock(0, 4),
                                    MockDeepDenseBlock(0, 4)])
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_dense = tf.placeholder(shape=[None, 4], name="test_input_dense", dtype=tf.float32)

        self.test_inputs = np.array([[1.0, 2.0, 3.0, 4.0],
                                     [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        self.session = tf.Session(graph = self.graph)
        self.network = DeepDense(self.graph, self.deepDenseConfig, "test")
        self.network.set_up(self.inputs_dense)

        with self.graph.as_default():
            self.init = tf.global_variables_initializer()

    def test_inference(self):
        with self.session.as_default():
            self.session.run(self.init)
            output = self.network.inference(self.session, self.test_inputs)

        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 2)

unittest.main()
