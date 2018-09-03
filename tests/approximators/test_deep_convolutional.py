import unittest
from approximators.ff_approximators import DeepConvolutional
import tensorflow as tf
import numpy as np


class MockDeepConvolutionalBlock:
    def __init__(self, num_filters,
                       kernelH, kernelW,
                       activation,
                       stride):
            self.num_filters = num_filters
            self.kernelH = kernelH
            self.kernelW = kernelW
            self.stride = stride
            self.activation = activation
            self.initializer = 1
            self.padding = 1


class MockOutput:
    def __init__(self, output):
        self.output = output
        self.num_actions = 2

    def WhichOneof(self, string):
        return "num_actions"

class MockDeepConvolutionalConfig:
    def __init__(self, output, blocks):
        self.block = blocks
        self.outputConfig = output

class TestDeepConvolutional(unittest.TestCase):
    """Tests for DeepConvolutional approximator"""

    def setUp(self):
        self.deepConvConfig = MockDeepConvolutionalConfig(
                                    MockOutput(0),
                                    [MockDeepConvolutionalBlock(2, 3, 3, 0, 1)])
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_conv = tf.placeholder(shape=[None, 3, 3, 1], name="test_input_conv", dtype=tf.float32)

        self.test_inputs = np.array([ [ [[1.0, 1.0, 1.0],
                                        [2.0, 2.0, 2.0],
                                        [3.0, 3.0, 3.0] ]],
                                         [[[2.0, 2.0, 2.0],
                                            [2.0, 2.0, 2.0],
                                            [1.0, 1.0, 1.0]]]], dtype=np.float32)
        self.test_inputs = self.test_inputs.transpose([0, 2, 3, 1])

        self.session = tf.Session(graph = self.graph)
        self.network = DeepConvolutional(self.graph, self.deepConvConfig, "test")
        self.network.set_up(self.inputs_conv)

        with self.graph.as_default():
            self.init = tf.global_variables_initializer()

    def test_inference(self):
        with self.session.as_default():
            self.session.run(self.init)
            output = self.network.inference(self.session, self.test_inputs)

        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[0], 2)

unittest.main()
