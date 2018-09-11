import unittest
import os
from approximators.ff_approximators import DeepConvolutional
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
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

class Field:
    def __init__(self):
        self.name = "num_actions"

class ValueOutput:
    def ListFields(self):
        yield (Field(), 2)

class Model:
    def __init__(self, block):
        self.block = block

class MockDeepConvolutionalConfig:
    def __init__(self, blocks):
        self.model = Model(blocks)
        self.value = ValueOutput()
        self.optimizer = 0
        self.learning_rate = .001
        self.name_scope = "test"
        self.reuse = False

    def WhichOneof(self, string):
        if string == "output":
            return "value"
        else:
            return "model"

class TestDeepConvolutional(unittest.TestCase):
    """Tests for DeepConvolutional approximator"""

    def setUp(self):
        self.deepConvConfig = MockDeepConvolutionalConfig(
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
        self.network = DeepConvolutional(self.graph, self.deepConvConfig)
        self.network.set_up(self.inputs_conv, [self.inputs_conv])

        with self.graph.as_default():
            self.init = tf.global_variables_initializer()

    def test_inference(self):
        with self.session.as_default():
            self.session.run(self.init)
            output = self.network.inference(self.session, {"test_input_conv": self.test_inputs})

        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[0], 2)

unittest.main()
