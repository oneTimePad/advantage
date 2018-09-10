import unittest
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


class Field:
    def __init__(self):
        self.name = "num_actions"

class ValueOutput:

    def ListFields(self):
        yield (Field(), 2)

class Model:
    def __init__(self, block):
        self.block = block

class MockDeepDenseConfig:
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

class TestDeepDense(unittest.TestCase):
    """Tests for DeepDense approximator"""

    def setUp(self):
        self.deepDenseConfig = MockDeepDenseConfig(
                                    [MockDeepDenseBlock(0, 4),
                                    MockDeepDenseBlock(0, 4)])
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_dense = tf.placeholder(shape=[None, 4], name="test_input_dense", dtype=tf.float32)

        self.test_inputs = np.array([[1.0, 2.0, 3.0, 4.0],
                                     [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        self.session = tf.Session(graph = self.graph)
        self.network = DeepDense(self.graph, self.deepDenseConfig)
        self.network.set_up(self.inputs_dense)

        with self.graph.as_default():
            self.init = tf.global_variables_initializer()

    def test_initialize(self):
        # TODO
        pass

    def test_inference(self):
        with self.graph.as_default():
            with self.session.as_default():
                self.session.run(self.init)
                output = self.network.inference(self.session, self.test_inputs)

            self.assertEqual(output.shape[0], 2)
            self.assertEqual(output.shape[1], 2)

    def test_copy(self):
        with self.graph.as_default():
            with self.session.as_default():
                params = self.network.trainable_parameters
                params_dict = {}
                new_params_dict = {}
                for param in params:
                    shape = [int(s) for s in param.get_shape()]
                    new_param = np.ones(shape=shape, dtype=np.float32)
                    new_params_dict[param.name] = new_param
                    params_dict[param.name] = new_param

                self.network.copy(self.session, params_dict)

                for p in params:
                    np.testing.assert_array_equal(self.session.run(p), new_params_dict[p.name])

unittest.main()
