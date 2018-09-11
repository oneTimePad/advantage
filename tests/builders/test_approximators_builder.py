import unittest
import os
from builders.approximators_builder import build
from utils.proto_parser import parse_obj_from_file
from approximators.ff_approximators import DeepConvolutional, DeepDense
from protos.approximators import approximators_pb2
import tensorflow as tf
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
class TestApproximatorsBuilder(unittest.TestCase):
    """ Tests for basic construction of deep models. """ #TODO add more as configuration becomes more complex
    def setUp(self):
        self.DEEP_CONV_CONFIG = os.path.join(__location__,  "../mock_configs/mock_deep_convolutional.config")
        self.DEEP_DENSE_CONFIG = os.path.join(__location__, "../mock_configs/mock_deep_dense.config")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_dense = tf.placeholder(shape=[None, 4], name="test_input_dense", dtype=tf.float32)
            self.inputs_conv = tf.placeholder(shape=[None, 32, 32, 1], name="test_input_conv", dtype=tf.float32)

    def test_build_DeepConvolutional(self):

        approximators_config = parse_obj_from_file(self.DEEP_CONV_CONFIG, approximators_pb2.Approximators)

        deepConv = build(self.graph, approximators_config, self.inputs_conv, [self.inputs_conv])

        if not isinstance(deepConv, DeepConvolutional):
            self.assertEqual(1, 0)

        if not isinstance(deepConv.network, tf.Tensor):
            self.assertEqual(1, 0)

        self.assertEqual(1, 1)


    def test_build_DeepDense(self):

        approximators_config = parse_obj_from_file(self.DEEP_DENSE_CONFIG, approximators_pb2.Approximators)

        deepDense = build(self.graph, approximators_config, self.inputs_dense, [self.inputs_dense])

        if not isinstance(deepDense, DeepDense):
            self.assertEqual(1, 0)

        if not isinstance(deepDense.network, tf.Tensor):
            self.assertEqual(1, 0)

        self.assertEqual(1, 1)

unittest.main()
