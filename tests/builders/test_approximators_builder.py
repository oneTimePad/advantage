import unittest
import os
from builders.approximators_builder import build
from utils.proto_parser import parse_approximators_from_file
from approximators.ff_approximators import DeepConvolutional, DeepDense
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

class TestApproximatorsBuilder(unittest.TestCase):
    """ Tests for basic construction of deep models. """ #TODO add more as configuration becomes more complex
    def setUp(self):
        self.DEEP_CONV_CONFIG = os.path.join(__location__,  "mock_deep_convolutional.config")
        self.DEEP_DENSE_CONFIG = os.path.join(__location__, "mock_deep_dense.config")

    def test_build_DeepConvolutional(self):

        approximators_config = parse_approximators_from_file(self.DEEP_CONV_CONFIG)

        deepConv = build(approximators_config)

        if not isinstance(deepConv, DeepConvolutional):
            self.assertEqual(1, 0)




    def test_build_DeepDense(self):

        approximators_config = parse_approximators_from_file(self.DEEP_DENSE_CONFIG)

        deepDense = build(approximators_config)

        if not isinstance(deepDense, DeepDense):
            self.assertEqual(1, 0)

unittest.main()
