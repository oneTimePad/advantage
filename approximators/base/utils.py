import tensorflow as tf
from advantage.protos.approximators.base import utils_pb2
from advantage.utils.proto_parsers import parse_enum_to_str
""" Various utilities specifying options for setting network
properties
"""

_OPTIMIZERS = {
    "AdamOptimizer": tf.train.AdamOptimizer,
    "GradientDescentOptimizer": tf.train.GradientDescentOptimizer
}

def parse_optimizer(optimizer):
    """ Converts optimizer from proto
    into actual tf optimizer function

        Args:
            optimizer: enum value from parsed protobuf

        Returns:
            tf optimizer function
    """
    return _OPTIMIZERS[parse_enum_to_str(utils_pb2,
                                     "optimizer",
                                         optimizer)]

_ACTIVATIONS = {
    "NONE": tf.identity,
    "RELU" : tf.nn.relu6,
    "SIGMOID": tf.nn.sigmoid,
    "ELU": tf.nn.elu
}

def parse_activation(activation):
    """ Converts activation from proto
    into actual tf activation function

        Args:
            activation: enum value from parsed protobuf

        Returns:
            tf activation function
    """
    return _ACTIVATIONS[parse_enum_to_str(utils_pb2,
                                          "activation",
                                          activation)]

_INITIALIZERS = {
    "ones_initializer": tf.ones_initializer(),
    "variance_scaling_initializer": tf.variance_scaling_initializer()
}

def parse_initializer(initializer):
    """ Converts initializer from proto
    into actual tf initializer function

        Args:
            initializer: enum value from parsed protobuf

        Returns:
            tf initializer function
    """
    return _INITIALIZERS[parse_enum_to_str(utils_pb2,
                                           "initializer",
                                           initializer)]


def parse_padding(padding):
    """ Converts padding from proto
    into actual tf padding function

        Args:
            initializer: enum value from parsed protobuf

        Returns:
            padding str
    """
    return parse_enum_to_str(utils_pb2,
                             "padding",
                             padding)
