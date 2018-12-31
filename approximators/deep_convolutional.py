import tensorflow as tf
import attr
from advantage.approximators.base.base_approximators import deep_approximator

""" Convolutional Network
"""

@attr.s(frozen=True)
class ConvBlock:
    """ Represents convolutional block
    for gin config file
    """
    num_filters = attr.ib(kw_only=True)
    stride = attr.ib(kw_only=True)
    kernel = attr.ib(kw_only=True)
    padding = attr.ib(kw_only=True)
    initializer = attr.ib(kw_only=True)
    activation = attr.ib(kw_only=True)

@deep_approximator
class DeepConvolutional:
    """ Convolutional Network"""

    def set_up(self, architecture, tensor_inputs, inputs_placeholders):
        """ TensorFlow construction of the approximator network
                Args:
                    architecture: architecture list
                    tensor_inputs: actual input to the network
                    inputs_placeholders: list, the required placeholders to fill before running.
                        These are the placholders that the tensor_inputs depend on.

                Returns:
                    the last block in the network
        """
        prev = tensor_inputs
        for block in architecture:
            prev = tf.layers.conv2d(prev,
                                    filters=block.num_filters,
                                    strides=block.stride,
                                    kernel_size=block.kernel,
                                    activation=block.activation,
                                    padding=block.padding,
                                    kernel_initializer=block.initializer)
        return prev
