import tensorflow as tf
import attr
from advantage.approximators.base.base_approximators import deep_approximator

""" Dense (fully-connected) Network
"""

@attr.s(frozen=True)
class DenseBlock:
    """ Represents dense block
    for gin config file
    """
    num_units = attr.ib(kw_only=True)
    initializer = attr.ib(kw_only=True)
    activation = attr.ib(kw_only=True)

@deep_approximator
class DeepDense:
    """ Fully-connected Network"""

    name_scope = "deep_convolutional"

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
            prev = tf.layers.dense(prev,
                                   block.num_units,
                                   activation=block.activation,
                                   kernel_initializer=block.initializer)

        return prev
