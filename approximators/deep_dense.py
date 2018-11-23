import tensorflow as tf
from advantage.approximators.base.base_approximators import DeepApproximator
from advantage.approximators.base.utils import parse_activation, parse_initializer
from advantage.checkpoint import checkpointable

class DeepDense(DeepApproximator):
    """ Fully-connected Network"""

    def set_up(self, config, tensor_inputs, inputs_placeholders):
        blocks = self.config.block
        prev = tensor_inputs
        for block in blocks:
            activation = parse_activation(block.activation)
            initializer = parse_initializer(block.initializer)
            prev = tf.layers.dense(prev,
                                   block.num_units,
                                   activation=activation,
                                   kernel_initializer=initializer)

        return prev
