import tensorflow as tf
from advantage.approximators.base.base_approximators import DeepApproximator
from advantage.approximators.base.utils import parse_activation, parse_initializer, parse_padding
from advantage.checkpoint import checkpointable

class DeepConvolutional(DeepApproximator):
    """ Convolutional Network"""

    def set_up(self, tensor_inputs, inputs_placeholders):
        blocks = self.config.block
        prev = tensor_inputs
        for block in blocks:
            activation = parse_activation(block.activation)
            initializer = parse_initializer(block.initializer)
            prev = tf.layers.conv2d(prev,
                                    filters=block.num_filters,
                                    strides=block.stride,
                                    kernel_size=[block.kernelH, block.kernelW],
                                    activation=activation,
                                    padding=parse_padding(block.padding),
                                    kernel_initializer=initializer)
        return prev
