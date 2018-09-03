from .base_approximators import DeepApproximator, ACTIVATIONS, OUTPUTS
from protos.approximators import helpers_pb2
import tensorflow as tf
""" Feed-forward Deep NN approximators """

class DeepConvolutional(DeepApproximator):
    """ Convolutional Network"""

    def __init__(self, graph, config, name_scope, reuse=False):
        self._scope = "DeepConvolutional"
        super(DeepConvolutional, self).__init__(graph, config, name_scope, reuse=reuse)

    @staticmethod
    def enum_padding_to_str(enum_value):
        return helpers_pb2._PADDING.values_by_number[enum_value].name

    def set_up(self, tensor_inputs):
        blocks = self._config.block
        prev = tensor_inputs
        with self._graph.as_default():
            with tf.variable_scope(self._scope + self._name_scope, reuse=self._reuse):
                for block in blocks:
                    activation_name = self.enum_activation_to_str(block.activation)
                    prev = tf.layers.conv2d(prev, filters=block.num_filters,
                                                    strides=block.stride,
                                                    kernel_size=[block.kernelH, block.kernelW],
                                                    activation=ACTIVATIONS[activation_name],
                                                    padding=self.enum_padding_to_str(block.padding))

                output_config = self._config.outputConfig
                output_name = self.enum_output_to_str(output_config.output)
                output = OUTPUTS[output_name](prev, getattr(output_config, output_config.WhichOneof("number")))

        self._network = output

    def inference(self, runtime_tensor_inputs):
        pass

    def gradients(self):
        pass

    def update(self, runtime_inputs):
        pass

class DeepDense(DeepApproximator):
    """ Fully Connected Network"""

    def __init__(self, graph, config, name_scope, reuse=False):
        self._scope = "DeepDense"
        super(DeepDense, self).__init__(graph, config, name_scope, reuse)

    def set_up(self, tensor_inputs):
        blocks = self._config.block
        prev = tensor_inputs
        with self._graph.as_default():
            with tf.variable_scope(self._scope + self._name_scope, reuse=self._reuse):
                for block in blocks:
                    activation_name = self.enum_activation_to_str(block.activation)
                    prev = tf.layers.dense(prev, block.num_units, activation=ACTIVATIONS[activation_name])

                output_config = self._config.outputConfig
                output_name = self.enum_output_to_str(output_config.output)
                output = OUTPUTS[output_name](prev, getattr(output_config, output_config.WhichOneof("number")))

        self._network = output


    def inference(self, runtime_tensor_inputs):
        pass

    def gradients(self):
        pass

    def update(self, runtime_inputs):
        pass
