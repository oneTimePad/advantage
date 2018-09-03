from .base_approximators import DeepApproximator, ACTIVATIONS, OUTPUTS, INITIALIZERS
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
            with tf.variable_scope(self._scope + self._name_scope, reuse=self._reuse) as scope:
                for block in blocks:
                    activation_name = self.enum_activation_to_str(block.activation)
                    initializer_name = self.enum_initializer_to_str(block.initializer)
                    prev = tf.layers.conv2d(prev, filters=block.num_filters,
                                                    strides=block.stride,
                                                    kernel_size=[block.kernelH, block.kernelW],
                                                    activation=ACTIVATIONS[activation_name],
                                                    padding=self.enum_padding_to_str(block.padding),
                                                    kernel_initializer=INITIALIZERS[initializer_name])

                output_config = self._config.outputConfig
                output_name = self.enum_output_to_str(output_config.output)
                output = OUTPUTS[output_name](prev, getattr(output_config, output_config.WhichOneof("number")))

        self._inputs_placeholder = tensor_inputs
        self._var_scope_obj = scope
        self._network = output

    def inference(self, session, runtime_tensor_inputs):
        if not isinstance(session, tf.Session):
            raise ValueError("Must pass in tf.Session")
        with session.as_default():
            return session.run(self._network, feed_dict={self._inputs_placeholder: runtime_tensor_inputs}) #TODO as of now this is duplicate code, might change later

    def gradients(self):
        pass

    def update(self, runtime_inputs):
        pass

class DeepDense(DeepApproximator):
    """ Fully Connected Network"""

    def __init__(self, graph, config, name_scope, reuse=False):
        self._scope = "DeepDense"
        self._var_scope_obj = None
        super(DeepDense, self).__init__(graph, config, name_scope, reuse)

    def set_up(self, tensor_inputs):
        blocks = self._config.block
        prev = tensor_inputs
        with self._graph.as_default():
            with tf.variable_scope(self._scope + self._name_scope, reuse=self._reuse) as scope:
                for block in blocks:
                    activation_name = self.enum_activation_to_str(block.activation)
                    initializer_name = self.enum_initializer_to_str(block.initializer)
                    prev = tf.layers.dense(prev, block.num_units, activation=ACTIVATIONS[activation_name],
                                                                    kernel_initializer=INITIALIZERS[initializer_name])
                output_config = self._config.outputConfig
                output_name = self.enum_output_to_str(output_config.output)
                output = OUTPUTS[output_name](prev, getattr(output_config, output_config.WhichOneof("number")))

        self._inputs_placeholder = tensor_inputs
        self._var_scope_obj = scope
        self._network = output


    def inference(self, session, runtime_tensor_inputs):
        if not isinstance(session, tf.Session):
            raise ValueError("Must pass in tf.Session")
        with session.as_default():
            return session.run(self._network, feed_dict={self._inputs_placeholder: runtime_tensor_inputs})

    def gradients(self):
        pass

    def update(self, runtime_inputs):
        pass
