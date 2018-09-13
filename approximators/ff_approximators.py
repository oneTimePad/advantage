from .base_approximators import DeepApproximator, ACTIVATIONS, INITIALIZERS
from protos.approximators import helpers_pb2
import tensorflow as tf
""" Feed-forward Deep NN approximators """

class DeepConvolutional(DeepApproximator):
    """ Convolutional Network"""

    def __init__(self, graph, config):
        self._scope = "DeepConvolutional"
        self._model = self.parse_specific_model_config(config)
        super(DeepConvolutional, self).__init__(graph, config)

    @staticmethod
    def enum_padding_to_str(enum_value):
        return helpers_pb2._PADDING.values_by_number[enum_value].name

    def set_up(self, tensor_inputs, inputs_placeholders, **kwargs):
        blocks = self._model.block
        prev = tensor_inputs
        with self._graph.as_default():
            with tf.variable_scope(self._scope + "_" + self._name_scope, reuse=self._reuse) as scope:
                for block in blocks:
                    activation_name = self.enum_activation_to_str(block.activation)
                    initializer_name = self.enum_initializer_to_str(block.initializer)
                    prev = tf.layers.conv2d(prev, filters=block.num_filters,
                                                    strides=block.stride,
                                                    kernel_size=[block.kernelH, block.kernelW],
                                                    activation=ACTIVATIONS[activation_name],
                                                    padding=self.enum_padding_to_str(block.padding),
                                                    kernel_initializer=INITIALIZERS[initializer_name])



        super(DeepConvolutional, self).set_up(tensor_inputs, inputs_placeholders, last_block=prev, var_scope_obj=scope)

    def inference(self, session, runtime_tensor_inputs):
        if not isinstance(session, tf.Session):
            raise ValueError("Must pass in tf.Session")
        feed_dict = super()._produce_feed_dict(runtime_tensor_inputs)
        with session.as_default():
            return session.run(self._network, feed_dict=feed_dict) #TODO as of now this is duplicate code, might change later


class DeepDense(DeepApproximator):
    """ Fully Connected Network"""

    def __init__(self, graph, config):
        self._scope = "DeepDense"
        self._model = self.parse_specific_model_config(config)
        super(DeepDense, self).__init__(graph, config)

    def set_up(self, tensor_inputs, inputs_placeholders, **kwargs):
        blocks = self._model.block
        prev = tensor_inputs
        with self._graph.as_default():
            with tf.variable_scope(self._scope + "_" + self._name_scope, reuse=self._reuse) as scope:
                for block in blocks:
                    activation_name = self.enum_activation_to_str(block.activation)
                    initializer_name = self.enum_initializer_to_str(block.initializer)
                    prev = tf.layers.dense(prev, block.num_units, activation=ACTIVATIONS[activation_name],
                                                                    kernel_initializer=INITIALIZERS[initializer_name])

        super(DeepDense, self).set_up(tensor_inputs, inputs_placeholders, last_block=prev, var_scope_obj=scope)


    def inference(self, session, runtime_tensor_inputs):
        if not isinstance(session, tf.Session):
            raise ValueError("Must pass in tf.Session")
        feed_dict = super()._produce_feed_dict(runtime_tensor_inputs)
        #with self._graph.as_default():
        return session.run(self._network, feed_dict=feed_dict)
