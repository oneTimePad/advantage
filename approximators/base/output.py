from functools import partial
import tensorflow as tf
from advantage.utils.proto_parsers import parse_which_one, proto_to_dict

""" Approximators can utilize various `outputs`
which represent different approximator policies
"""

class Output:
    """ Holds the various output polices that
    an approximator can use
    """
    @classmethod
    def from_config(cls, config):
        """ Parses out the Output for this Approximator
                Args:
                    config: the approximators_pb2 object
                Returns:
                    the partial fn for the output to be applied to the
                    base network output
        """
        selected_output = parse_which_one(config, "output")
        output = getattr(cls, selected_output)
        output_config = getattr(config, selected_output)

        return partial(output, **proto_to_dict(output_config))

    @staticmethod
    def multinomial(tensor_inputs, axis):
        """Useful for constructing output layers for a multinomial policy settings
                Args:
                    tensor_inputs: output of network to pass in
                    axis: axis to softmax on
                Returns:
                    output of multinomial policy
        """
        return tf.nn.softmax(tensor_inputs, axis=axis)

    @staticmethod
    def binomial(tensor_inputs):
        """Useful for constructing output layers for a binomial policy setting
                Args:
                    tensor_inputs: output of network to pass in
                Returns:
                    output of binomial policy
        """
        return tf.nn.sigmoid(tensor_inputs)

    @staticmethod
    def value(tensor_inputs, **kwargs):
        """Useful for constructing output of state-value function or action-value function
                Args:
                    tensor_inputs: output of network to pass in
                    **kwargs:
                        num_actions: number of actions in a determinstic settings or
                            1 for value function
                Returns:
                    regression layer output
        """
        shape = tensor_inputs.get_shape()
        # conv2d needs 4D input
        if len(shape) != 4:
            for _ in range(4 - len(shape)):
                tensor_inputs = tf.expand_dims(tensor_inputs, axis=1)
        shape = tensor_inputs.get_shape()
        kernel_h = int(shape[1])
        kernel_w = int(shape[2])
        conv = tf.layers.conv2d(tensor_inputs,
                                filters=kwargs["num_actions"],
                                kernel_size=[kernel_h, kernel_w],
                                activation=None,
                                name="value")
        # remove extra dimensions added
        return tf.squeeze(conv, axis=[1, 2])

    @staticmethod
    def gaussian(tensor_inputs, **kwargs):
        """Useful for constructing output layers for continuous stochastic policy
                Args:
                    tensor_inputs: output of network to pass in
                    **kwargs:
                        num_actions: shape of action tensor output
                Returns:
                    mean and sigma gaussian policy
        """
        mean = tf.layers.dense(tensor_inputs, kwargs["num_actions"], activation=None)
        sigma = tf.layers.dense(tf.ones([1, kwargs["num_actions"]]),
                                kwargs["num_actions"],
                                activation=None,
                                use_bias=False)
        return mean, sigma
