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
