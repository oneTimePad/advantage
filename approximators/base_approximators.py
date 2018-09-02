from abc import ABCMeta

""" Approximators for Approximate Reinforcement Learning
    Allows the User to specify their own function approximators.

    In the future, this can be used to define various commonly used
    Deep Architectures for Deep RL
"""

class DeepApproximator(object):
    """ Interface for deep approximators for value, policies, meta-losses, etc...
    """
    __metaclass__ = ABCMeta

    def __init__(self, proto_config):
        pass

    @abstractmethod
    def set_up(self, tensor_inputs):
        """ TensorFlow construction of the approximator network
                Args:
                    tensor_inputs: Tensor inputs to the network

                Returns:
                    output tensor of the network
        """
        raise NotImplementedError()

    @abstractmethod
    def inference(self, runtime_tensor_inputs):
        """ Performs Runtime inference on the network. Usually setups a Session
                Args:
                    runtime_tensor_inputs: inputs to be passed in at runtime
        """
        raise NotImplementedError()

    @abstractmethod
    def gradients(self):
        """ Compute the network gradients for parameter update

            Returns:
                gradients tensor
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, runtime_inputs):
        """ Perform a network parameter update
                Args:
                    runtime_inputs: usually training batch inputs
        """
        raise NotImplementedError()


def multinomial_policy(tensor_inputs, axis):
    """Useful for constructing output layers for a multinomial policy settings
            Args:
                tensor_inputs: output of network to pass in
                axis: axis to softmax on
            Returns:
                output of multinomial policy
    """
    return tf.nn.softmax(tensor_inputs, axis=axis)


def binomial_policy(tensor_inputs):
    """Useful for constructing output layers for a binomial policy setting
            Args:
                tensor_inputs: output of network to pass in
            Returns:
                output of binomial policy
    """
    return tf.nn.sigmoid(tensor_inputs)


def value_function(tensor_inputs, num_actions):
    """Useful for constructing output of state-value function or action-value function
            Args:
                tensor_inputs: output of network to pass in
                num_actions: number of actions in a determinstic settings or 1 for value function
            Returns:
                regression layer output
    """
    return tf.layers.dense(tensor_inputs, num_actions)


def gaussian_policy(tensor_inputs, num_actions):
    """Useful for constructing output layers for continuous stochastic policy
            Args:
                tensor_inputs: output of network to pass in
                num_actions: shape of action tensor output
            Returns:
                mean and sigma gaussian policy
    """
    mean = tf.layers.dense(tensor_inputs, num_actions, activation=None)
    sigma = tf.layers.dense(tf.ones([1, num_actions]), activation=None, use_bias=False)
    return mean, sigma
