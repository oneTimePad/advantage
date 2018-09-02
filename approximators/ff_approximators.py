from base_approximators import DeepApproximator

""" Feed-forward Deep NN approximators """

class DeepConvolutional(DeepApproximator):
    """ Convolutional Network"""

    def __init__(self, proto_config):
        pass
        super(DeepConvolutional, self).__init__(proto_config)

    def set_up(self, tensor_inputs):
        pass

    def inference(self, runtime_tensor_inputs):
        pass

    def gradients(self):
        pass

    def update(self, runtime_inputs):
        pass

class DeepDense(DeepApproximator):
    """ Fully Connected Network"""

    def __init__(self, proto_config):
        pass
        super(DeepDense, self).__init__(proto_config)

    def set_up(self, tensor_inputs):
        pass

    def inference(self, runtime_tensor_inputs):
        pass

    def gradients(self):
        pass

    def update(self, runtime_inputs):
        pass
