from protos.approximators import approximators_pb2
import approximators

def build(approximators_config):
    """Constructs approximator
            Args:
                approximators_config: configuration file
            Returns:
                DeepApproximator
            Raises:
                ValueError: invalid configuration or building params
    """
    if not isinstance(approximators_config, approximators_pb2.Approximators):
        raise ValueError("approximators_config not of type approximators_pb2.Approximator")

    model_name = approximators_config.WhichOneof("model")
    model_name = model_name[0].capitalize() + model_name[1:]

    try:
        model_class = eval("approximators." + model_name)
    except AttributeError:
        raise ValueError("Model %s in configuration does not exist" % model)

    return model_class()
