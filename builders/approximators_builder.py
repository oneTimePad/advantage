from protos.approximators import approximators_pb2
import approximators

def build(graph, approximators_config, inputs):
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

    model = model_class(graph, getattr(approximators_config, approximators_config.WhichOneof("model")),  approximators_config.name_scope, approximators_config.reuse)
    model.set_up(inputs)
    return model
