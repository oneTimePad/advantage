from advantage.protos.approximators.base import approximators_pb2
from advantage.utils.proto_parsers import parse_which_one_cls
from advantage.utils.tf_utils import ScopeWrap
import advantage.approximators

def build_approximator(graph, upper_scope, approximators_config, tensor_inputs, inputs_placeholders):
    """Constructs approximator
            Args:
                graph: TF graph
                upper_scope: upper level ScopeWrap
                approximators_config: configuration file
                tensor_inputs: the input tensor to the network
                inputs_placeholders: list of feed_dict elements or input placeholders
                    (might be equal to [tensor_inputs])

            Returns:
                DeepApproximator

            Raises:
                ValueError: invalid configuration or building params
    """
    if not isinstance(approximators_config, approximators_pb2.Approximators):
        raise ValueError("approximators_config not of type approximators_pb2.Approximator")

    approximator_name = parse_which_one_cls(approximators_config, "approximator")

    try:
        approximator_class = getattr(advantage.approximators, approximator_name)
    except AttributeError:
        raise ValueError("Approximator %s in configuration does not exist" % approximator_name)

    approximator_scope = ScopeWrap.build(upper_scope,
                                         approximators_config.name_scope)

    approximator = approximator_class(graph, approximators_config, approximator_scope)

    approximator.set_up(tensor_inputs, inputs_placeholders)

    return approximator
