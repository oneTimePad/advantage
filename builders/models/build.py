from functools import partial
import tensorflow as tf
from advantage.protos.models.base import models_pb2
from advantage.utils.tf_utils import ScopeWrap
from advantage.utils.proto_parsers import parse_which_one_cls, parse_which_one
from advantage.builders.agents import build_agent
from advantage.builders.models.builders import ModelBuilders
import advantage.models

"""Build function for constructing the various Models
"""


def build_model(models_config, environment, is_training):
    """ Builds a Model based on configuration
            Args:
                models_config: configuration from protobuf
                environment: OpenAI Gym Gym object
                is_training: whether we are building a model
                    for training

            Returns:
                a Model object
    """
    if not isinstance(models_config, models_pb2.Models):
        raise ValueError("models_config not of type models_pb2.Models")

    model_name = parse_which_one_cls(models_config, "model")

    try:
        model_builder = getattr(ModelBuilders, "build_" + model_name)
    except AttributeError:
        raise ValueError("Model %s in configuration does not exist" % model_name)

    try:
        model = getattr(advantage.models, model_name)
    except AttributeError:
        raise ValueError("Model %s in configuration does not exist" % model_name)

    graph = tf.Graph()

    model_scope = ScopeWrap(graph,
                            models_config.name_scope,
                            models_config.reuse)

    agents_config = models_config.agent

    agent = build_agent(graph,
                        environment,
                        model_scope,
                        is_training,
                        agents_config)

    model = partial(model,
                    graph,
                    environment,
                    model_scope,
                    agent)

    specific_model_config = getattr(models_config,
                                    parse_which_one(models_config, "model"))

    return model_builder(model,
                         environment,
                         model_scope,
                         agent,
                         specific_model_config,
                         is_training)
