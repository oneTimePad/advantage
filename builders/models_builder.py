from protos.models import models_pb2
from functools import partial
from agents import DeepQAgent
import tensorflow as tf
import models
import builders.agents_builder as agents_builder
import builders.buffers_builder as buffers_builder




class ModelBuilders:

    @staticmethod
    def build_DeepQModel(model, graph, environment, agent, config):


        if not isinstance(agent, DeepQAgent):
            raise ValueError("DeepQModel expects agent DeepQAgent in protobuf config")

        buffers_config = config.buffer
        experience_replay_buffer = buffers_builder.build(buffers_config)

        sarsa_config = config.sarsa
        sarsa_attrs_to_normalize = []
        if sarsa_config.normalizeState:
            sarsa_attrs_to_normalize.append("state")
        if sarsa_config.normalizeAction:
            sarsa_attrs_to_normalize.append("action")
        if sarsa_config.normalizeReward:
            sarsa_attrs_to_normalize.append("reward")

        sarsa_attrs_to_normalize = tuple(sarsa_attrs_to_normalize)

        return model(experience_replay_buffer,
                        sarsa_attrs_to_normalize,
                        config.improve_target_modulo,
                        config.iterations_of_improvement,
                        config.batch_size,
                        config.train_sample_less)



def build(models_config, environment):
    """ Builds a Model based on configuration
            Args:
                models_config: configuration from protobuf
                environment: OpenAI Gym Gym object

            Returns:
                a Model object
    """
    if not isinstance(models_config, models_pb2.Models):
        raise ValueError("models_config not of type models_pb2")

    model_name_lower = models_config.WhichOneof("model")
    model_name = model_name_lower[0].capitalize() + model_name_lower[1:]

    try:
        model_builder = eval("ModelBuilders.build_" + model_name) #TODO use getattr
    except AttributeError:
        raise ValueError("Model %s in configuration does not exist" % agent_name)

    try:
        model = getattr(models, model_name)
    except AttributeError:
        raise ValueError("Model %s in configuration does not exist" % model_name)

    graph = tf.Graph()

    agents_config = models_config.agent

    agent = agents_builder.build(agents_config, graph, environment)

    model = partial(model,
                    graph, environment, agent,
                    models_config.steps_to_run_for,
                    models_config.improve_policy_modulo,
                    models_config.steps_for_act_iter)

    return model_builder(model,
                        graph,
                        environment,
                        agent,
                        getattr(models_config, model_name_lower))
