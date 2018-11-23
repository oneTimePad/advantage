from google.protobuf.json_format import MessageToDict
import tensorflow as tf
from advantage.builders.buffers import build_buffer



"""Builders for constructing the various Models
"""


class ModelBuilders:
    """ Static methods to construct
    specific types of Models.
    """

    def __new__(cls):
        raise NotImplementedError("Can't instantiate")

    #pylint: disable=C0103
    #reason-disabled: method needs to contain propert cls name
    @staticmethod
    def build_DeepQModel(model,
                         environment,
                         model_scope,
                         agent,
                         config,
                         is_training):
        """ Constructs the DeepQModel obj based on configurations
                Args:
                    model : DeepQModel
                    environment: associated Environment from main config
                    model_scope: scope that can be used to add variables
                    agent: DeepQAgent
                    config: DeepQAgent protobuf config
                    is_training: whether we are building a model for training
                        allows building to take certain actions based on
                        whether it is in training mode or not

                Returns:
                    DeepQModel
        """
        buffers_config = config.buffer
        experience_replay_buffer = build_buffer(buffers_config)

        sarsa_as_dict = MessageToDict(config.sarsa)
        sarsa_attrs_to_normalize = filter(lambda x: x in sarsa_as_dict.keys(),
                                          ["state", "action", "reward"])

        return model(experience_replay_buffer,
                     sarsa_attrs_to_normalize,
                     config.improve_target_modulo,
                     config.iterations_of_improvement,
                     config.batch_size,
                     config.train_sample_less)
