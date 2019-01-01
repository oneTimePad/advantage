import tensorflow as tf
import gin
from advantage.agents.base.base_agents import DecoupledValueGradientAgent
from advantage.utils.value_agent import decayed_epsilon
from advantage.utils.gin_utils import gin_bind_init
from advantage.agents.objectives import Objectives
from advantage.buffers.replay_buffers import RandomizedReplayBuffer


@gin.configurable(blacklist=["scope", "environment"])
class DeepQNetworks(DecoupledValueGradientAgent):
    """ Implements the DeepQNetworks Agent. The DQN Agent
    utilizes two networks to stabilize Q-Learning for Deep RL approximators.
    A ExperienceReplayBuffer is utilize to allow for the I.I.D necessary condition
    for Neural Networks.
    """

    name_scope = "deep_q_networks"

    # pylint: disable=too-many-arguments
    # reason-disabled: argument format acceptable
    def __init__(self,
                 scope,
                 environment,
                 epsilon):

        self._epsilon = epsilon

        super().__init__(scope, environment)

    @classmethod
    def gin_wire(cls):
        gin_bind_init(cls,
                      "RandomizedReplayBuffer.element_cls",
                      Sarsa)
        gin_bind_init(cls,
                      Objectives.DECOUPLED_VALUE_GRADIENT,
                      "replay_buffer",
                      RandomizedReplayBuffer())
        super().gin_wire()
