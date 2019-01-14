import gin
from advantage.agents.base.base_agents import DecoupledValueGradientAgent
from advantage.utils.gin_utils import gin_bind_init
from advantage.agents.objectives import Objectives
from advantage.agents.policies import Policies
from advantage.buffers.replay_buffers import RandomizedReplayBuffer
from advantage.buffers.elements import Sarsa

@gin.configurable(blacklist=["scope", "environment"])
class DeepQNetworks(DecoupledValueGradientAgent):
    """ Implements the DeepQNetworks Agent. The DQN Agent
    utilizes two networks to stabilize Q-Learning for Deep RL approximators.
    A ExperienceReplayBuffer is utilize to allow for the I.I.D necessary condition
    for Neural Networks.
    """

    name_scope = "deep_q_networks"

    def pre_iteration(self):
        pass

    def post_iteration(self):
        pass

    @classmethod
    def gin_wire(cls):
        gin_bind_init(cls,
                      Policies.VALUE,
                      "use_epsilon",
                      True)
        gin_bind_init(cls,
                      Objectives.DECOUPLED_VALUE_GRADIENT,
                      "replay_buffer",
                      RandomizedReplayBuffer(Sarsa))
        super().gin_wire()
