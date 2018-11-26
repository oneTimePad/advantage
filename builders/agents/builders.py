import tensorflow as tf

"""Builders for constructing the various Agents
"""

class AgentBuilders:
    """ Contains various builders
    for constructing agents
    """

    def __new__(cls):
        raise NotImplementedError("Can't instantiate")

    # pylint: disable=C0103
    # reason-disabled: naming is done on purpose to select methods
    @staticmethod
    def build_DeepQAgent(agent,
                         environment,
                         agent_scope,
                         build_approximator,
                         config,
                         is_training):
        """Builds the DeepQAgent
            Args:
                agent: the agent to build,
                environment: agent environment
                agent_scope: scope that can be used to add variables
                config: the deep_q_agent protobuf object
                is_training: whether are building for training

            Returns:
                DeepQAgent
        """
        tgt_network_config = config.tgt_network
        policy_config = config.policy

        state_shape = environment.tf_state_shape

        with agent_scope():
            if is_training:
                tgt_state_plh = tf.placeholder(shape=state_shape,
                                               dtype=tf.float32,
                                               name="tgt_state_plh")

            policy_state_plh = tf.placeholder(shape=state_shape,
                                              dtype=tf.float32,
                                              name="policy_state_plh")
        epsilon = None
        tgt_network = None
        if is_training:
            from advantage.utils.value_agent import Epsilon

            tgt_network = build_approximator(tgt_network_config,
                                             tgt_state_plh,
                                             [tgt_state_plh])

            epsilon = Epsilon.from_config(config.epsilon)

        policy_network = build_approximator(policy_config,
                                            policy_state_plh,
                                            [policy_state_plh])

        return agent(policy_network,
                     tgt_network,
                     epsilon)
