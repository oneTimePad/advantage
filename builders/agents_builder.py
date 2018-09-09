from protos.agents import agents_pb2
import tensorflow as tf
import agents
import builders.approximators_builder as approximators_builder

"""Builders for constructing the various Agents"""

def get_env_state_shape(environment):
    """Fetch env shape as list
        Args:
            environment: Environment or GymEnv

        Returns:
            shape as list for placeholder
    """
    state_shape = environment.reset().shape

    return [None] + list(state_shape)

class AgentBuilders:

    @staticmethod
    def build_DeepQAgent(graph, session, environment, config):
        """Builds the DeepQAgent
            Args:
                graph: tf.Graph
                session: tf.Session
                _environment: Environment or GymEnv
                config: the deep_q_agent protobuf object

            Returns:
                DeepQAgent
        """
        tgt_network_config = config.tgt_network
        policy_config = config.policy

        with graph.as_default():
            state_shape = get_env_state_shape(environment)

            tgt_placeholder = tf.placeholder(shape=state_shape, dtype=tf.float32, name="tgt_state_plh")

            policy_placeholder = tf.placeholder(shape=state_shape, dtype=tf.float32, name="policy_state_plh")


            tgt_network = approximators_builder.build(graph, tgt_network_config, tgt_placeholder)
            policy_network = approximators_builder.build(graph, policy_config, policy_placeholder)

            return agents.DeepQAgent(graph, session, policy_network, tgt_network, environment)


def build(agents_config, graph, environment):
    """ Builds an Agent based on configuration
            Args:
                agents_config: configuration from protobuf for agent
                environment: OpenAi Gym Gym object

            Returns:
                an Agent object
    """
    if not isinstance(agents_config, agents_pb2.Agents):
        raise ValueError("agents_config not of type agents_pb2.Approximator")

    agent_name_lower = agents_config.WhichOneof("agent")
    agent_name = agent_name_lower[0].capitalize() + agent_name_lower[1:]


    try:
        agent_builder = eval("AgentBuilders.build_" + agent_name)
    except AttributeError:
        raise ValueError("Model %s in configuration does not exist" % model)

    session = tf.Session(graph=graph)

    return agent_builder(graph, session, environment, getattr(agents_config, agent_name_lower)) #TODO there might be other configuration options for agents later on
