from protos.agents import agents_pb2
from functools import partial
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


def function_attr(fn, **kwargs):
    """Add a set of keyword attrs to a function"""
    for k, v  in kwargs.items():
        setattr(f, k, v)
    return fn


class AgentBuilders:

    @staticmethod
    def build_DeepQAgent(agent, graph, environment, config):
        """Builds the DeepQAgent
            Args:
                agent: the agent to build,
                graph: tf.Graph
                environment: agent environment
                config: the deep_q_agent protobuf object

            Returns:
                DeepQAgent
        """
        tgt_network_config = config.tgt_network
        policy_config = config.policy
        epsilon = config.epsilon
        with graph.as_default():
            state_shape = get_env_state_shape(environment)


            tgt_state_plh = tf.placeholder(shape=state_shape, dtype=tf.float32, name="tgt_state_plh")

            policy_state_plh = tf.placeholder(shape=state_shape, dtype=tf.float32, name="policy_state_plh")


            tgt_network = approximators_builder.build(graph, tgt_network_config, tgt_state_plh, [tgt_state_plh])
            policy_network = approximators_builder.build(graph, policy_config, policy_state_plh, [policy_state_plh])

            return agent(policy_network, tgt_network, epsilon)


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
        agent_builder = eval("AgentBuilders.build_" + agent_name) #TODO use getattr
    except AttributeError:
        raise ValueError("Agent %s in configuration does not exist" % agent_name)

    try:
        agent = getattr(agents, agent_name)
    except AttributeError:
        raise ValueError("Agent %s in configuration does not exist" % agent_name)

    session = tf.Session(graph=graph)

    agent = partial(agent,
                    graph,
                    session,
                    environment,
                    agents_config.discount_factor)

    return agent_builder(agent,
                         graph,
                         environment,
                         getattr(agents_config, agent_name_lower)) #TODO there might be other configuration options for agents later on
