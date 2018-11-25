
from functools import partial
import tensorflow as tf
from advantage.utils.proto_parsers import parse_which_one_cls, parse_which_one
from advantage.utils.tf_utils import ScopeWrap
from advantage.protos.agents.base import agents_pb2
from advantage.builders.approximators import build_approximator
from advantage.builders.agents.builders import AgentBuilders
import advantage.agents

"""Build function for constructing the various Agents
"""

def build_agent(graph, environment, upper_scope, is_training, agents_config):
    """ Builds an Agent based on configuration
            Args:
                graph: TF graph
                environment: OpenAI Gym `Gym` object
                upper_scope: ScopeWrap from higher level (Models)
                agents_config: configuration from protobuf for agent

            Returns:
                an Agent object
    """
    if not isinstance(agents_config, agents_pb2.Agents):
        raise ValueError("agents_config not of type agents_pb2.Agents")

    agent_name = parse_which_one_cls(agents_config, "agent")

    try:
        agent_builder = getattr(AgentBuilders, "build_" + agent_name)
    except AttributeError:
        raise ValueError("Agent %s in configuration does not exist" % agent_name)

    try:
        agent = getattr(advantage.agents, agent_name)
    except AttributeError:
        raise ValueError("Agent %s in configuration does not exist" % agent_name)

    agent_scope = ScopeWrap.build(upper_scope,
                                  agents_config.name_scope)

    agent = partial(agent,
                    environment,
                    graph,
                    agent_scope,
                    agents_config.discount_factor)

    build_approximator_partial = partial(build_approximator,
                                         graph,
                                         agent_scope)

    specific_agent_config = getattr(agents_config,
                                    parse_which_one(agents_config, "agent"))

    return agent_builder(agent,
                         environment,
                         agent_scope,
                         build_approximator_partial,
                         specific_agent_config,
                         is_training)
