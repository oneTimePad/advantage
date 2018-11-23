from advantage.protos import environments_pb2
from advantage.utils.proto_parsers import parse_which_one_cls, parse_which_one
from advantage.builders.environments.builders import EnvironmentBuilders
import advantage.environments

"""Build function for constructing the various Models
"""

def build_environment(environments_config):
    """ Builds an  Environment based on configuration
            Args:
                environments_config: configuration from protobuf

            Returns:
                Environment object
    """
    if not isinstance(environments_config, environments_pb2.Environments):
        raise ValueError("environments_config not of type environments_pb2.Environments")

    env_name = parse_which_one_cls(environments_config, "environment")

    try:
        env_builder = getattr(EnvironmentBuilders, "build_" + env_name)
    except AttributeError:
        raise ValueError("Environment `%s` in configuration does not exist" % env_name)

    try:
        env = getattr(advantage.environments, env_name)
    except AttributeError:
        raise ValueError("Environment `%s` in configuration does not exist" % env_name)

    specific_env_config = getattr(environments_config,
                                  parse_which_one(environments_config, "environment"))

    return env_builder(env, specific_env_config)
