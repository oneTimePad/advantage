
""" Constructs Environment's from protobuf config files
"""

class EnvironmentBuilders:
    """ Static methods to construct
    specific types of Environments.
    """
    def __init__(self):
        raise NotImplementedError("Can't instantiate")

    #pylint: disable=C0103
    #reason-disabled: method needs to contain propert cls name
    @staticmethod
    def build_GymEnvironment(env, config):
        """ Builds OpenAI Gym environment
                Args:
                    env: Environment cls
                    config: protobuf config obj

                Returns:
                    GymEnvironment
        """
        return env(config.name)
