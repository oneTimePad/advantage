

"""Builders for constructing the various Buffers
"""

class BufferBuilders:
    """ Static methods to construct
    specific types of Buffers.
    """

    def __new__(cls):
        raise NotImplementedError("Can't instantiate")

    # pylint: disable=C0103
    # reason-disabled: naming is done on purpose to select methods
    @staticmethod
    def build_ExperienceReplayBuffer(buffer, config):
        """ Constructs the ExperienceReplayBuffer
                Args:
                    buffer : the buffer class
                    config: specific buffer configuration

                Returns:
                    ExperienceReplayBuffer
        """
        return buffer()
