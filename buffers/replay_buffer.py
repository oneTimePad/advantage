from abc import ABCMeta, abstractmethod

""" Replay Buffer interface
"""


class ReplayBuffer(metaclass=ABCMeta):
    """ Represents a Replay Buffer
    for Off-Policy learning
    """

    @abstractmethod
    @property
    def buffer_size(self):
        """ propety for Limit size
        of buffer
        """
        raise NotImplementedError()

    @abstractmethod
    @property
    def len(self):
        """ property for current
        buffer length
        """
        raise NotImplementedError()

    @abstractmethod
    def push(self, item):
        """Appends and element to the buffer
            Args:
                item: item to added
        """
        raise NotImplementedError()

    @abstractmethod
    def random_sample_and_pop(self, batch_size, sample_less=False):
        """ Sample a random batch of Sarsa tuple and remove them
                Args:
                    batch_size: number of samples to collect
                    sample_less: whether to allow sampling less than requested amount

                Raises:
                    ValueError: invalid amount of samples requested
                        and sample_less is False

                Returns:
                    list of Elements
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size, sample_less=False):
        """Sample a determinstic batch of Sarsa tuples
            Args:
                batch_size: number of samples to collect
                sample_less: whether to allow sampling less than requested amount

            Raises:
                ValueError: invalid amount of samples requested
                    and sample_less is False

            Returns:
                list of Elements
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_and_pop(self, batch_size, sample_less=False):
        """ Sample a determinstic batch of Sarsa tuple and remove them
                Args:
                    batch_size: number of samples to collect
                    sample_less: whether to allow sampling less than requested amount

                Raises:
                    ValueError: invalid amount of samples requested
                        and sample_less is False

                Returns:
                    list of Elements
        """
        raise NotImplementedError()
