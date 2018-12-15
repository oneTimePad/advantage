import random
from advantage.buffers.base.base_buffers import ReplayBuffer

class ExperienceReplayBuffer(ReplayBuffer):
    """Allows for collecting various `Elements` made by an agent.
    This buffer is commonly used in many approximate RL algorithms to
    allow for I.I.D data to training non-linear approximators (i.e. NN)
    """

    def sample(self, batch_size, sample_less=False):
        """Randomly samples a batch of Sarsa tuples
            Args:
                batch_size: number of samples to collect
                sample_less: whether to allow sampling less than requested amount

            Raises:
                ValueError: invalid amount of samples requested
                    and sample_less is False

            Returns:
                list of a Sarsa tuples
        """
        random.shuffle(self._buffer)
        return super().sample(batch_size, sample_less=sample_less)

    def sample_and_pop(self, batch_size, sample_less=False):
        """ Sample a random batch of Sarsa tuple and remove them
                Args:
                    batch_size: number of samples to collect
                    sample_less: whether to allow sampling less than requested amount

                Raises:
                    ValueError: invalid amount of samples requested
                        and sample_less is False

                Returns:
                    list of Sarsa tuples
        """
        random.shuffle(self._buffer)
        return super().sample_and_pop(batch_size, sample_less=sample_less)
