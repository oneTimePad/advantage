import collections


""" Sarsa tuple element to put in experience replay buffer """
Sarsa = collections.namedtuple("Sarsa", ["state", "action", "reward", "done", "next_state", "next_action"])


class ExperienceReplayBuffer(object):
    """Allows for collecting various SARSA tuples taken by an agent.
    This buffer is commonly used in many approximate RL algorithms to
    allow for I.I.D data to traing non-linear approximators (i.e. NN)
    """

    def __init__(self, buffer_size, add_padding=False):
        """
            Args:
                buffer_size: size of deque
                add_padding: whether to fill deque up with padded values
        """
        self._buffer_size =  buffer_size

        self._cur_buffer_size = 0

        if not add_padding:
            self._buffer = collections.deque([], maxlen=buffer_size)
        else:
            # construct empty Sarsa elements for paddings
            NUM_TUPLE_ELEMENTS_FOR_SARSA = 6
            padding_elements = [Sarsa(*([None] * NUM_TUPLE_ELEMENTS_FOR_SARSA))] * buffer_size
            self._buffer = collections.deque(padding_elements, maxlen=buffer_size)
            self._cur_buffer_size = buffer_size

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def len(self):
        return self._cur_buffer_size

    def push(self, sarsa):
        """Appends and element to the buffer
            Args:
                sarsa: a Sarsa tuple

            Raises:
                ValueError: invalid tuple type
        """
        if not isinstance(sarsa, Sarsa):
            raise ValueError("Must push in Sarsa NamedTuple not %s" % str(type(sarsa)))

        self._cur_buffer_size = self._cur_buffer_size + 1 if self._cur_buffer_size < self._buffer_size else self._cur_buffer_size

        self._buffer.append(sarsa)

    def random_sample(self, batch_size, sample_less=False):
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
        return self.sample(batch_size, sample_less=sample_less)

    def random_sample_and_pop(self, batch_size, sample_less=False):
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
        return self.sample_and_pop(batch_size, sample_less=sample_less)

    def sample(self, batch_size, sample_less=False):
        """Sample a determinstic batch of Sarsa tuples
            Args:
                batch_size: number of samples to collect
                sample_less: whether to allow sampling less than requested amount

            Raises:
                ValueError: invalid amount of samples requested
                    and sample_less is False

            Returns:
                list of Sarsa tuples
        """

        if not sample_less and batch_size > self._cur_buffer_size:
            raise ValueError("Specify sample_less=True to retrieve less than specified amount")

        if not sample_less:
            batch = list(self._buffer)[:batch_size]
        else:
            batch = list(self._buffer)

        return batch

    def sample_and_pop(self, batch_size, sample_less=False):
        """ Sample a determinstic batch of Sarsa tuple and remove them
                Args:
                    batch_size: number of samples to collect
                    sample_less: whether to allow sampling less than requested amount

                Raises:
                    ValueError: invalid amount of samples requested
                        and sample_less is False

                Returns:
                    list of Sarsa tuples
        """
        if not sample_less and batch_size > self._cur_buffer_size:
            raise ValueError("Specify sample_less=True to retrieve less than specified amount")

        batch = []
        if not sample_less:
            for _ in range(batch_size):
                batch.append(self._buffer.popleft())
        else:
            for _ in range(self._cur_buffer_size):
                batch.append(self._buffer.popleft())

        self._cur_buffer_size -= batch_size if batch_size <= self._cur_buffer_size else self._cur_buffer_size

        return batch
