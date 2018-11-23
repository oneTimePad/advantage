import collections
import random

# TODO maybe make an Interface for buffers?
class ExperienceReplayBuffer(object):
    """Allows for collecting various SARSA(usually but not required) tuples taken by an agent.
    This buffer is commonly used in many approximate RL algorithms to
    allow for I.I.D data to traing non-linear approximators (i.e. NN)

    """

    def __init__(self, buffer_size):
        """
            Args:
                buffer_size: size of deque
                buffer_type: type of elements to be pushed to buffer
        """

        self._buffer_size =  buffer_size

        #self._buffer_type = buffer_type -> keeping type safe seems unecessary
        """
        if not hasattr(buffer_type, "are_compatible"):
            raise ValueError("buffer_type must implement 'are_compatible' classmethod")
        self._are_compatible = buffer_type.are_compatible

        self._template_item = None
        """
        self._cur_buffer_size = 0

        self._buffer = collections.deque([], maxlen=buffer_size)


    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def len(self):
        return self._cur_buffer_size

    def push(self, item):
        """Appends and element to the buffer
            Args:
                item: item to add of type 'buffer_type'
        """

        """
        if not isinstance(item, self._buffer_type):
            raise ValueError("item must be of type %s, but argument is of type %s" %(self._buffer_type, type(item)))
        """
        """
        # first element determines compatbility
        if self._template_item is None:
            self._template_item = item

        if not self._are_compatible(self._template_item, item):
            raise ValueError("item is not compatible other items in buffer")
        """
        self._cur_buffer_size = self._cur_buffer_size + 1 if self._cur_buffer_size < self._buffer_size else self._cur_buffer_size

        self._buffer.append(item)

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
