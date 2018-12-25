from abc import ABCMeta, abstractmethod
import collections


""" Base Buffer Interfaces
"""

class Buffer(metaclass=ABCMeta):
    """ Represents a Buffer to hold
    environment `Elements`
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


class ReplayBuffer(Buffer):
    """ Stores Elements and Replays them
    (allows them to be fetched latter on)
    """

    def __init__(self, buffer_size):
        """
            Args:
                buffer_size: size of deque
        """

        self._buffer_size = buffer_size

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
        if self._cur_buffer_size < self._buffer_size:
            self._cur_buffer_size += 1

        self._buffer.append(item)

    def sample(self, batch_size, sample_less=False):
        """ Samples a batch of  `Elements` from buffer
            Args:
                batch_size: number of samples to collect
                sample_less: whether to allow sampling less than requested amount

            Raises:
                ValueError: invalid amount of samples requested
                    and sample_less is False

            Returns:
                list of a Sarsa tuples
        """
        if not sample_less and batch_size > self._cur_buffer_size:
            raise ValueError("Specify sample_less=True to retrieve less than specified amount")

        if not sample_less:
            batch = list(self._buffer)[:batch_size]
        else:
            batch = list(self._buffer)

        return batch

    def sample_and_pop(self, batch_size, sample_less=False):
        """ Sample a batch of `Elements` from buffer and  remove them
                Args:
                    batch_size: number of samples to collect
                    sample_less: whether to allow sampling less
                        than requested amount

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

        batch_size = batch_size if batch_size <= self._cur_buffer_size else self._cur_buffer_size

        self._cur_buffer_size -= batch_size

        return batch

    def sample_batches(self,
                       batch_size,
                       num_batches=None,
                       sample_less=False,
                       pop=True):
        """ Generator for sampling batches of `Elements` from buffer

                Args:
                    batch_size: number of samples to collect
                        per yield
                    num_batches: number of batches to fetch
                        `None` means fetch until buffer is empty
                    sample_less: whether to allow sampling less
                        than requested amount
                    pop: whether to remove elements from batch
        """
        num_batches_in_buffer = self.len // batch_size

        if num_batches > num_batches_in_buffer:
            num_batches = None # requesting more than is in buffer, mean take all of it

        if not num_batches:
            # check if there is an remainder in the buffer
            num_batches = num_batches_in_buffer + int((self.len % batch_size) != 0)

        sample_func = self.sample_and_pop if pop else self.sample
        for _ in range(num_batches):
            yield sample_func(batch_size)

        # whether we should return the remainder
        if sample_less and num_batches > num_batches_in_buffer:
            yield sample_func(batch_size, sample_less=True)
