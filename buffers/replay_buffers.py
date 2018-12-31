from abc import ABCMeta, abstractmethod
import numpy as np
from advantage.buffers.elements import build_idx_to_attr_map

""" This module contains `Replay Buffers`. These
buffer take in `Element`s and replay them to the agent
at a latter time
"""

class Buffer(metaclass=ABCMeta):
    """ Represents a Buffer to hold
    environment `Elements`
    """

    @abstractmethod
    @property
    def max_buffer_size(self):
        """ propety for Limit size
        of buffer
        """
        raise NotImplementedError()

    @abstractmethod
    def push(self, element):
        """Appends and element to the buffer
            Args:
                element: `Element` to add
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size):
        """Sample a determinstic batch of Sarsa tuples
            Args:
                batch_size: number of samples to collect

            Returns:
                list of Elements
        """
        raise NotImplementedError()


# Reference: https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
class ReplayBuffer(Buffer):
    """ Stores Elements and Replays them
    (allows them to be fetched latter on)
    """

    def __init__(self, element_cls, buffer_size, init=None):
        """
            Args:
                element_cls: subclass of `Element`
                buffer_size: size of deque
                init: initialize from another buffer
        """
        self._element_cls = element_cls

        self._max_buffer_size = buffer_size

        self._cur_buffer_size = 0

        self._index = 0

        self._buffer = np.empty(shape=buffer_size, dtype=np.float32)

        if init: # set values from `init`
            self._buffer[:len(init)] = init.sample(len(init))

        self._element_idx_to_attr_map = None
        self._element_len = None

    def __len__(self):
        """ property for the number of elements
        `_cur_buffer_size`
        """
        return self._cur_buffer_size

    @property
    def max_buffer_size(self):
        """ property for `_max_buffer_size`
        """
        return self._max_buffer_size

    def push(self, element):
        if self._cur_buffer_size < self._max_buffer_size:
            self._cur_buffer_size += 1

        if not self._element_idx_to_attr_map:
            self._element_len, self._element_idx_to_attr_map = build_idx_to_attr_map(element)

        element.idx_to_attr_map = self._element_idx_to_attr_map
        element.len = self._element_len

        self._buffer[self._index] = element
        self._cur_buffer_size = min(self._cur_buffer_size + 1, self._max_buffer_size)
        self._index = (self._index + 1) % self._max_buffer_size

    def sample_from_indices(self, indices):
        """ Samples batch of `Elements` from buffer
        using the list of `indices`.
            Args:
                indices: list of indices to select from buffer

            Raise:
                IndexError: if an index is out of bounds

            Returns:
                `Element`
        """
        return self._element_cls.from_numpy(self._buffer[indices],
                                            self._element_idx_to_attr_map,
                                            self._element_len)

    def sample(self, batch_size):
        return self._element_cls.from_numpy(self._buffer[np.arange(batch_size)],
                                            self._element_idx_to_attr_map,
                                            self._element_len)

    def sample_batches(self,
                       batch_size,
                       num_batches):
        """ Generator for sampling batches of `Elements` from buffer

                Args:
                    batch_size: number of samples to collect
                        per yield
                    num_batches: number of batches to fetch
                        `None` means fetch until buffer is empty
        """

        for _ in range(num_batches):
            yield self.sample(batch_size)

    def clear(self):
        """ Clears out buffer
        """
        old_buffer = self._buffer
        self._buffer = np.empty(shape=self.max_buffer_size, dtype=np.float32)
        del old_buffer

class RandomizedReplayBuffer(ReplayBuffer):
    """Allows for collecting various `Elements` made by an agent.
    This buffer is commonly used in many approximate RL algorithms to
    allow for I.I.D data to training non-linear approximators (i.e. NN)
    A.K.A Experience Replay Buffer
    """

    def sample(self, batch_size):
        """Randomly samples a batch of Sarsa tuples
            Args:
                batch_size: number of samples to collect

            Returns:
                list of a `Elements`
        """
        indices = np.random.randint(len(self), size=batch_size)
        return super().sample_from_indices(indices)
