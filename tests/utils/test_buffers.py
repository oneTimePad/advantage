import unittest
import utils
from utils.buffers import ExperienceReplayBuffer
from utils.sarsa import Sarsa
import numpy as np

def pad(er_buffer, amt, **kwargs):
    for _ in range(amt):
        er_buffer.push(Sarsa.zero_initialize(**kwargs))

class TestExperienceReplayBuffer(unittest.TestCase):
    """ Tests for the various functionalities of the ExperienceReplayBuffer """
    def setUp(self):
        self.BUFFER_SIZE = 2
        self.sarsa_one = Sarsa.make(np.array([1.], dtype=np.float32),
                            np.array([2.], dtype=np.float32),
                            1.0,
                            False,
                            np.array([3.], dtype=np.float32))
        self.sarsa_two = Sarsa.make(np.array([2.]),
                            np.array([2.]),
                            1.0,
                            False,
                            np.array([3.]))

        self.sarsa_three = Sarsa.make(np.array([3.]),
                            np.array([2.]),
                            1.0,
                            False,
                            np.array([3.]))



    def test_push_comaptibility(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)

        ebuffer.push(self.sarsa_one)

        try:
            ebuffer.push(Sarsa.make(np.array([3.]),
                                    np.array([2.]),
                                    1.0,
                                    0.0,
                                    np.array([3., 4.])))
            self.assertEqual(1, 0)
        except ValueError as e:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)

    def test_push_valid(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)

        self.assertEqual(ebuffer.len, 0)

        ebuffer.push(self.sarsa_one)

        self.assertEqual(ebuffer.len, 1)

    def test_push_not_valid(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        self.assertEqual(ebuffer.len, 0)

        try:
            ebuffer.push(int(0))
            self.assertEqual(1, 0)
        except ValueError as e:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)



    def test_push_padding_not_valid(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        pad(ebuffer, 2)
        self.assertEqual(ebuffer.len, 2)

        try:
            ebuffer.push(int(0))
            self.assertEqual(1, 0)
        except ValueError:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)


    def test_sample_no_padding_no_sample_less_no_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        self.assertEqual(ebuffer.len, 0)

        ebuffer.push(self.sarsa_one)
        ebuffer.push(self.sarsa_two)

        batch = ebuffer.sample(self.BUFFER_SIZE)

        self.assertEqual(len(batch), self.BUFFER_SIZE)

        for act, exp in zip(batch, [self.sarsa_one, self.sarsa_two]):
            np.testing.assert_array_equal(act.state, exp.state)

        self.assertEqual(ebuffer.len, self.BUFFER_SIZE)

        batch_two = ebuffer.sample(1)

        self.assertEqual(len(batch), self.BUFFER_SIZE)

        act = batch_two[0]

        exp = self.sarsa_one

        np.testing.assert_array_equal(act.state, exp.state)

        ebuffer.push(self.sarsa_three)

        batch_three = ebuffer.sample(1)

        act = batch_three[0]

        exp = self.sarsa_two

        np.testing.assert_array_equal(act.state, exp.state)

    def test_sample_padding_no_sample_less_no_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        pad(ebuffer, 2, state_size=1,
                            action_size=1,
                            reward_size=None,
                            #done_size=None,
                            next_state_size=1)
        self.assertEqual(ebuffer.len, self.BUFFER_SIZE)

        batch = ebuffer.sample(self.BUFFER_SIZE)

        self.assertEqual(len(batch), self.BUFFER_SIZE)

        for b in batch:
            self.assertEqual(b.state, 0)

        ebuffer.push(self.sarsa_one)

        batch_two = ebuffer.sample(self.BUFFER_SIZE)

        self.assertEqual(batch_two[0].state, 0)
        np.testing.assert_array_equal(batch_two[1].state, self.sarsa_one.state)

    def test_sample_no_padding_less_no_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        self.assertEqual(ebuffer.len, 0)

        ebuffer.push(self.sarsa_one)
        ebuffer.push(self.sarsa_two)

        try:
            ebuffer.sample(self.BUFFER_SIZE + 1)
            self.assertEqual(1, 0)
        except ValueError:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)

        batch = ebuffer.sample(self.BUFFER_SIZE + 1, sample_less=True)

        self.assertEqual(len(batch), self.BUFFER_SIZE)

        for act, exp in zip(batch, [self.sarsa_one, self.sarsa_two]):
            np.testing.assert_array_equal(act.state, exp.state)

    def test_sample_padding_less_no_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        pad(ebuffer, 2)
        self.assertEqual(ebuffer.len, self.BUFFER_SIZE)

        try:
            ebuffer.sample(self.BUFFER_SIZE + 1)
            self.assertEqual(1, 0)
        except ValueError:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)

        batch = ebuffer.sample(self.BUFFER_SIZE + 1, sample_less=True)

        self.assertEqual(len(batch), self.BUFFER_SIZE)

        for b in batch:
            self.assertEqual(b.state, 0)



    def test_sample_no_padding_no_less_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)

        self.assertEqual(ebuffer.len, 0)

        ebuffer.push(self.sarsa_one)
        ebuffer.push(self.sarsa_two)

        batch = ebuffer.sample_and_pop(self.BUFFER_SIZE)

        for act, exp in zip(batch, [self.sarsa_one, self.sarsa_two]):
            np.testing.assert_array_equal(act.state, exp.state)

        self.assertEqual(ebuffer.len, 0)


    def test_sample_padding_no_less_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        pad(ebuffer, 2, state_size=1,
                            action_size=1,
                            reward_size=None,
                            #done_size=None,
                            next_state_size=1)
        self.assertEqual(ebuffer.len, self.BUFFER_SIZE)


        batch = ebuffer.sample_and_pop(self.BUFFER_SIZE)
        for b in batch:
            self.assertEqual(b.state, 0)

        self.assertEqual(ebuffer.len, 0)

        ebuffer.push(self.sarsa_one)

        self.assertEqual(ebuffer.len, 1)

        batch = ebuffer.sample_and_pop(1)

        np.testing.assert_array_equal(batch[0].state, self.sarsa_one.state)

        self.assertEqual(ebuffer.len, 0)

    def test_sample_no_padding_less_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)

        self.assertEqual(ebuffer.len, 0)

        ebuffer.push(self.sarsa_one)

        try:
            ebuffer.sample_and_pop(self.BUFFER_SIZE)
            self.assertEqual(1, 0)
        except ValueError:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)

        batch = ebuffer.sample_and_pop(self.BUFFER_SIZE, sample_less=True)

        self.assertEqual(ebuffer.len, 0)

        np.testing.assert_array_equal(batch[0].state, self.sarsa_one.state)

    def test_sample_padding_less_pop(self):
        ebuffer = ExperienceReplayBuffer(self.BUFFER_SIZE, Sarsa)
        pad(ebuffer, 2)
        self.assertEqual(ebuffer.len, self.BUFFER_SIZE)

        try:
            ebuffer.sample_and_pop(self.BUFFER_SIZE + 1)
            self.assertEqual(1, 0)
        except ValueError:
            self.assertEqual(1, 1)
        except Exception:
            self.assertEqual(1, 0)

        batch = ebuffer.sample_and_pop(self.BUFFER_SIZE + 1, sample_less=True)

        self.assertEqual(ebuffer.len, 0)

        for b in batch:
            self.assertEqual(b.state, 0)



unittest.main()
