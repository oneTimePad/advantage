import unittest
from src.environments.mock_environment import MockEnvironment
import numpy as np

class TestMockEnvironment(unittest.TestCase):

    def setUp(self):
        self._mock = MockEnvironment(initial_state=np.array([1.0, 1.0]),
                            state_step=np.array([2.0, 2.0]))

    def test_step(self):
        state, action, done = self._mock.step(1)
        #self.assertEqual(state, np.array([3.0, 3.0]))
        np.testing.assert_array_equal(state, np.array([3.0, 3.0]))

    def test_reset(self):
        state, action, done = self._mock.step(1)

        state = self._mock.reset()
        np.testing.assert_array_equal(state, np.array([1.0, 1.0]))
unittest.main()
